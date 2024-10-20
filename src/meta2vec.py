# src/meta2vec.py

import os
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
import xml.etree.ElementTree as ET
from github import Github

class Meta2Vec:
    def __init__(self, repo_path, vector_size, github_token=None):
        self.repo_path = repo_path
        self.vector_size = vector_size
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.github_token = github_token
        self.github_client = Github(github_token) if github_token else None

    def generate(self):
        metadata = self.extract_metadata()
        processed_metadata = self.preprocess_text(metadata)
        
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(processed_metadata)]
        model = Doc2Vec(documents, vector_size=self.vector_size, window=5, min_count=1, workers=4, epochs=100)
        
        return model.infer_vector(processed_metadata)

    def extract_metadata(self):
        metadata = {}
        
        # Extract basic metadata
        metadata['title'] = os.path.basename(self.repo_path)
        metadata['description'] = self.get_repo_description()
        metadata['topics'] = self.get_repo_topics()
        
        # Extract README content
        readme_content = self.get_file_content('README.md')
        if readme_content:
            metadata['readme'] = readme_content
        
        # Extract commit messages
        metadata['commit_messages'] = self.get_commit_messages()
        
        # Extract issues
        metadata['issues'] = self.get_issues()
        
        # Extract package.json data
        package_json = self.get_file_content('package.json')
        if package_json:
            try:
                package_data = json.loads(package_json)
                metadata['dependencies'] = list(package_data.get('dependencies', {}).keys())
                metadata['dev_dependencies'] = list(package_data.get('devDependencies', {}).keys())
            except json.JSONDecodeError:
                pass
        
        # Extract pom.xml data
        pom_xml = self.get_file_content('pom.xml')
        if pom_xml:
            try:
                root = ET.fromstring(pom_xml)
                metadata['groupId'] = root.find('groupId').text
                metadata['artifactId'] = root.find('artifactId').text
                metadata['version'] = root.find('version').text
            except ET.ParseError:
                pass
        
        # Combine all metadata into a single string
        combined_metadata = ' '.join(str(value) for value in metadata.values() if value)
        return combined_metadata

    def get_file_content(self, filename):
        file_path = os.path.join(self.repo_path, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        return None

    def get_repo_description(self):
        if self.github_client:
            repo_name = os.path.basename(self.repo_path)
            repo = self.github_client.get_repo(repo_name)
            return repo.description
        return ""

    def get_repo_topics(self):
        if self.github_client:
            repo_name = os.path.basename(self.repo_path)
            repo = self.github_client.get_repo(repo_name)
            return repo.get_topics()
        return []

    def get_commit_messages(self):
        if self.github_client:
            repo_name = os.path.basename(self.repo_path)
            repo = self.github_client.get_repo(repo_name)
            commits = repo.get_commits()
            return [commit.commit.message for commit in commits]
        return []

    def get_issues(self):
        if self.github_client:
            repo_name = os.path.basename(self.repo_path)
            repo = self.github_client.get_repo(repo_name)
            issues = repo.get_issues(state='all')
            return [issue.title + " " + (issue.body or "") for issue in issues]
        return []

    def preprocess_text(self, text):
        tokens = simple_preprocess(text)
        lemmatized = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return lemmatized
