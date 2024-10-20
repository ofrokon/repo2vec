# src/meta2vec.py

import os
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class Meta2Vec:
    def __init__(self, repo_path, vector_size):
        self.repo_path = repo_path
        self.vector_size = vector_size
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def generate(self):
        metadata = self.extract_metadata()
        processed_metadata = self.preprocess_text(metadata)
        
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(processed_metadata)]
        model = Doc2Vec(documents, vector_size=self.vector_size, window=5, min_count=1, workers=4, epochs=100)
        
        return model.infer_vector(processed_metadata)

    def extract_metadata(self):
        metadata = ""
        metadata_files = ['README.md', 'DESCRIPTION', 'package.json', 'setup.py', 'pom.xml']
        
        for file in metadata_files:
            file_path = os.path.join(self.repo_path, file)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    metadata += f.read() + " "
        
        return metadata

    def preprocess_text(self, text):
        tokens = simple_preprocess(text)
        lemmatized = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return lemmatized
