# src/source2vec.py

import os
import numpy as np
import javalang
from javalang import parse
from javalang.ast import Node
from code2vec.model import Code2VecModel
from code2vec.config import Config

class Source2Vec:
    def __init__(self, repo_path, vector_size):
        self.repo_path = repo_path
        self.vector_size = vector_size
        self.code2vec_model = self._load_code2vec_model()

    def _load_code2vec_model(self):
        config = Config(set_defaults=True, load_from_args=False, config_dict={
            'SAVED_MODEL_PATH': 'path/to/pretrained/code2vec_model'
        })
        model = Code2VecModel(config)
        model.load()
        return model

    def generate(self):
        java_files = self.get_java_files()
        method_vectors = []

        for file in java_files:
            with open(file, 'r') as f:
                content = f.read()

            try:
                tree = parse.parse(content)
                for _, node in tree.filter(Node):
                    if isinstance(node, javalang.tree.MethodDeclaration):
                        method_vector = self.get_method_vector(node)
                        method_vectors.append(method_vector)
            except Exception:
                pass

        if not method_vectors:
            return np.zeros(self.vector_size)

        return np.mean(method_vectors, axis=0)

    def get_java_files(self):
        java_files = []
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith('.java'):
                    java_files.append(os.path.join(root, file))
        return java_files

    def get_method_vector(self, method_node):
        method_text = self.get_method_text(method_node)
        try:
            prediction = self.code2vec_model.predict(method_text)
            return prediction.code_vector
        except Exception:
            return np.zeros(self.vector_size)

    def get_method_text(self, method_node):
        return method_node.name
