# src/repo2vec.py

import numpy as np
from .meta2vec import Meta2Vec
from .struct2vec import Struct2Vec
from .source2vec import Source2Vec

class Repo2Vec:
    def __init__(self, repo_path, vector_size=128, weights=(1, 1, 1)):
        self.repo_path = repo_path
        self.vector_size = vector_size
        self.weights = weights
        self.meta2vec = Meta2Vec(repo_path, vector_size)
        self.struct2vec = Struct2Vec(repo_path, vector_size)
        self.source2vec = Source2Vec(repo_path, vector_size)

    def generate_embedding(self):
        meta_vector = self.meta2vec.generate()
        struct_vector = self.struct2vec.generate()
        source_vector = self.source2vec.generate()

        w_m, w_s, w_c = self.weights
        combined_vector = np.concatenate([
            w_m * meta_vector, 
            w_s * struct_vector, 
            w_c * source_vector
        ])

        # Normalize the combined vector
        combined_vector = combined_vector / np.linalg.norm(combined_vector)

        return combined_vector
