# src/repo2vec.py

import numpy as np
from .meta2vec import Meta2Vec
from .struct2vec import Struct2Vec
from .source2vec import Source2Vec

class Repo2Vec:
    def __init__(self, repo_path, vector_size=128, combination_method='weighted_sum', weights=(1, 1, 1), normalize=True):
        self.repo_path = repo_path
        self.vector_size = vector_size
        self.combination_method = combination_method
        self.weights = weights
        self.normalize = normalize
        self.meta2vec = Meta2Vec(repo_path, vector_size)
        self.struct2vec = Struct2Vec(repo_path, vector_size)
        self.source2vec = Source2Vec(repo_path, vector_size)

    def generate_embedding(self):
        meta_vector = self.meta2vec.generate()
        struct_vector = self.struct2vec.generate()
        source_vector = self.source2vec.generate()

        if self.normalize:
            meta_vector = self._normalize(meta_vector)
            struct_vector = self._normalize(struct_vector)
            source_vector = self._normalize(source_vector)

        if self.combination_method == 'concatenate':
            combined_vector = self._concatenate([meta_vector, struct_vector, source_vector])
        elif self.combination_method == 'sum':
            combined_vector = self._element_wise_sum([meta_vector, struct_vector, source_vector])
        elif self.combination_method == 'average':
            combined_vector = self._element_wise_average([meta_vector, struct_vector, source_vector])
        elif self.combination_method == 'median':
            combined_vector = self._element_wise_median([meta_vector, struct_vector, source_vector])
        elif self.combination_method == 'weighted_sum':
            combined_vector = self._weighted_sum([meta_vector, struct_vector, source_vector], self.weights)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")

        if self.normalize:
            combined_vector = self._normalize(combined_vector)

        return combined_vector

    def _normalize(self, vector):
        return vector / np.linalg.norm(vector)

    def _concatenate(self, vectors):
        return np.concatenate(vectors)

    def _element_wise_sum(self, vectors):
        return np.sum(vectors, axis=0)

    def _element_wise_average(self, vectors):
        return np.mean(vectors, axis=0)

    def _element_wise_median(self, vectors):
        return np.median(vectors, axis=0)

    def _weighted_sum(self, vectors, weights):
        return np.sum([w * v for w, v in zip(weights, vectors)], axis=0)
