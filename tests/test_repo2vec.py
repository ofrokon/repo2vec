# tests/test_repo2vec.py

import unittest
import numpy as np
from src.repo2vec import Repo2Vec

class TestRepo2Vec(unittest.TestCase):
    def setUp(self):
        self.repo_path = "path/to/test/repo"
        self.repo2vec = Repo2Vec(self.repo_path)

    def test_generate_embedding(self):
        embedding = self.repo2vec.generate_embedding()
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (self.repo2vec.vector_size * 3,))

if __name__ == '__main__':
    unittest.main()
