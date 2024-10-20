# tests/test_meta2vec.py

import unittest
import numpy as np
from src.meta2vec import Meta2Vec

class TestMeta2Vec(unittest.TestCase):
    def setUp(self):
        self.repo_path = "path/to/test/repo"
        self.vector_size = 128
        self.meta2vec = Meta2Vec(self.repo_path, self.vector_size)

    def test_generate(self):
        vector = self.meta2vec.generate()
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.shape, (self.vector_size,))

if __name__ == '__main__':
    unittest.main()
