# tests/test_source2vec.py

import unittest
import numpy as np
from src.source2vec import Source2Vec

class TestSource2Vec(unittest.TestCase):
    def setUp(self):
        self.repo_path = "path/to/test/repo"
        self.vector_size = 128
        self.source2vec = Source2Vec(self.repo_path, self.vector_size)

    def test_generate(self):
        vector = self.source2vec.generate()
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.shape, (self.vector_size,))

if __name__ == '__main__':
    unittest.main()
