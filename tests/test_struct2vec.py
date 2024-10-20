# tests/test_struct2vec.py

import unittest
import numpy as np
from src.struct2vec import Struct2Vec

class TestStruct2Vec(unittest.TestCase):
    def setUp(self):
        self.repo_path = "path/to/test/repo"
        self.vector_size = 128
        self.struct2vec = Struct2Vec(self.repo_path, self.vector_size)

    def test_generate(self):
        vector = self.struct2vec.generate()
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.shape, (self.vector_size,))

if __name__ == '__main__':
    unittest.main()
