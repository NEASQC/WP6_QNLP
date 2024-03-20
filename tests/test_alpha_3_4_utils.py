import os
import random
import sys
import unittest

import numpy as np
import torch

# The two lines below will be removed when converting the library to a package.
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/")
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/alpha_3_4/")

from utils import *


class TestAlpha3Utils(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """
        Sets up class for testing the utils module.
        """
        cls.num_vectors = 10
        cls.vector_size = np.random.randint(2, 101)
        cls.num_classes = np.random.randint(2, 11)
        vectors = []
        labels = []

        for _ in range(cls.num_vectors):
            vector = np.random.rand(cls.vector_size)
            vectors.append(vector)
            labels.append(np.random.randint(cls.num_classes))

        tensors = [torch.tensor(vector) for vector in vectors]
        cls.dataset = Dataset(tensors, labels)

    def test_one_hot_encoding_does_what_one_expects(self) -> None:
        """
        Tests that get_labels_one_hot_encoding behaves as one would
        expect.
        """
        labels_train = [0, 1, 2, 0]
        labels_val = [1, 2, 1]
        labels_test = [0, 2]

        one_hot_encoded_labels, num_classes = get_labels_one_hot_encoding(
            labels_train, labels_val, labels_test
        )

        expected_train_one_hot = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
        expected_val_one_hot = [[0, 1, 0], [0, 0, 1], [0, 1, 0]]
        expected_test_one_hot = [[1, 0, 0], [0, 0, 1]]
        expected_num_classes = 3

        self.assertEqual(one_hot_encoded_labels[0], expected_train_one_hot)
        self.assertEqual(one_hot_encoded_labels[1], expected_val_one_hot)
        self.assertEqual(one_hot_encoded_labels[2], expected_test_one_hot)
        self.assertEqual(num_classes, expected_num_classes)

    def test_utils_getitem_method_returns_tensor(self) -> None:
        """
        Tests that the __getitem__ method returns a torch.float32
        object.
        """
        n = np.random.randint(self.num_vectors)
        nth_item = self.dataset.__getitem__(n)
        self.assertEqual(nth_item[0].dtype, torch.float32)
        self.assertEqual(nth_item[1].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
