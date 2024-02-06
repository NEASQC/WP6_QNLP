import unittest 
import os 
import sys 
import argparse

import numpy as np 

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/beta/")
from beta_1 import QuantumKNearestNeighbours as qkn
from utils import normalise_vector, pad_vector_with_zeros

class TestBeta1(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(args.seed)
        cls.x_train = [np.random.uniform(
            -args.x_limit, args.x_limit, size=args.x_size) for _ in range(
                args.n_train)]
        cls.x_test = [np.random.uniform(
            -args.x_limit, args.x_limit, size=args.x_size) for _ in range(
                args.n_test)]
        for i,xtr in enumerate(cls.x_train):
            xtr = normalise_vector(xtr)
            cls.x_train[i] = pad_vector_with_zeros(xtr)
        for j,xte in enumerate(cls.x_test):
            xte = normalise_vector(xte)
            cls.x_test[j] = pad_vector_with_zeros(xte)
        cls.y_train = np.random.randint(0, args.n_classes, size = args.n_train)
        cls.y_test = np.random.randint(0, args.n_classes, size = args.n_test)
        cls.beta_1_model = qkn(
            cls.x_train, cls.x_test, cls.y_train, cls.y_test, args.k
        )
        cls.beta_1_model.compute_test_train_distances()
        cls.beta_1_model.compute_closest_vectors_indexes()
        cls.beta_1_model.compute_test_predictions()
    
    def test_number_of_train_test_distances_is_correct(self) -> None:
        """
        Checks that for each train instance, its distance with all the 
        test vectors is computed 
        """
        for dist_train in self.beta_1_model.test_train_distances:
            self.assertEqual(
                len(dist_train), args.n_train)

    def test_number_of_closest_vectors_indexes_is_correct(self) -> None:
        """
        Checks that for each train instance, the length of the list 
        with the closest vectors indexes has length equal to k 
        """
        for indexes in self.beta_1_model.closest_vectors_indexes:
            self.assertEqual(
                len(indexes), args.k
            )

    def test_save_and_load_train_test_distances(self) -> None:
        """
        Checks usage of save and load train_test_distances, by ensuring
        that it the distances have the same value before and after saving
        """
        distances_before_saving = self.beta_1_model.test_train_distances
        self.beta_1_model.save_test_train_distances(
            'test_train_distances', './'
        )
        self.beta_1_model.load_test_train_distances(
            'test_train_distances.pickle', './'
        )
        self.assertEqual(
            distances_before_saving, self.beta_1_model.test_train_distances
        )

    def test_predictions_are_integers(self) -> None:
        """
        Checks that the predictions output by the model are integers
        """
        for pred in self.beta_1_model.test_predictions:
            self.assertIs(type(pred), np.int64)
            



        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ntr", "--n_train", type = int,
        help = "Number of random samples in train dataset",
        default = 25
    )
    parser.add_argument(
        "-nte", "--n_test", type = int,
        help = "Number of random samples in test dataset",
        default = 5
    )
    parser.add_argument(
        "-xs", "--x_size", type = int, 
        help = "Length of generated random samples",
        default = 4
    )
    parser.add_argument(
        "-xl", "--x_limit", type = int, 
        help = "Limits of the generated random samples",
        default = 1000
    )
    parser.add_argument(
        "-nc", "--n_classes", type = int, 
        help = "Number of generated random classes",
        default = 4
    )
    parser.add_argument(
        "-k", "--k", type = int, 
        help = "Number of k neighbours",
        default = 4
    )
    parser.add_argument(
        "-s", "--seed", type = int,
        help = "Random seed for generating the vectors",
        default = 180567
    )
    args, remaining = parser.parse_known_args()
    remaining.insert(0, sys.argv[0])
    unittest.main(argv=remaining)