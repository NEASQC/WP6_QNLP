import unittest 
import os 
import sys 
import argparse

import numpy as np 

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/beta/")
from beta_2 import QuantumKMeans as qkm

class TestBeta2(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(args.seed)
        cls.x_train = [np.random.uniform(
            -args.x_limit, args.x_limit, size=args.x_size) for _ in range(
                args.n_train)]
        cls.beta_2_model = qkm(
            cls.x_train, args.k
        )
        if args.random_cluster_initialiser == 0:
            cls.beta_2_model.initialise_cluster_centers(
                random_initialisation = True,
                seed = args.seed
            )
        else : 
            cls.beta_2_model.initialise_cluster_centers(
                random_initialisation=False)
        cls.beta_2_model.run_k_means_algorithm()

    def test_number_of_train_predictions_is_correct(self):
        """
        Checks that the number of predictions output by the model is correct
        """
        self.assertEqual(
            len(self.beta_2_model.get_train_predictions()),
            len(self.x_train)
        )

    def test_number_of_clusters_is_correct(self):
        """
        Checks that the number of predictions output by the model is correct
        """
        self.assertEqual(
            len(self.beta_2_model.get_clusters_centers()),
            args.k
        )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ntr", "--n_train", type = int,
        help = "Number of random samples in train dataset",
        default = 25
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
        "-k", "--k", type = int, 
        help = "Number of clusters",
        default = 4
    )
    parser.add_argument(
        "-ci", "--random_cluster_initialiser", type = int, 
        help = "Wether to use random cluster (0) or not (1)",
        default = 0
    )
    parser.add_argument(
        "-s", "--seed", type = int,
        help = "Random seed for generating the vectors",
        default = 180567
    )
    args, remaining = parser.parse_known_args()
    remaining.insert(0, sys.argv[0])
    unittest.main(argv=remaining)