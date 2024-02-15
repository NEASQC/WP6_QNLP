import argparse
import os 
import sys 
import unittest 

import numpy as np 

# The two lines below will be removed when converting the library to a package.
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/beta/")
from beta_2 import QuantumKMeans as qkm


class TestBeta2(unittest.TestCase):
    @classmethod
    def setUpClass(cls)-> None:
        """
        Set up the class for testing.
        """
        np.random.seed(args.seed)
        cls.x_train = [np.random.uniform(
            -args.x_limit, args.x_limit, size=args.x_size) for _ in range(
                args.n_train)]
        cls.beta_2 = qkm(
            cls.x_train, args.k, args.tolerance, args.itermax
        )
        if args.random_cluster_initialiser == 0:
            cls.beta_2.initialise_cluster_centers(
                random_initialisation = True,
                seed = args.seed
            )
        else : 
            cls.beta_2.initialise_cluster_centers(
                random_initialisation=False)
        cls.beta_2.train_k_means_algorithm()
        cls.beta_2.compute_wce()
        cls.beta_2.compute_predictions()

    def test_number_of_train_predictions_is_correct(self)-> None:
        """
        Check that the number of predictions output by the model is correct.
        """
        for i in range(self.beta_2.n_its):
            self.assertEqual(
                len(self.beta_2.predictions[i]),
                len(self.x_train)
            )

    def test_number_of_clusters_is_correct(self)-> None:
        """
        Check that the number of clusters output by the model is correct.
        """
        for i in range(self.beta_2.n_its):
            self.assertEqual(
                len(self.beta_2.clusters[i]),
                (args.k)
            )
    
    def test_number_of_centeres_is_correct(self)-> None:
        """
        Check that the number of centers output by the model is correct.
        """
        for i in range(self.beta_2.n_its):
            self.assertEqual(
                len(self.beta_2.centers[i]),
                args.k
            )

    def test_wce_is_float(self)-> None:
        """
        Check that the within clusters sum of errors values (WCE):
        $[error=\sum_{i=0}^{N}quantum_distance(x_{i}-center(x_{i}))$]
        are floats.
        """
        for i in range(self.beta_2.n_its):
            self.assertIs(
                type(self.beta_2.total_wce[i]),
                float
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ntr", "--n_train", type = int,
        help = "Number of random samples in train dataset.",
        default = 25
    )
    parser.add_argument(
        "-xs", "--x_size", type = int, 
        help = "Length of generated random samples.",
        default = 4
    )
    parser.add_argument(
        "-xl", "--x_limit", type = int, 
        help = "Limits of the generated random samples.",
        default = 1000
    )
    parser.add_argument(
        "-k", "--k", type = int, 
        help = "Number of clusters.",
        default = 4
    )
    parser.add_argument(
        "-tol", "--tolerance", type = float, 
        help = "Tolerance for the training phase of the algorithm.",
        default = 0.001
    )
    parser.add_argument(
        "-it", "--itermax", type = int, 
        help = "Number of maximum iterations of the training phase.",
        default = 200
    )
    parser.add_argument(
        "-ci", "--random_cluster_initialiser", type = int, 
        help = "Wether to use random cluster (0) or not (1).",
        default = 0
    )
    parser.add_argument(
        "-s", "--seed", type = int,
        help = "Random seed for generating the vectors.",
        default = 180567
    )
    args, remaining = parser.parse_known_args()
    remaining.insert(0, sys.argv[0])
    unittest.main(argv=remaining)