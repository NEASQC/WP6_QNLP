import unittest 
import os 
import sys 
import argparse

import numpy as np 
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# The two lines below will be removed when converting the library to a package.
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/beta/")
from quantum_distance import QuantumDistance as qd
from utils import normalise_vector, pad_vector_with_zeros


class TestQuantumDistance(unittest.TestCase):
    @classmethod
    def setUpClass(cls)-> None:
        """
        Set up the class for testing.
        """
        cls.n_sizes = len(args.sizes_that_dont_need_padding)
        cls.vectors_that_dont_need_padding = [
            [] for _ in range(cls.n_sizes)
        ]
        cls.vectors_that_need_padding = [
            [] for _ in range(cls.n_sizes)
        ]
        for i,(size_vector_1,size_vector_2) in enumerate(zip(
            args.sizes_that_dont_need_padding,
            args.sizes_that_need_padding
        )):
            for _ in range(args.n_samples):
                cls.vectors_that_dont_need_padding[i].append(
                    np.random.uniform(
                        -args.x_limit, args.x_limit, size = size_vector_1
                    )
                )
                cls.vectors_that_need_padding[i].append(
                    np.random.uniform(
                        -args.x_limit, args.x_limit, size = size_vector_2
                    )
                )
        np.random.seed(args.seed)

    def test_value_error_is_raised_if_vectors_dont_have_same_length(
        self
    )-> None:
        """
        Test that a ValueError is raised when computing the quantum distance
        for two vectors with different length.
        """
        for i in range (self.n_sizes):
            for vector_1,vector_2 in zip(
                self.vectors_that_dont_need_padding[i],
                self.vectors_that_need_padding[i]
            ):
                with self.assertRaises(ValueError):
                    quantum_distance = qd(
                        vector_1,vector_2
                    ).compute_quantum_distance()

    def test_value_error_is_raised_if_vectors_dont_have_norm_1(
        self
    )-> None:
        """
        For two vectors with same length, test that a ValueError is
        raised when computing the quantum distance if one or both of
        them have norm different from 1.
        """
        for i in range (self.n_sizes):
            for j in range(args.n_samples -1):
                vector_1 = self.vectors_that_dont_need_padding[i][j]
                vector_2 = self.vectors_that_dont_need_padding[i][j + 1]
                with self.assertRaises(ValueError):
                    quantum_distance = qd(
                        vector_1,vector_2
                    ).compute_quantum_distance()
             
    def test_model_encode_vectors_correctly(self)-> None:
        """
        Test that for a given vector the model encodes it correctly
        in a quantum circuit, i.e., the amplitudes of the quantum states
        of the circuit represent the input vector.
        """
        vector_1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        vector_2 = [1,1]
        circuit = qd(vector_1,vector_2)
        gate = circuit.build_gate_state_preparation(vector_1)
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.append(gate,[0,1])
        probabilities_dict = Statevector(qc).probabilities_dict()
        # We know the statevector of this circuit is 1/sqrt(2)|01> + 1/sqrt(2)|11>
        for k,v in probabilities_dict.items():
            self.assertAlmostEqual(v,0.5)
        self.assertIn('01', list(probabilities_dict.keys()))
        self.assertIn('11', list(probabilities_dict.keys()))

    def test_quantum_distance_equals_eucl_distance(self)-> None:
        """
        Test that the quantum distance and the euclidean distance output 
        the same value (up to a small error).
        """
        for i,vectors_list in enumerate(
            (
                self.vectors_that_dont_need_padding,
                self.vectors_that_need_padding
            )
        ):
            for j in range(self.n_sizes):
                for k in range(args.n_samples -1):
                    vector_1 = vectors_list[j][k]
                    vector_2 = vectors_list[j][k + 1]
                    vector_1 = normalise_vector(vector_1)
                    vector_2 = normalise_vector(vector_2)
                    if i == 1:
                        vector_1 = pad_vector_with_zeros(vector_1)
                        vector_2 = pad_vector_with_zeros(vector_2)
                    quantum_distance = qd(
                        vector_1,vector_2
                    ).compute_quantum_distance()
                    euclidean_distance = np.linalg.norm(vector_2 - vector_1)
                    self.assertAlmostEqual(
                        quantum_distance,
                        euclidean_distance
                    )
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--n_samples", type = int,
        help = "Number of random samples to be generated for testing.",
        default = 100
    )
    parser.add_argument(
        "-xl", "--x_limit", type = int, 
        help = "Limits of the generated random vectors to test.",
        default = 1000
    )
    parser.add_argument(
        "-spf", "--sizes_that_dont_need_padding", type = int, 
        help = ("List with sizes of the vectors that don't need padding."
                "All values must be powers of 2." 
                "It must have the same length as sizes_with_padding."),
        nargs = "+", default = [2,4,8]
    )
    parser.add_argument(
        "-spt", "--sizes_that_need_padding", type = int, 
        help = ("List with sizes of the vectors that need padding."
                "None of the values can be a power of 2." 
                "It must have the same length as sizes_without_padding."),
        nargs = "+", default = [3,5,7]
    )
    parser.add_argument(
        "-s", "--seed", type = int,
        help = "Random seed for generating the vectors.",
        default = 180567
    )
    args, remaining = parser.parse_known_args()
    remaining.insert(0, sys.argv[0])
    unittest.main(argv=remaining)