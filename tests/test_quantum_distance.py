import unittest 
import os 
import sys 
import argparse

import numpy as np 
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/beta/")
from quantum_distance import QuantumDistance as qd


class TestQuantumDistance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(args.seed)

    def test_proper_state_preparation(self):
        """
        Tests that the script doing the proper state 
        preparation
        """
        x1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        x2 = [1,1]
        circuit = qd(x1,x2)
        gate = circuit.build_gate_state_preparation(x1)
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.append(gate,[0,1])
        probabilities_dict = Statevector(qc).probabilities_dict()
        # We know the statevector of this circuit is 1/sqrt(2)|01> + 1/sqrt(2)|11>
        for k,v in probabilities_dict.items():
            self.assertAlmostEqual(v,0.5)
        self.assertIn('01', list(probabilities_dict.keys()))
        self.assertIn('11', list(probabilities_dict.keys()))

    def test_quantum_distance_equals_eucl_distance(self):
        """
        Tests that for normalised vectors that don't need padding, the 
        quantum distance and the euclidean distance output 
        the same value (up to a small error)
        """
        for n in args.sizes_without_padding:
            for _ in range(args.n_samples):
                x1 = np.random.uniform(-args.x_limit,args.x_limit, size=n)
                x2 = np.random.uniform(-args.x_limit,args.x_limit, size=n)
                x1 = x1/np.linalg.norm(x1)
                x2 = x2/np.linalg.norm(x2)
                quantum_distance = qd(x1,x2).compute_quantum_distance()
                euclidean_distance = np.linalg.norm(x2 - x1)
                self.assertAlmostEqual(quantum_distance, euclidean_distance)

    def test_quantum_distance_equals_eucl_distance_padding(self):
        """
        Tests that for normalised vectors when padding is needed
        (lengths of the vecotrs are not a power of 2), the quantum distance
        and euclidean distance output the same value (up to a small error)
        """
        for n in args.sizes_with_padding:
            for _ in range(args.n_samples):
                x1 = np.random.uniform(-args.x_limit,args.x_limit, size=n)
                x2 = np.random.uniform(-args.x_limit,args.x_limit, size=n)
                x1 = x1/np.linalg.norm(x1)
                x2 = x2/np.linalg.norm(x2)
                quantum_distance = qd(x1, x2).compute_quantum_distance()
                euclidean_distance = np.linalg.norm(x2 - x1)
                self.assertAlmostEqual(quantum_distance, euclidean_distance)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--n_samples", type = int,
        help = "Number of random samples to be generated for testing",
        default = 100
    )
    parser.add_argument(
        "-xl", "--x_limit", type = int, 
        help = "Limits of the generated random vectors to test",
        default = 1000
    )
    parser.add_argument(
        "-sp", "--sizes_without_padding", type = int, 
        help = ("Sizes of the vectors that don't need padding."
                "All values must be powers of 2" ),
        nargs = "+", default = [2,4,8]
    )
    parser.add_argument(
        "-swp", "--sizes_with_padding", type = int, 
        help = ("Sizes of the vectors that need padding."
                "None of the values can be a power of 2" ),
        nargs = "+", default = [3,5,7]
    )
    parser.add_argument(
        "-s", "--seed", type = int,
        help = "Random seed for generating the vectors",
        default = 180567
    )
    args, remaining = parser.parse_known_args()
    remaining.insert(0, sys.argv[0])
    unittest.main(argv=remaining)