import unittest 
import os 
import sys 
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/beta/")
from quantum_distance import QuantumDistance as qd
from quantum_k_nearest_neighbours_bert import QuantumKNearestNeighbours_bert as qkn
import numpy as np 
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

class TestQuantumDistance(unittest.TestCase):

    def test_state_preparation(self):
        x1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        x2 = [1,1]
        model = qd(x1,x2)
        gate = model.build_gate_state_preparation(x1)
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.append(gate,[0,1])
        probabilities_dict = Statevector(qc).probabilities_dict()
        # We know the statevector of this circuit is 1/sqrt(2)|01> + 1/sqrt(2)|11>
        for k,v in probabilities_dict.items():
            self.assertAlmostEqual(v,0.5)
        self.assertIn('01', list(probabilities_dict.keys()))
        self.assertIn('11', list(probabilities_dict.keys()))

    def test_real_quantum_distance(self):
        sizes = [2, 4, 8, 16]
        np.random.seed(18051967)
        for i,n in enumerate(sizes):
            for j in range(100):
                x1 = np.random.uniform(-1000,1000, size=n)
                x2 = np.random.uniform(-1000,1000, size=n)
                x1 = (x1/np.linalg.norm(x1))
                x2 = (x2/np.linalg.norm(x2))
                model = qd(x1,x2)
                quantum_distance = model.compute_quantum_distance()
                euclidean_distance = np.linalg.norm(x2 - x1)
                self.assertAlmostEqual(quantum_distance, euclidean_distance)

    def test_real_quantum_distance_with_padding(self):
        sizes = [5,7]
        for i,n in enumerate(sizes):
            X1 = [np.random.uniform(-1000,1000,size =n) for i in range(100)]
            X2 = [np.random.uniform(-1000,1000,size =n) for i in range(100)]
            X1 = qkn.normalise_vector(X1)
            X2 = qkn.normalise_vector(X2)
            X1 = qkn.pad_zeros_vector(X1)
            X2 = qkn.pad_zeros_vector(X2)
            for i in range(100):
                x1 = X1[i]
                x2 = X2[i]
                model = qd(x1, x2)
                quantum_distance = model.compute_quantum_distance()
                euclidean_distance = np.linalg.norm(x2 - x1)
                self.assertAlmostEqual(quantum_distance, euclidean_distance)




if __name__ == '__main__':

    unittest.main()