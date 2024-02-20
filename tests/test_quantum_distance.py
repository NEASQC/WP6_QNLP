import os 
import sys 
import unittest 

import numpy as np 
from parameterized import parameterized_class
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# The two lines below will be removed when converting the library to a package.
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/beta/")
from quantum_distance import QuantumDistance as qd
from utils import normalise_vector, pad_vector_with_zeros

test_args = {
    'n_runs' : 5,
    'seed' : 30031935,
    'n_samples' : 100,
    'vectors_limit_value' : 1000
}

def set_up_test_parameters(test_args : dict)-> list:
    """
    Generate parameters for different test runs. 

    Parameters
    ----------
    test_args : dict
        Dictionary with test arguments to generate the test parameters of the 
        different runs. Its keys are the following: 
            n_runs : Number of test runs.
            seed : Random seed for generating the different parameters.
            n_samples : Number of vectors to generate in each test run.
            vectors_limit_value : Limit value that the components of the vectors
        can have.
    
    Return
    ------
    params_list : list 
        List with the parameters to be used 
        on each run.
    """
    params_list = []
    np.random.seed(test_args['seed'])
    powers_of_two = [2,4,8]
    not_powers_of_two = [3,5,6,7]
    for _ in range(test_args['n_runs']):
        params_run = []
        size_that_doesnt_need_padding = np.random.choice(
            powers_of_two
        )
        size_that_needs_padding = np.random.choice(
            not_powers_of_two
        )
        vectors_that_dont_need_padding = [np.random.uniform(
            -test_args['vectors_limit_value'],
            test_args['vectors_limit_value'],
            size = size_that_doesnt_need_padding
        ) for _ in range(test_args['n_samples'])]
        vectors_that_need_padding = [np.random.uniform(
            -test_args['vectors_limit_value'],
            test_args['vectors_limit_value'],
            size = size_that_needs_padding
        ) for _ in range(test_args['n_samples'])]
        params_run.append(vectors_that_need_padding)
        params_run.append(vectors_that_dont_need_padding)
        params_run.append(test_args['n_runs'])
        params_run.append(test_args['n_samples'])
        params_list.append(params_run)
    return params_list

names_parameters = (
'vectors_that_need_padding',
'vectors_that_dont_need_padding',
'n_runs', 'n_samples'
)
@parameterized_class(names_parameters, set_up_test_parameters(test_args))
class TestQuantumDistance(unittest.TestCase):
    @classmethod
    def setUpClass(cls)-> None:
        """
        Set up the class for testing. The attributes are defined 
        with the decorator on line 77. 
        """
        pass

    def test_value_error_is_raised_if_vectors_dont_have_same_length(
        self
    )-> None:
        """
        Test that a ValueError is raised when computing the quantum distance
        for two vectors with different length.
        """
        for vector_1,vector_2 in zip(
            self.vectors_that_dont_need_padding,
            self.vectors_that_need_padding
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
        for j in range(self.n_samples -1):
            vector_1 = self.vectors_that_dont_need_padding[j]
            vector_2 = self.vectors_that_dont_need_padding[j + 1]
            with self.assertRaises(ValueError):
                quantum_distance = qd(
                    vector_1,vector_2
                ).compute_quantum_distance()

    def test_value_error_is_raised_if_vectors_are_not_powers_of_two(
        self
    )-> None:
        """
        For two vectors with same length, test that a ValueError is
        raised when computing the quantum distance if one or both of
        them have norm different from 1.
        """
        for j in range(self.n_samples -1):
            vector_1 = self.vectors_that_need_padding[j]
            vector_1 = normalise_vector(vector_1)
            vector_2 = self.vectors_that_need_padding[j + 1]
            vector_2 = normalise_vector(vector_2)
            with self.assertRaises(ValueError):
                quantum_distance = qd(
                    vector_1,vector_2
                ).compute_quantum_distance()
             
    def test_input_vectors_are_encoded_correctly_in_quantum_circuit(
        self
    )-> None:
        """
        Test that input vectors are encoded correctly in the quantum circuit
        to compute the quantum distance, i.e., the amplitudes of the
        quantum states of the circuit represent the input vector.
        """
        vector_1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        vector_2 = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        circuit = qd(vector_1,vector_2)
        gate = circuit.build_gate_state_preparation(vector_1)
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.append(gate,[0,1])
        probabilities_dict = Statevector(qc).probabilities_dict()
        # We know the statevector of this circuit
        # is 1/sqrt(2)|01> + 1/sqrt(2)|11>
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
            for k in range(self.n_samples -1):
                vector_1 = vectors_list[k]
                vector_2 = vectors_list[k + 1]
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
    unittest.main()