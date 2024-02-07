import argparse
import math as mt 
import os 
import sys 
import unittest

import pennylane as qml
import torch

# The two lines below will be removed when converting the library to a package.
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/")
import circuit as circ

class TestCircuit(unittest.TestCase):
    """
    Class for testing Circuit class.
    """
    @classmethod
    def setUpClass(cls)-> None:
        """
        Set up the class for testing.
        """
        torch.manual_seed(args.seed)
        cls.n_qubits = args.n_qubits
        cls.n_layers = args.n_layers
        cls.axis_encoding = args.axis_encoding
        observables_funcs_dict = {
            'X' : qml.PauliX, 
            'Y' : qml.PauliY,
            'Z' : qml.PauliZ
        }
        cls.observables = [
            observables_funcs_dict[
                args.observable] for _ in range(args.n_qubits)
        ]
        cls.ansatze = [circ.Sim14, circ.Sim15, circ.StronglyEntangling]
        cls.circuits = [
            ansatz(
                cls.n_qubits, cls.n_layers,
                cls.axis_encoding, cls.observables) for ansatz in cls.ansatze
        ]
        cls.params = [
            torch.nn.Parameter(
                torch.randn(
                    circuit.parameters_shape)) for circuit in cls.circuits
        ]
        cls.input = torch.nn.Parameter(torch.randn(cls.n_qubits))

    def test_circuits_output_correct_type(self)-> None:
        """
        Test the that the the three different ansatze implemented 
        work, asserting that the expected value obtained
        after running the circuit is a tensor.
        """
        for i,circuit in enumerate(self.circuits):
            circuit_function = circuit.build_circuit_function
            results = circuit.run_and_measure_circuit(
                circuit_function)(self.input, self.params[i])
            ### We make sure the output has the proper format
            for r in results:
                self.assertIsInstance(r, torch.Tensor)

    def test_circuit_outputs_of_different_rescaling_functions_are_different(
        self)-> None:
        """
        Test the circuit output after applying rescaling functions on inputs,
        ensuring that it varies when different 
        rescaling functions are applied.
        """
        data_rescaling_functions = [
            lambda x : 0.5 * mt.pi * x,
            lambda x : mt.pi * x,
            lambda x : 2.0 * mt.pi * x
        ]
        for i,ansatz in enumerate(self.ansatze):
            results_list = []
            for f in data_rescaling_functions:
                circuit = ansatz(
                    self.n_qubits, self.n_layers,
                    self.axis_encoding, self.observables,
                    data_rescaling = f
                )
                circuit_function = circuit.build_circuit_function
                results = circuit.run_and_measure_circuit(
                circuit_function)(self.input, self.params[i])
                results_list.append(results)
            self.assertNotEqual(results_list[0], results_list[1])
            self.assertNotEqual(results_list[1], results_list[2])
            self.assertNotEqual(results_list[2], results_list[0])
                          
    def test_circuit_outputs_of_different_observables_are_different(
        self)-> None:
        """
        Test the circuit output of different observables,
        ensuring that it varies when different 
        observables are measured 
        """
        operators = [
            qml.Identity, qml.Hadamard, qml.PauliX,
            qml.PauliY, qml.PauliZ
        ]
        observables = []
        for op in operators:
            observables.append([op for _ in range(self.n_qubits)])
        for i, ansatz in enumerate(self.ansatze):
            results_list =[]
            for ob in observables:
                circuit = ansatz(
                    self.n_qubits, self.n_layers,
                    self.axis_encoding, ob
                )
                circuit_function = circuit.build_circuit_function
                results = circuit.run_and_measure_circuit(
                circuit_function)(self.input, self.params[i])
                self.assertNotIn(results, results_list)
                results_list.append(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nq", "--n_qubits", type = int,
        help = "Number of qubits in the circuits.",
        default = 3
    )
    parser.add_argument(
        "-nl", "--n_layers", type = int,
        help = "Number of layers in the circuit.",
        default = 1
    )
    parser.add_argument(
        "-ax", "--axis_encoding", type = str,
        help = "Axis for rotation encoding. Must be in (X,Y,Z).",
        default = 'X'
    )
    parser.add_argument(
        "-ob", "--observable", type = str,
        help = ("Pauli operator for which output the expected value."
        "Must be in (X,Y,Z)."),
        default = 'Z'
    )
    parser.add_argument(
        "-s", "--seed", type = str,
        help = "Seed for random params and inputs.",
        default = 1997
    )
    args, remaining = parser.parse_known_args()
    remaining.insert(0, sys.argv[0])
    unittest.main(argv=remaining)