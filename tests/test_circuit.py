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
        cls.axis_embedding = args.axis_embedding
        observables_funcs_dict = {
            'X' : qml.PauliX, 
            'Y' : qml.PauliY,
            'Z' : qml.PauliZ,
            'H' : qml.Hadamard
        }
        cls.observables = {
            args.qubit_index[i] : observables_funcs_dict[args.observables[i]]
            for i in range(len(args.qubit_index))
        }
        cls.ansatze = [circ.Sim14, circ.Sim15, circ.StronglyEntangling]
        circuits_exp_value = [
            ansatz(
                cls.n_qubits, cls.n_layers,
                cls.axis_embedding, cls.observables, 
                output_probabilities = False
            ) for ansatz in cls.ansatze
        ]
        circuits_probs = [
            ansatz(
                cls.n_qubits, cls.n_layers,
                cls.axis_embedding, cls.observables,
                output_probabilities = True
            ) for ansatz in cls.ansatze
        ]
        cls.circuits = [circuits_exp_value, circuits_probs]
        cls.params = [
            torch.nn.Parameter(
                torch.randn(
                    circuit.parameters_shape)) for circuit in cls.circuits[0]
        ]
        cls.input = torch.nn.Parameter(torch.randn(cls.n_qubits))
        cls.results_circuits = [[], []]
        for i,c in enumerate(cls.circuits):
            for j,circuit in enumerate(c):
                    circuit_function = circuit.build_circuit_function
                    cls.results_circuits[i].append(
                        circuit.run_and_measure_circuit(
                        circuit_function)(cls.input, cls.params[j])
                    )

    def test_circuits_return_a_tensor(self)-> None:
        """
        Test the that the three different ansatze implemented 
        work (for expectation value and prob outputs),
        asserting that a tensor is obtained as output.
        """
        for i,r in enumerate(self.results_circuits):
            for result in r:
                with self.subTest(result=result):
                    for tensor in result:
                        self.assertIsInstance(tensor, torch.Tensor)

    def test_circuit_outputs_probabilities_between_0_and_1(self)-> None:
        """
        For all the ansatze, test that when probabilities are output,
        they lay in the range (0,1).
        """
        for result in self.results_circuits[1]:
            for tensor in result:
                with self.subTest(tensor=tensor):
                    self.assertIs(
                        bool((tensor >= 0).all() and (tensor <=1).all()), True
                    )
                
    def test_circuit_outputs_of_different_rescaling_functions_are_different(
        self)-> None:
        """
        Test the circuit output after applying rescaling functions on inputs,
        ensuring that it varies when different functions are applied.
        A value for the input embedding and observables will be set for which
        we know that the outputs will vary.
        """
        r1 = 2 * torch.rand(1).item()
        r2 = 2 * torch.rand(1).item()
        print(r1, r2)
        data_rescaling_functions = [
            lambda x : r1 * mt.pi * x,
            lambda x : r2 * mt.pi * x,
        ]
        embedding = 'X'
        observables = {k : qml.PauliZ for k in self.observables}
        for i, ansatz in enumerate(self.ansatze):
            for output_probabilities in (True, False):
                with self.subTest(
                    ansatz = ansatz,
                    output_probabilities = output_probabilities
                ):
                    results_list = []
                    for f in data_rescaling_functions:
                        circuit = ansatz(
                        self.n_qubits, self.n_layers,
                        embedding, observables,
                        output_probabilities = output_probabilities,
                        data_rescaling = f
                        )
                        circuit_function = circuit.build_circuit_function
                        results = circuit.run_and_measure_circuit(
                        circuit_function
                        )(self.input, self.params[i])
                        results_list.append(results)
                    for t1, t2 in zip(results_list[0], results_list[1]):
                        self.assertFalse(
                            torch.allclose(t1, t2)
                        )

                          
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
        "-ax", "--axis_embedding", type = str,
        help = "Axis for rotation encoding. Must be in (X,Y,Z).",
        default = 'X'
    )
    parser.add_argument(
        "-ob", "--observables", type = str, nargs = '+',
        help = (" List of Pauli operators for which output"
        "the expected value or probability."
        "Must be in (X,Y,Z) and length equal to qubit index length."),
        default = ['Z', 'Z', 'Z']
    )
    parser.add_argument(
        "-qi", "--qubit_index", type = int, nargs = '+',
        help = (" List of qubits indexes where the operators act."
        "Its length must be equal to observables length."),
        default = [0,1,2]
    )
    parser.add_argument(
        "-s", "--seed", type = str,
        help = "Seed for random params and inputs.",
        default = 1997
    )
    args, remaining = parser.parse_known_args()
    remaining.insert(0, sys.argv[0])
    unittest.main(argv=remaining)