import unittest
import os 
import sys 
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/")
from circuit import *
import pennylane as qml
import math as mt 
import torch
import numpy as np 
import lambeq


class TestCircuit(unittest.TestCase):

    def setUp(self):
        """
        Set Up parameters to compare against lambeq circuits
        """
        self.sentences = ['John likes Mary']
        self.parser = lambeq.BobcatParser()
        self.diagrams = [self.parser.sentence2diagram(
            s) for s in self.sentences]
        self.qn = 1
        self.qs = 1
        self.n_layers = 1
        self.n_single_qubits_params = 1

    def test_lambeq_circuit(self):
        lambeq_ansatze_list = [
            lambeq.Sim14Ansatz,
            lambeq.Sim15Ansatz,
            lambeq.StronglyEntanglingAnsatz
        ]
        alpha_3_ansatze_list = [
            Sim14, Sim15, StronglyEntangling
        ]
        ansatze_names = ['Sim14', 'Sim15', 'StronlgyEntangling']
        n_qubits = 3 ; n_layers = 1 ; axis_embedding = 'Z'
        observables = [qml.PauliZ, qml.PauliZ, qml.PauliZ]
        input = torch.nn.Parameter(torch.randn(n_qubits))
        for i in range(len(lambeq_ansatze_list)):
            ansatz_lambeq = lambeq_ansatze_list[i](
            {lambeq.AtomicType.NOUN: self.qn,
            lambeq.AtomicType.SENTENCE: self.qs},
            n_layers=self.n_layers,
            n_single_qubit_params=self.n_single_qubits_params)
            circuits = [ansatz_lambeq(d) for d in self.diagrams]
            model = lambeq.PennyLaneModel.from_diagrams(circuits)
            model.initialise_weights()
            circuit = model.circuit_map[circuits[0]]
            circuit.initialise_concrete_params(model.symbol_weight_map)
            print(f'{ansatze_names[i]} lambeq circuit : ', '\n')
            print(circuit.draw())
            print('\n')
            circuit = alpha_3_ansatze_list[i](
            n_qubits, n_layers, axis_embedding, observables)
            params = torch.nn.Parameter(torch.randn(circuit.parameters_shape))
            print('Alpha 3 circuit params are : ', '\n', params)
            c = circuit.run_and_measure(circuit.circuit_function)
            print(f'Alpha3 {ansatze_names[i]} circuit', '\n')
            print(qml.draw(c)(input, params))
            print('\n\n\n')


    def test_n_shots(self):
        """
        Test kwargs usage of Circuit class to add number of shots
        """
        n_qubits = 2 ; n_layers = 2 ; axis_embedding = 'Z'
        observables = [qml.PauliZ, qml.PauliY, qml.PauliX]
        circuit = Sim14(
            n_qubits, n_layers, axis_embedding, observables)
        params = torch.nn.Parameter(torch.randn(circuit.parameters_shape))
        input = torch.nn.Parameter(torch.randn(n_qubits))
        circuit_function = circuit.circuit_function
        results = circuit.run_and_measure(circuit_function)(input, params)
        ### We make sure the output has the proper format
        for r in results:
            self.assertIsInstance(r, torch.Tensor)

    def test_rescaling_function(self):
        """
        Test the usage of data_mapping in the inputs to embed
        """
        n_qubits = 2 ; n_layers = 1 ; axis_embedding = 'X'
        observables = [qml.PauliZ, qml.PauliY, qml.PauliX]
        def data_rescaling(x):
            return 2 * mt.pi * x
        circuit = Sim15(
            n_qubits, n_layers,
            axis_embedding, observables,
            data_rescaling = data_rescaling)
        params = torch.nn.Parameter(torch.randn(circuit.parameters_shape))
        input = torch.nn.Parameter(torch.randn(n_qubits))
        circuit_function = circuit.circuit_function
        results = circuit.run_and_measure(circuit_function)(input, params)
        ### We make sure the output has the proper format 
        for r in results:
            self.assertIsInstance(r, torch.Tensor)

    def test_observables(self):
        """
        Test the usage of observables other than Pauli operators
        """
        n_qubits = 3 ; n_layers = 1 ; axis_embedding = 'Y'
        observables = [qml.Hadamard, qml.PauliY, qml.PauliX]
        def data_rescaling(x):
            return 2 * mt.pi * x
        circuit = StronglyEntangling(
            n_qubits, n_layers,
            axis_embedding, observables,
            data_rescaling = data_rescaling)
        params = torch.nn.Parameter(torch.randn(circuit.parameters_shape))
        input = torch.nn.Parameter(torch.randn(n_qubits))
        circuit_function = circuit.circuit_function
        results = circuit.run_and_measure(circuit_function)(input, params)
        ### We make sure the output has the proper format 
        for r in results:
            self.assertIsInstance(r, torch.Tensor)



if __name__ == "__main__":
    unittest.main()
