"""
Circuit
=======
Module containing the base class for the variational circuits of Alpha3 model

"""
from abc import ABC, abstractmethod
from typing import Callable

import pennylane as qml
from pennylane.measurements.expval import ExpectationMP
import torch

class Circuit(ABC):
    """
    Base class for circuits
    """
    def __init__(
        self, n_qubits : int,  n_layers : int,
        axis_embedding : str, observables : list[qml.operation.Operator],
        device_name : str = "default.qubit", data_rescaling : Callable = None,
        **kwargs
    )-> None:
        """
        Initialises the circuit class

        Parameters 
        ----------
        n_qubits : int
            Number of qubits of the circuit
        n_layers : int
            Number of times the ansatz is applied to the circuit
        axis_embedding : str 
            Rotation gate to use for the angle encoding of the inputs
            Must be one of ['X','Y','Z']
        observables : list[qml.operation.Operator]
            List with Pennylane operators, one acting on each qubit
            The circuit will output the expected value of each 
            of the operators
        device_name : str, default = "default.qubit"
            Pennylane simulator to use. The available devices can be found in
            https://docs.pennylane.ai/en/stable/code/api/pennylane.device.html
        data_rescaling : Callable, default = None
            Function to apply to rescale the inputs that will be encoded
        ** kwargs 
            Keyword arguments to be introduced to the pennylane device
            More info can be found in 
            https://docs.pennylane.ai/en/stable/code/api/pennylane.device.html
        
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.axis_embedding = axis_embedding
        self.observables = observables
        self.device = qml.device(
            device_name, wires=self.n_qubits, **kwargs)
        self.data_rescaling = data_rescaling

    @abstractmethod
    def build_circuit_function(
        self, input :torch.tensor, params : torch.tensor
    )-> list[ExpectationMP]:
        """
        Builds the circuit function that can be run using pennylane simulators

        Parameters
        ----------
        input : torch.tensor
            Input features introduced. These will be encoded in the circuit 
            using angle encoding techniques
        params : torch.tensor
            Variational paramaters of the ansatz. Will be optimised
            within the model

        Returns
        -------
        list[ExpectationMP]
            List with expectation values (one for each qubit) of some
            observables
        """

    def run_and_measure_circuit(self, circuit_function : Callable)->qml.QNode:
        """
        Builds a quantum node containing a circuit function (output
        of build_circuit_function) and device to
        be run on. More info can be found in 
        https://docs.pennylane.ai/en/stable/code/api/pennylane.qnode.html

        Parameters
        ----------
        circuit_function : Callable
            Circuit function
        
        Returns
        -------
        qnode : qml.QNode
            Pennylane Quantum node. When called, it outputs the results of 
            the measurements in our circuit function
        """
        qnode = qml.QNode(circuit_function, self.device, interface = "torch")
        return qnode
    

class Sim14(Circuit):
    """
    Class containing ansatz 14 of
    https://arxiv.org/pdf/1905.10876.pdf
    with minor modificatinons in the entanglement structure
    """
    def __init__(
        self, n_qubits : int,  n_layers : int,
        axis_embedding : str, observables : list[qml.operation.Operator],
        device_name : str = "default.qubit", data_rescaling : Callable = None,
        **kwargs
    )-> None:
        """
        Initialises the Sim14 class

        Parameters 
        ----------
        n_qubits : int
            Number of qubits of the circuit
        n_layers : int
            Number of times the ansatz is applied to the circuit
        axis_embedding : str 
            Rotation gate to use for the angle encoding of the inputs
            Must be one of ['X','Y','Z']
        oservables : list[qml.operation.Operator]
            List with Pennylane operators, one acting on each qubit
            The circuit will output the expected value of each 
            of the operators
        device_name : str, default = "default.qubit"
            Pennylane simulator to use. The available devices can be found in
            https://docs.pennylane.ai/en/stable/code/api/pennylane.device.html
        data_rescaling : Callable, default = None
            Function to apply to rescale the inputs that will be encoded
        ** kwargs 
            Keyword arguments to be introduced to the pennylane device
            More info can be found in 
            https://docs.pennylane.ai/en/stable/code/api/pennylane.device.html
        
        """
        super().__init__(
            n_qubits, n_layers, axis_embedding, observables,
            device_name, data_rescaling, **kwargs
        )
        self.parameters_shape = (n_layers, 4  * n_qubits)

    def build_circuit_function(
        self, inputs : torch.Tensor,
        params : torch.Tensor
    ) -> list[ExpectationMP]:
        """
        Builds the circuit function for Sim14 class

        Parameters
        ----------
        input : torch.tensor
            Input features introduced. These will be encoded in the circuit 
            using angle encoding techniques
        params : torch.tensor
            Variational paramaters of the ansatz. They will be optimised
            within the model

        Returns
        -------
        list[ExpectationMP]
            List with expectation values (one for each qubit) of some
            observables
        """
        if self.data_rescaling is not None:
            inputs = self.data_rescaling(inputs)
        
        qml.AngleEmbedding(
            features=inputs,
            wires = range(self.n_qubits),
            rotation = self.axis_embedding
        )
        
        for i in range(self.n_layers):
            idx = 0
            for j in range(self.n_qubits):
                qml.RY(params[i, j + idx], wires = j)
            idx += self.n_qubits
            for j in range(self.n_qubits):
                ctrl = j
                tgt = (j - 1) % self.n_qubits
                qml.CRX(phi=params[i, j + idx], wires=[ctrl,tgt])
            idx += self.n_qubits
            for j in range(self.n_qubits):
                qml.RY(params[i, j + idx], wires = j)
            idx += self.n_qubits
            for j in range(self.n_qubits, 0, -1):
                ctrl = j % self.n_qubits
                tgt = (j + 1) % self.n_qubits
                qml.CRX(params[i, j + idx -1], wires= [ctrl, tgt])
        return [qml.expval(
            self.observables[k](k)) for k in range (self.n_qubits)]
        
class Sim15(Circuit):
    """
    Class containing ansatz 15 of
    https://arxiv.org/pdf/1905.10876.pdf
    with minor modificatinons in the entanglement structure
    """
    def __init__(
        self, n_qubits : int,  n_layers : int,
        axis_embedding : str, observables : list[qml.operation.Operator],
        device_name : str = "default.qubit", data_rescaling : Callable = None,
        **kwargs
    )-> None:
        """
        Initialises the Sim15 class

        Parameters 
        ----------
        n_qubits : int
            Number of qubits of the circuit
        n_layers : int
            Number of times the ansatz is applied to the circuit
        axis_embedding : str 
            Rotation gate to use for the angle encoding of the inputs
            Must be one of ['X','Y','Z']
        oservables : list[qml.operation.Operator]
            List with Pennylane operators, one acting on each qubit
            The circuit will output the expected value of each 
            of the operators
        device_name : str, default = "default.qubit"
            Pennylane simulator to use. The available devices can be found in
            https://docs.pennylane.ai/en/stable/code/api/pennylane.device.html
        data_rescaling : Callable, default = None
            Function to apply to rescale the inputs that will be encoded
        ** kwargs 
            Keyword arguments to be introduced to the pennylane device
            More info can be found in 
            https://docs.pennylane.ai/en/stable/code/api/pennylane.device.html
        
        """
        
        super().__init__(
            n_qubits, n_layers, axis_embedding, observables,
            device_name, data_rescaling, **kwargs
        )
        self.parameters_shape = (n_layers, 2  * n_qubits)
    
    def build_circuit_function(
        self, inputs : torch.Tensor,
        params : torch.Tensor
    ) -> list[ExpectationMP]:
        """
        Builds the circuit function for Sim15 class

        Parameters
        ----------
        input : torch.tensor
            Input features introduced. These will be encoded in the circuit 
            using angle encoding techniques
        params : torch.tensor
            Variational paramaters of the ansatz. They will be optimised
            within the model

        Returns
        -------
        list[ExpectationMP]
            List with expectation values (one for each qubit) of some
            observables
        """
        if self.data_rescaling is not None:
            inputs = self.data_rescaling(inputs)

        qml.AngleEmbedding(
            features=inputs,
            wires = range(self.n_qubits),
            rotation = self.axis_embedding
        )

        for i in range(self.n_layers):
            idx = 0
            for j in range(self.n_qubits):
                qml.RY(params[i, j], wires = j)
            idx += self.n_qubits
            for j in range(self.n_qubits):
                ctrl = j
                tgt = (j - 1) % self.n_qubits
                qml.CNOT(wires=[ctrl,tgt])
            for j in range(self.n_qubits):
                qml.RY(params[i, j + idx], wires = j)
            for j in range(self.n_qubits, 0, -1):
                ctrl = j % self.n_qubits
                tgt = (j + 1) % self.n_qubits
                qml.CNOT(wires= [ctrl, tgt])

        return [qml.expval(
            self.observables[k](k)) for k in range (self.n_qubits)]

        
class StronglyEntangling(Circuit):
    """
    Class containing StronglyEntanglingAnstaz of
    https://docs.pennylane.ai/en/stable/code/api/pennylane.StronglyEntanglingLayers.html
    """
    def __init__(
        self, n_qubits : int,  n_layers : int,
        axis_embedding : str, observables : list[qml.operation.Operator],
        device_name : str = "default.qubit", data_rescaling : Callable = None,
        **kwargs
    )-> None:
        """
        Initialises the StronglyEntangling class

        Parameters 
        ----------
        n_qubits : int
            Number of qubits of the circuit
        n_layers : int
            Number of times the ansatz is applied to the circuit
        axis_embedding : str 
            Rotation gate to use for the angle encoding of the inputs
            Must be one of ['X','Y','Z']
        oservables : list[qml.operation.Operator]
            List with Pennylane operators, one acting on each qubit
            The circuit will output the expected value of each 
            of the operators
        device_name : str, default = "default.qubit"
            Pennylane simulator to use. The available devices can be found in
            https://docs.pennylane.ai/en/stable/code/api/pennylane.device.html
        data_rescaling : Callable, default = None
            Function to apply to rescale the inputs that will be encoded
        ** kwargs 
            Keyword arguments to be introduced to the pennylane device
            More info can be found in 
            https://docs.pennylane.ai/en/stable/code/api/pennylane.device.html
        
        """
        
        super().__init__(
            n_qubits, n_layers, axis_embedding, observables,
            device_name, data_rescaling, **kwargs
        )
        self.parameters_shape = (n_layers, 3  * n_qubits)

    def build_circuit_function(
        self, inputs : torch.Tensor,
        params : torch.Tensor
    ) -> list[ExpectationMP]:
        """
        Builds the circuit function for StronglyEntangling class

        Parameters
        ----------
        input : torch.tensor
            Input features introduced. These will be encoded in the circuit 
            using angle encoding techniques
        params : torch.tensor
            Variational paramaters of the ansatz. They will be optimised
            within the model

        Returns
        -------
        list[ExpectationMP]
            List with expectation values (one for each qubit) of some
            observables
        """
        if self.data_rescaling is not None:
            inputs = self.data_rescaling(inputs)

        qml.AngleEmbedding(
            features=inputs,
            wires = range(self.n_qubits), 
            rotation = self.axis_embedding)

        for i in range(self.n_layers):
            idx = 0
            for j in range(self.n_qubits):
                qml.RZ(params[i, j + idx], wires = j)
                qml.RY(params[i, j + idx + 1], wires = j)
                qml.RZ(params[i, j + idx + 2], wires = j)
                idx += 2
            for j in range(self.n_qubits - 1):
                ctrl = j 
                tgt = (j + 1) 
                qml.CNOT(wires= [ctrl, tgt])
            qml.CNOT(wires = [self.n_qubits - 1, 0])
        return [qml.expval(
            self.observables[k](k)) for k in range (self.n_qubits)]
            
            