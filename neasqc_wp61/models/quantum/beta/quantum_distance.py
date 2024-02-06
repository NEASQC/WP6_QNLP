"""
QuantumDistance
===============
Module containing the base class for computing the quantum distance
"""

import cmath

import numpy as np 

from qiskit import QuantumCircuit, execute
from qiskit_aer import AerSimulator
from qiskit.circuit.controlledgate import ControlledGate
from qiskit_aer import AerSimulator

class QuantumDistance:
    """
    Class for implementing algorithm for the quantum distance based on 
    the SWAP test (see [1] for further reference)
    **Reference**
    [1] Sarma, Abhijat, et al. "Quantum unsupervised and supervised
    learning on superconducting processors."
    arXiv preprint arXiv:1909.04226 (2019).
    """   
    def __init__(self, x1 : np.array, x2 : np.array) -> None:
        """
        Initialiser of the class

        Parameters
        ----------
        x1 : np.array
            First of the vectors to compute the distance between
        x2 : np. array
            Second of the vectors to compute the distance between
        """
        if len(x1) != len(x2):
            raise ValueError(
                'x1 and x2 need to have the same length'
            )
        self.x1 = x1
        self.x2 = x2
        self.normalise_vectors()
        if len(self.x1) != 2 * int(np.ceil(np.log2(len(self.x1)))):
            self.pad_with_zeros()

        # Number of qubits needed to encode our vectors
        self.nq_encoding = int(np.ceil(np.log2(len(x1))))
        # Number of total qubits in the quantum circuit
        self.nq = self.nq_encoding + 3
        self.qc = QuantumCircuit(self.nq)

    def normalise_vectors(self):
        """
        Normalises x1,x2 so that their norm is equal to 1
        """
        self.x1 = self.x1/np.linalg.norm(self.x1)
        self.x2 = self.x2/np.linalg.norm(self.x2)
    
    def pad_with_zeros(self):
        """
        Pads vectors x1,x2 with zeros until their lengths
        are powers of 2
        """
        n = len(self.x1)
        next_power_of_2 = 2 ** int(np.ceil(np.log2(n)))
        zero_padding = np.zeros(next_power_of_2 - n)
        self.x1 = np.concatenate((self.x1, zero_padding), axis = 0)
        self.x2 = np.concatenate((self.x2, zero_padding), axis = 0)

    def build_gate_state_preparation(self, x : np.array) -> ControlledGate:
        """
        Builds a gate that embbeds the vector
        x using amplitude encoding. The gate will be controlled 
        on an ancilla qubit. 

        Parameters
        ----------
        x : np.array
            The vector to encode. Must be normalised so that the 
            sum of the squares of its components is equal to 1.
        
        Returns
        -------
        xgate : ControlledGate
            A controlled gate that embbeds x using amplitude encoding
        """
        qcx = QuantumCircuit(self.nq_encoding)
        qcx.prepare_state(x.tolist(), range(self.nq_encoding))
        xgate = qcx.to_gate().control(1)
        return xgate
    
    def build_psi_state(self) -> None:
        """
        Builds the \psi state defined in [1]
        """
        x1gate = self.build_gate_state_preparation(self.x1)
        x2gate = self.build_gate_state_preparation(self.x2)
        self.qc.h(0)
        self.qc.append(x2gate, range(self.nq_encoding + 1))
        self.qc.x(0)
        self.qc.append(x1gate, range(self.nq_encoding + 1))
        self.qc.x(0)

    def build_phi_state(self) -> None:
        """
        Builds the \phi state defined in [1]
        """
        phi = np.array([1/np.sqrt(2), -1/np.sqrt(2)])
        self.qc.prepare_state(phi, self.nq_encoding + 1)
    
    def apply_swap_test(self):
        """
        Applies a swap test between the first qubit of state \psi 
        and the state \phi
        """
        self.qc.h(self.nq_encoding + 2)
        self.qc.cswap(self.nq_encoding + 2, 0, self.nq_encoding + 1)
        self.qc.h(self.nq_encoding + 2)
    
    def execute_qc(
        self, method : str = 'statevector', device : str = 'CPU'
    ) -> None:
        """
        Executes the qc on a simulator. 

        Parameters
        ----------
        method : str, default : statevector
            Simulator method to use. More info can be found in
            https://github.com/Qiskit/qiskit-aer/blob/main/qiskit_aer/backends/aer_simulator.py
        device : str, default : CPU
            Decides whether to use CPU or GPU.

        """
        self.qc.save_statevector()
        sim = AerSimulator(method=method, device=device)
        result = execute(self.qc, sim, shots=None).result()
        self.state_vector = result.get_statevector()  
        self.state_probabilities = self.state_vector.probabilities_dict() 

    def compute_quantum_distance(
        self, method : str = 'statevector', device : str = 'CPU'
    ) -> float:
        """
        Executes the full pipeline for computing the distance

        Parameters
        ----------
        method : str, default : statevector
            Simulator method to use. More info can be found in
            https://github.com/Qiskit/qiskit-aer/blob/main/qiskit_aer/backends/aer_simulator.py
        device : str, default : CPU
            Decides whether to use CPU or GPU.

        Returns
        -------
        quantum_distance : str
            Estimation of the quantum distance
        """
        self.build_psi_state()
        self.build_phi_state()
        self.apply_swap_test()
        self.execute_qc(method=method, device=device)
        quantum_distance = self.distance_prob_relation(self.state_probabilities)
        return quantum_distance

    @staticmethod
    def distance_prob_relation(probabilities_dict : dict) -> float:
        """
        Computes the relation between the probability of measuring 0
        in the ancilla qubit of the SWAP test, and the euclidean distance
        as seen in [1].

        Parameters
        ----------
        probabilities_dict : dict
            Dictionary with the probabilities of each state
        
        Returns
        -------
        quantum_distance : 
            Estimation of the quantum distance
        """
        prob0 = 0
        for k,v in probabilities_dict.items():
            if k[0] == '0':
                prob0 += v
        quantum_distance = abs(cmath.sqrt((8 * prob0 -4)))
        return quantum_distance
        

    


    


          
