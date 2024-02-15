"""
QuantumDistance
===============
Module containing the base class for computing the quantum distance.
"""

import cmath

import numpy as np 
from qiskit import execute, QuantumCircuit
from qiskit.circuit.controlledgate import ControlledGate
from qiskit_aer import AerSimulator


class QuantumDistance:
    """
    Class for implementing algorithm for the quantum distance based on 
    the SWAP test (see [1] for further reference).
    **Reference**
    [1] Sarma, Abhijat, et al. "Quantum unsupervised and supervised
    learning on superconducting processors."
    arXiv preprint arXiv:1909.04226 (2019).
    """   
    def __init__(self, vector_1 : np.array, vector_2 : np.array)-> None:
        """
        Initialise the class.

        Parameters
        ----------
        vector_1 : np.array
            First of the vectors between which the distance is calculated.
        vector_2: np. array
            Second of the vectors between which the distance is calculated.
        """
        self.vector_1 = vector_1
        self.vector_2 = vector_2
        self.verify_vectors_satisfy_conditions()
        self.n_qubits_to_encode_vector = int(
            np.ceil(np.log2(len(self.vector_1)))
        )
        # Number of total qubits in the quantum circuit.
        self.n_qubits_quantum_circuit = self.n_qubits_to_encode_vector + 3
        self.qc = QuantumCircuit(self.n_qubits_quantum_circuit)

    def verify_vectors_satisfy_conditions(self)-> None:
        """
        Checks if vector_1 and vector_2 satisfy the conditions needed 
        in order to compute the quantum distance between them.
        """
        if len(self.vector_1) != len(self.vector_2):
            raise ValueError('Vectors must be the same length.')
        elif (
            abs(np.linalg.norm(self.vector_1) - 1.0) > 1e-09
            or (np.linalg.norm(self.vector_1) - 1.0) > 1e-09
        ):
            raise ValueError('Both vectors must have norm 1.')
        elif (
            not np.log2(len(self.vector_1)).is_integer()
            or not np.log2(len(self.vector_2)).is_integer()
        ):
            raise ValueError('Both vectors lengths must be power of 2.')
        else:
            pass
        
    def build_gate_state_preparation(
        self, vector : np.array
    )-> ControlledGate:
        """
        Build a gate that embbed a vector using amplitude encoding
        The gate will be controlled on an ancilla qubit. 

        Parameters
        ----------
        x : np.array
            The vector to encode. Must be normalised so that the 
            sum of the squares of its components is equal to 1.
        
        Returns
        -------
        xgate : ControlledGate
            A controlled gate that embbeds x using amplitude encoding.
        """
        qcx = QuantumCircuit(self.n_qubits_to_encode_vector)
        qcx.prepare_state(
            vector.tolist(), range(self.n_qubits_to_encode_vector)
        )
        xgate = qcx.to_gate().control(1)
        return xgate
    
    def build_psi_state(self)-> None:
        """
        Build the \psi state defined in [1].
        """
        vector_1_embedding_gate = self.build_gate_state_preparation(
            self.vector_1
        )
        vector_2_embedding_gate = self.build_gate_state_preparation(
            self.vector_2
        )
        self.qc.h(0)
        self.qc.append(
            vector_2_embedding_gate, range(self.n_qubits_to_encode_vector + 1)
        )
        self.qc.x(0)
        self.qc.append(
            vector_1_embedding_gate, range(self.n_qubits_to_encode_vector + 1)
        )
        self.qc.x(0)

    def build_phi_state(self)-> None:
        """
        Build the \phi state defined in [1].
        """
        phi = np.array([1/np.sqrt(2), -1/np.sqrt(2)])
        self.qc.prepare_state(phi, self.n_qubits_to_encode_vector + 1)
    
    def apply_swap_test(self)-> None:
        """
        Apply a swap test between the first qubit of state \psi 
        and the state \phi.
        """
        self.qc.h(self.n_qubits_to_encode_vector + 2)
        self.qc.cswap(
            self.n_qubits_to_encode_vector + 2, 0, self.n_qubits_to_encode_vector + 1
        )
        self.qc.h(self.n_qubits_to_encode_vector + 2)
    
    def execute_qc(
        self, method : str = 'statevector', device : str = 'CPU'
    )-> None:
        """
        Execute the qc on a simulator. 

        Parameters
        ----------
        method : str, default : statevector
            Simulator method to use. More info can be found in
            https://github.com/Qiskit/qiskit-aer/blob/main/qiskit_aer/backends/aer_simulator.py.
        device : str
            Decides whether to use CPU or GPU. (Default = CPU).
        """
        self.qc.save_statevector()
        sim = AerSimulator(method=method, device=device)
        result = execute(self.qc, sim, shots=None).result()
        self.state_vector = result.get_statevector()  
        self.state_probabilities = self.state_vector.probabilities_dict() 

    def compute_quantum_distance(
        self, method : str = 'statevector', device : str = 'CPU'
    )-> float:
        """
        Execute the full pipeline for computing the distance.

        Parameters
        ----------
        method : str, default : statevector
            Simulator method to use. More info can be found in
            https://github.com/Qiskit/qiskit-aer/blob/main/qiskit_aer/backends/aer_simulator.py.
        device : str, default : CPU
            Decides whether to use CPU or GPU. (Default = CPU).

        Returns
        -------
        quantum_distance : str
            Estimation of the quantum distance.
        """
        self.build_psi_state()
        self.build_phi_state()
        self.apply_swap_test()
        self.execute_qc(method=method, device=device)
        quantum_distance = self.distance_prob_relation(
            self.state_probabilities
        )
        return quantum_distance

    @staticmethod
    def distance_prob_relation(probabilities_dict : dict)-> float:
        """
        Compute the relation between the probability of measuring 0
        in the ancilla qubit of the SWAP test, and the euclidean distance
        as seen in [1].

        Parameters
        ----------
        probabilities_dict : dict
            Dictionary with the probabilities of each state.
        
        Returns
        -------
        float 
            Estimation of the quantum distance.
        """
        prob0 = 0
        for k,v in probabilities_dict.items():
            if k[0] == '0':
                prob0 += v
        quantum_distance = abs(cmath.sqrt((8 * prob0 -4)))
        return quantum_distance
        

    


    


          
