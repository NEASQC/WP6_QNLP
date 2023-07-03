import numpy as np
import qiskit 
from qiskit_aer import AerSimulator
import cmath

class QuantumDistance:
    """
    Class for implementing a simplistic version of quantum distance
    """
    def __init__(self, x1 : np.array, x2 : np.array) -> None:
        """
        Initialiser of the class

        Parameters
        ----------
        x1 : np.array
            First vector
        x2 : np.array
            Second vector
        """
        self.x1norm = self.normalise_vectors(x1)
        self.x2norm = self.normalise_vectors(x2)
        self.circuit = self.build_circuit(
            self.x1norm, self.x2norm)
        self.counts = self.get_results_qc_shots(self.circuit)
        self.dist = self.euclidean_probability_relation(self.counts)
        self.real_dist = self.euclidean_distance(
            self.x1norm, self.x2norm)

    def normalise_vectors(
        self, x : np.array) -> np.array:
        """
        Normalises a vector [x1, x2] so that (x1**2 + x2**2) =1 

        Parameters 
        ----------
        x : np.array
            Vector we want to normalise
        
        Returns
        -------
        x_norm : np.array
            Normalised vector
        """
        x_norm = np.array([])
        Z = np.sum([x[j]**2 for j in range(len(x))])
        # Normalisation constant
        for i in range(len(x)):
            x_norm = np.append(
                x_norm, x[i]/np.sqrt(Z))
        return x_norm
    
    def euclidean_distance(
        self, x1 : np.array, x2 : np.array
    ) -> float:
        """
        Computes the real euclidean distance between two vectors

        Parameters
        ----------
        x1 : np.array
            First vector
        x2 : np.array
            Second vector
        
        Returns
        -------
        euclidean_distance : float
            The real euclidean distance
        """
        euclidean_distance = np.sqrt(
            np.sum((x1[i]-x2[i])**2 for i in range(len(x1))))
        return euclidean_distance
    
    def build_circuit(
        self, x1 : np.array, x2 : np.array
    ) -> qiskit.QuantumCircuit:
        """
        Builds the circuit with the encoding of the two vectors and the SWAP
        test

        Parameters
        ----------
        x1 : np.array
            First vector
        x2 : np.array
            Second vector

        Returns
        -------
        qc : qiskit.QuantumCircuit
            The circuit implementing the SWAP test    
        """
        theta1 = 2*np.arcsin(x1[1])
        theta2 = 2*np.arcsin(x2[1])

        qc = qiskit.QuantumCircuit(3,1)
        qc.ry(theta1, 1)
        qc.ry(theta2, 2)
        qc.barrier()
        qc.h(0)
        qc.cswap(0,1,2)
        qc.h(0)
        qc.measure(0,0)
        return qc
    
    def get_results_qc_shots(
        self, qc : qiskit.QuantumCircuit, shots = 2**10,
        backend = AerSimulator()
    ) -> dict:
        """
        Gets the results of running the circuit in dictionary format

        Parameters
        ----------
        qc : qiskit.QuantumCircuit
            The circuit we want to analyse
        shots : int, default : 2**10
            The number of shots to perform
        backend : callable, default : AerSimulator
            The quantum backend where circuits are run
        
        Returns
        -------
        counts : dict
            Dictionary containing the number of times each
            state appears
        """
        qc_compiled = qiskit.transpile(qc, backend)
        job = backend.run(qc_compiled, shots = shots)
        results = job.result()
        counts = results.get_counts(qc_compiled)
        return counts
    
    def euclidean_probability_relation(
        self, counts
    ) -> float:
        """
        For normalised vectors in the SWAP test, 
        computes the relation of the probability between 
        obtaining a 0 in the control qubit and the euclidean
        distance.

        Parameters
        ----------
        counts : dict
            Dictionary with the quantum states as keys 
            and the number of times they were obtained as values
        
        Returns
        -------
        dist : float 
            Euclidean distance computed from SWAP test
        """
        if '1' in counts.keys():
            p0 = counts['0']/ (counts['0'] + counts['1'])
        else : 
            p0 = 1
    
        dist = np.sqrt(2 - 2 * abs(cmath.sqrt(2 * p0 -1)))
        return dist
    
    


