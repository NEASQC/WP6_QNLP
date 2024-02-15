"""
QuantumKMeans
=============
Module containing the base class for the k-means algorithm using
the quantum distance.
"""
import numpy as np 

from pyclustering.cluster.center_initializer import (
    kmeans_plusplus_initializer, random_center_initializer)
from pyclustering.cluster.kmeans import kmeans, kmeans_observer
from pyclustering.utils.metric import distance_metric, type_metric

from quantum_distance import QuantumDistance as qd
from utils import normalise_vector, pad_vector_with_zeros


class QuantumKMeans:
    """
    Class for implementing the K-Means algorithm 
    using a quantum distance.
    """

    def __init__(
        self, x_train : np.array, k : int,
        tolerance : float = 0.001, itermax : int = 200
    )-> None:
        """
        Initialise the class.

        Parameters
        ----------
        x_train : list[np.array]
            List with sentence vectors. 
        k : int
            Number of clusters.
        tolerance : float
            Value that dictates stopping condition. If for a given iteration,
            the sum of distances of each item wrt to its cluster
            is less than the tolerance, the algorithm stops.
        itermax : int
            Number of maximum iterations of the algorithm during the training
            phase.
        """
        self.x_train = x_train
        self.k = k 
        self.metric = distance_metric(
            type_metric.USER_DEFINED, func = self.wrapper_quantum_distance
        )
        self.observer = kmeans_observer()
        self.tolerance = tolerance
        self.itermax = itermax

    @staticmethod
    def wrapper_quantum_distance(vector_1 : np.array, vector_2 : np.array)-> float:
        """
        Wrapper for implementing the quantum distance as a metric of
        the k-means algorithm.

        Parameters
        ----------
        vector_1 : np.array
            First of the vectors between which to compute the distance.
        vector_2 : np.array
            Second of the vectors between which to compute the distance.
        
        Returns
        -------
        float
            Quantum distance between the two vectors.
        """
        vector_1 = normalise_vector(vector_1)
        vector_1 = pad_vector_with_zeros(vector_1)
        vector_2 = normalise_vector(vector_2)
        vector_2 = pad_vector_with_zeros(vector_2)
        return qd(vector_1, vector_2).compute_quantum_distance()
    
    def initialise_cluster_centers(
        self, random_initialisation = True,
        seed = 30031935
    )-> None:
        """
        Initialise the cluster centers for the algorithm.

        Parameters
        ----------
        random_center_initializer : bool
            If True randomly initialises the cluster centers.
            If False, uses the K-Means++ algorirthm to initialise
            cluster center. More info can be found in pycluster docs
            https://pyclustering.github.io/docs/0.8.2/html/index.html.
        seed : int
            Seed to use when random_center_intializer = True.
        """
        if random_initialisation == True:
            self.initial_centers = random_center_initializer(
                self.x_train, self.k, random_state = seed
            ).initialize()
        else:
            self.initial_centers = kmeans_plusplus_initializer(
                self.x_train, self.k).initialize()

    def train_k_means_algorithm(self)-> None:
        """
        Instantiate and train k-means algorithm. Assign the clusters, 
        centers and n_iterations as attributes of the class.
        """
        self.k_means_instance = kmeans(
            self.x_train, self.initial_centers,
            tolerance = self.tolerance, observer = self.observer,
            metric = self.metric, itermax = self.itermax
        )
        self.k_means_instance.process()
        self.clusters = self.observer._kmeans_observer__evolution_clusters
        self.centers = self.observer._kmeans_observer__evolution_centers
        self.n_its = len(self.clusters)
    
    def compute_wce(self)-> None:
        """
        For each iteration of the algorithm training process, compute the within cluster
        sum of errors:  
        $[error=\sum_{i=0}^{N}quantum_distance(x_{i}-center(x_{i}))$].
        """
        self.total_wce = []
        for i in range(self.n_its):
            wce = 0
            for j in range(self.k):
                for item in self.clusters[i][j]:
                    wce += self.wrapper_quantum_distance(
                        self.x_train[item], self.centers[i][j]
                    )
            self.total_wce.append(wce)

    def compute_predictions(self)-> None:
        """
        For each iteration of the algorithm training process, compute 
        the predictions of the trainining instances, i.e.,
        what the index of their closest cluster is. 
        """
        self.predictions = [
            [None for _ in range(len(self.x_train))] for _ in range(
                self.n_its)]
        for i in range(self.n_its):
            for j, cluster in enumerate(self.clusters[i]):
                for item in cluster:
                    self.predictions[i][item] = j

        
    

