"""
QuantumKMeans
=============
Module containing the base class for the k-means algorithm using
the quantum distance.
"""
import numpy as np 

from pyclustering.cluster.center_initializer import (
    kmeans_plusplus_initializer, random_center_initializer)
from pyclustering.cluster.kmeans import kmeans
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
    )-> None:
        """
        Initialise the class.

        Parameters
        ----------
        x_train : list[np.array]
            List with sentence vectors. 
        k : int
            Number of clusters.
        """
        self.x_train = x_train
        self.k = k 
        self.metric = distance_metric(
            type_metric.USER_DEFINED, func = self.quantum_distance
        )

    @staticmethod
    def quantum_distance(x1 : np.array, x2 : np.array)-> float:
        """
        Wrapper for implementing the quantum distance as a metric of
        the k-means algorithm.

        Parameters
        ----------
        x1 : np.array
            First of the vectors between which to compute the distance.
        x2 : np.array
            Second of the vectors between which to compute the distance.
        
        Returns
        -------
        float
            Quantum distance between the two vectors.
        """
        x1 = normalise_vector(x1)
        x1 = pad_vector_with_zeros(x1)
        x2 = normalise_vector(x2)
        x2 = pad_vector_with_zeros(x2)
        return qd(x1, x2).compute_quantum_distance()
    
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
            https://pyclustering.github.io/docs/0.8.2/html/index.html
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

    def run_k_means_algorithm(self)-> None:
        """
        Instantiate and run k-means algorithm.
        """
        self.k_means_instance = kmeans(
            self.x_train, self.initial_centers, metric = self.metric
        )
        self.k_means_instance.process()

    def get_train_predictions(self)-> list[int]:
        """
        Get the predictions for the train instances.

        Returns
        -------
        list[int]
            Predictions for each of the train instances. 
        """
        return self.k_means_instance.predict(self.x_train) 
    
    def get_total_metric_error(self)-> float:
        """
        Get the sum of metric errors with respect to quantum distance: 
        \f[error=\sum_{i=0}^{N}distance(x_{i}-center(x_{i}))\f].

        Returns
        -------
        float 
            Sum of metric errors wrt to quantum distance.
        """
        return self.k_means_instance.get_total_wce()
    
    def get_clusters_centers(self)-> list[list]:
        """
        Get the centers of each cluster.

        Returns
        -------
        list [list]
            List with the centers of each of the clusters.
        """
        return self.k_means_instance.get_centers()
    

    

