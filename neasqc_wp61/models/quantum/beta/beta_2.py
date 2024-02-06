import numpy as np 

from quantum_distance import QuantumDistance as qd
from pyclustering.cluster.kmeans import kmeans, kmeans_observer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import distance_metric, type_metric

class QuantumKMeans:
    """
    Class for implementing the K Means algorithm 
    using a quantum distance.
    """

    def __init__(
        self, x_train : np.array, k : int
    )-> None:
        """
        Initialises the class.

        Parameters
        ----------
        x_train : list[np.array]
            List with sentence vectors 
        k : int
            Number of clusters
        """
        self.x_train = x_train
        self.k = k 
        quantum_distance = lambda x,y : qd(x,y).compute_quantum_distance()
        self.metric = distance_metric(
            type_metric.USER_DEFINED, func = quantum_distance
        )
        self.initial_centers = kmeans_plusplus_initializer(
            self.X_train, self.k).initialize()

    def run_k_means_algorithm(self):
        """
        Instantiates and runs k-means algorithm
        """
        self.k_means_instance = kmeans(
            self.x_train, self.initial_centers, metric = self.metric
        )
        self.k_means_instance.process()

    def get_predictions(self) -> list[int]:
        """
        Gets the predictions for the train instances.

        Returns
        -------
        list[int]
            Predictions for each of the train instances. 
        """
        return self.k_means_instance.predict(self.x_train) 
    
    def get_total_metric_errors(self) -> float:
        """
        Gets the sum of metric errors with respect to quantum distance: 
        \f[error=\sum_{i=0}^{N}distance(x_{i}-center(x_{i}))\f]

        Returns
        -------
        float 
            Sum of metric errors wrt to quantum distance
        """
        return self.k_means_instance.get_total_wce()
    
    def get_clusters_centers(self) -> list[list]:
        """
        Gets the centers of each cluster

        Returns
        -------
        list [list]
            Centers of each of the clusters
        """
        return self.k_means_instace.get_centers()
    

    

