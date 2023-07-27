from QuantumDistance import QuantumDistance as qd
import pandas as pd 
import json
from collections import Counter
import numpy as np 


class QuantumKNearestNeighbours:
    """
    Class for implementing the K Nearest Neighbors algorithm 
    using the quantum distance
    """
    
    def __init__(
            self, dataset_dir : str, train_vectors :list,
            test_vectors : list, k : int
        )-> None:
        """
        Initialises the class. 

        Parameters
        ----------
        dataset_dir : str
            Directory where the dataset with the
            training  labels is stored
        vectors_train_dir : str
            List containig the vectors for training
        vectors_test_dir : str
            List containing the vectors for testing
        k : int
            Number of neighbors of the algorithm
        """
        self.labels = self.load_labels(dataset_dir)
        self.train_vectors = train_vectors
        self.test_vectors = test_vectors
        self.k = k
        self.predictions = []
        for i in self.test_vectors:
            distances = self.compute_distances(i)
            closest_indexes = self.compute_minimum_distances(distances, self.k)
            pred = self.labels_majority_vote(closest_indexes)
            self.predictions.append(pred)
        

    @staticmethod
    def load_labels(
            dataset : str  
    ) -> list[list[str],list[list[int]]]:
        """
        Loads the chosen dataset as pandas dataframe.

        Parameters
        ----------
        dataset : str
            Directory where the dataset is stored

        Returns
        -------
        labels: list[int]
            List with the labels of the dataset.
            0 False, and 1 True
        """
        df = pd.read_csv(
            dataset, sep='\t+',
            header=None, names=['label', 'sentence', 'structure_tilde'],
            engine='python'
        )
        labels = []
        sentences = df['sentence'].tolist()

        for i in range(df.shape[0]):
            if df['label'].iloc[i] == 1:
                labels.append(0)
            else:
                labels.append(1)

        return labels

    def compute_distances(self, sample : np.array):
        """
        For a given vector, computes its distance to all other vectors in
        the dataset

        Parameters
        ----------
        sample : np.array
            The vector to which we are computing the distance

        Returns
        -------
        distances : list[float]
            The distance of the sample vector to all other vectors in
            training dataset
        """
        distances = []
        for vector in(self.train_vectors):
            distances.append(qd(sample, vector).dist)
        return distances

    @staticmethod
    def compute_minimum_distances(distances, k):
        """
        Computes the indexes of the k closest elements of the training 
        dataset

        Parameters
        ----------
        distances : list[float]
            List with the distances to all elements of the training dataset
        k : int
            Number of selected k neighbors

        Returns 
        -------
        closest_indexes : list[int]
            indexes of the k closest vectors
        """
        closest_indexes = sorted(
            range(len(distances)), key=lambda j: distances[j]
            )[:k]
        return closest_indexes
    
    def labels_majority_vote(self, closest_indexes):
        """
        Gets the labels of the closest vectors and returns the most 
        frequent one among them 

        Parameters 
        ----------
        closest_indexes : list[int]
            List with the indexes of the k closest vectors
        
        Returns
        -------
        label : int
            Most frequent label among the k closest vectors
        """
        closest_labels = []
        for i in closest_indexes:
            closest_labels.append(self.labels[i])
        c = Counter(closest_labels)
        label, count = c.most_common()[0]
        return label





    