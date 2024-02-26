"""
QuantumKNearestNeighbours
=========================
Module containing the base class for the k-nearest neighbours
algorithm using a quantum distance.
"""
import pickle
from collections import Counter

import numpy as np

from quantum_distance import QuantumDistance as qd


class QuantumKNearestNeighbours:
    """
    Class for implementing the k-nearest neighbours algorithm 
    using the quantum distance.
    """
    def __init__(
        self, vectors_train : list[np.array], vectors_test : list[np.array],
        labels_train : list[int], labels_test : list[int], k : int,
        seed : int = 300330
    )-> None:
        """
        Initialise the class.

        Parameters
        ----------
        vectors_train : list[np.array]
            Train vectors.
        vectors_test : list[np.array]
            Test vectors.
        labels_train : list[int]
            Train labels.
        labels_test = list[int]
            Test labels.
        k : int
            k closest vectors to make predictions with.
        seed : int
            Random seed. (Default = 300330)
        """
        self.vectors_train = vectors_train
        self.vectors_test = vectors_test
        self.labels_train = labels_train
        self.labels_test = labels_test
        self.k = k
        self.n_train = len(self.vectors_train)
        self.n_test = len(self.vectors_test)
        np.random.seed(seed)

    def compute_test_train_distances(self)-> None:
        """
        Compute and store the quantum distance
        between each pair of test/train instances.
        """
        self.test_train_distances = [[] for _ in range(self.n_test)]
        for i,vector_test in enumerate(self.vectors_test):
            for vector_train in self.vectors_train:
                self.test_train_distances[i].append(
                    qd(vector_test, vector_train).compute_quantum_distance()
                )

    def save_test_train_distances(self, filename : str, path : str)-> None:
        """
        Save the computed test_train distances as a pickle file.

        Parameters
        ----------
        filename : str
            Name of the file to save to.
        path : str
            Path to store the distances.
        """
        with open(f'{path}{filename}.pickle', 'wb') as file:
            pickle.dump(self.test_train_distances, file)

    def load_test_train_distances(self, filename : str, path : str)-> None:
        """
        Load pre-computed test_train distances list from a pickle file and 
        assign it as object attribute.

        Parameters
        ----------
        filename : str 
            Name of the file to be loaded.
        path : str
            Path of the file to be loaded.
        """    
        with open(f'{path}{filename}', 'rb') as file:
            self.test_train_distances = pickle.load(file)

    def compute_closest_vectors_indexes(self)-> None:
        """
        Compute the closest vectors to each test instance.
        """
        self.closest_vectors_indexes = []
        for i in range(self.n_test):
            self.closest_vectors_indexes.append(
                sorted(
                    range(self.n_train),
                    key = lambda j : self.test_train_distances[i][j]
                )[:self.k]
            )

    def compute_test_predictions(self)-> None:
        """
        Compute the test predictions by majority vote of the labels
        of the closest vectors to each test instance.
        In case of a tie, a random choice between the labels which are 
        tied will be assigned as prediction.
        """
        self.test_predictions = []
        for cv in self.closest_vectors_indexes:
            cv_labels = [
                self.labels_train[cv[i]] for i in range(self.k)
            ]
            c = Counter(cv_labels)
            counter = c.most_common()
            labels = [item[0] for item in counter]
            counts = [item[1] for item in counter]
            n = 1
            for i in range(1,len(counts)):
                if counts[i] == counts[i-1]:
                    n += 1
                else:
                    break
            closest_labels = labels[:n]
            self.test_predictions.append(np.random.choice(closest_labels))

