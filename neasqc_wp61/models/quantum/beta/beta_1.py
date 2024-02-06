"""
QuantumKNearestNeighbours
=========================
Module containing the base class for the quantum k-nearest neighbours algorithm
"""

import time
import random as rd 
import pickle
from collections import Counter

import numpy as np

from quantum_distance import QuantumDistance as qd

class QuantumKNearestNeighbours:
    """
    Class for implementing the k-nearest neighbors algorithm 
    using the quantum distance
    """
    
    def __init__(
        self, x_train : list[np.array], x_test : list[np.array],
        y_train : list[int], y_test : list[int], k : int
    )-> None:
        """
        Initialiser of the class

        Parameters
        ----------
        x_train : list[np.array]
            Train vectors
        x_test : list[np.array]
            Testing vectors
        y_train : list[int]
            Train labels
        y_test = list[int]
            Test labels
        k : int
            k closest vectors to make predictions with
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.k = k
        self.n_train = len(self.x_train)
        self.n_test = len(self.x_test)

    def compute_train_test_distances(self) -> None:
        """
        For each training instance, computes and stores the quantum distance
        between it and all the test instances
        """
        self.train_test_distances = [[] for _ in range(self.n_train)]
        for i,xtr in enumerate(self.x_train):
            for xte in self.x_test:
                self.train_test_distances[i].append(
                    qd(xtr, xte).compute_quantum_distance()
                )

    def save_train_train_test_distances(self, path : str, name : str) -> None:
        """
        Saves the computed train_test distances as pickle file

        Parameters
        ----------
        path : Path 
        """

    
    def compute_closest_vectors_indexes(self) -> None:
        """
        For each training instance, and provided that distances
        have been computed, stores the indexes
        of the closest k neighbours. 
        """
        self.closest_vectors_indexes = []
        for i in range(self.n_train):
            self.closest_vectors_indexes.append(sorted(
                range(self.n_test), key = lambda j : self.train_test_distances[i][j])
            [:self.k])




        
    def compute_predictions(
        self, compute_checkpoints : bool = False,
        ) -> None:
        """
        Makes the predictions of the model

        Parameters
        ----------
        compute_checkpoints : bool
            Decides whether to store checkpoints or not 
        """
        self.predictions = []
        for index,i in enumerate(self.test_vectors):
            t1 = time.time()
            distances = self.compute_distances(i)
            closest_indexes_list = self.compute_minimum_distances(distances, self.k_values)
            pred_list = self.labels_majority_vote(closest_indexes_list)
            self.predictions.append(pred_list)
            
            if index % 25 == 0 and compute_checkpoints == True:
                with open(
                   current_path +  '/../../../benchmarking/results/raw/temporary_predictions_beta.pickle', 'wb'
                ) as file:
                    pickle.dump(self.predictions, file)
            t2 = time.time()
            print('Time to do the iteration : ', t2 - t1)
    
    def get_predictions(self):
        """
        Getter for the predictions
        """
        return self.predictions





    @staticmethod
    def compute_minimum_distances(distances, k_values) -> list:
        """
        Computes the indexes of the k closest elements of the training 
        dataset

        Parameters
        ----------
        distances : list[float]
            List with the distances to all elements of the training dataset
        k_values : lsit
            Number of selected k neighbors

        Returns 
        -------
        closest_indexes : list[int]
            indexes of the k closest vectors
        """
        closest_indexes_list = []
        closest_indexes = sorted(
                range(len(distances)), key=lambda j: distances[j]
                )

        for k in k_values:
            closest_indexes_list.append(closest_indexes[:k])

        return closest_indexes_list
    
    def labels_majority_vote(self, closest_indexes_list):
        """
        Gets the labels of the closest vectors and returns the most 
        frequent one among them 

        Parameters 
        ----------
        closest_indexes_list : list[int]
            List with the indexes of the k closest vectors
        
        Returns
        -------
        label : list
            Most frequent label among the k closest vectors
        """
        label_list = []

        for closest_indexes in closest_indexes_list:
            closest_labels = []
            for i in closest_indexes:
                closest_labels.append(self.labels[i])
            c = Counter(closest_labels)
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
            label = rd.choice(closest_labels)



            label_list.append(label)

        return label_list

