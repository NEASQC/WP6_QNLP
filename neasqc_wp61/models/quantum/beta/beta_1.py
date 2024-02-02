from quantum_distance import QuantumDistance as qd
import pickle
from collections import Counter
import numpy as np
import time
import os
import random as rd 
current_path = os.path.dirname(os.path.abspath(__file__))


class QuantumKNearestNeighbours:
    """
    Class for implementing the K Nearest Neighbors algorithm 
    using the quantum distance
    """
    
    def __init__(
        self, X_train, X_test, y_train, y_test, k_values : list[int]
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
        checkpoint_output : str
            Path where to store the checkpoints with predictions
        """
        self.y_test = y_test

        self.labels = y_train
        self.train_vectors = X_train
        self.test_vectors = X_test
        self.k_values = k_values
        
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
            distances.append(qd(sample, vector).compute_quantum_distance())
        return distances

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

