from QuantumDistance import QuantumDistance as qd
import pandas as pd 
import json
from collections import Counter
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA

class QuantumKNearestNeighbours_bert:
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
        """
        self.y_test = y_test

        self.labels = y_train
        self.train_vectors = X_train
        self.test_vectors = X_test
        self.k_values = k_values
        self.predictions = []
        for i in self.test_vectors:
            distances = self.compute_distances(i)
            closest_indexes_list = self.compute_minimum_distances(distances, self.k_values)
            pred_list = self.labels_majority_vote(closest_indexes_list)
            self.predictions.append(pred_list)

            print("Prediction made for vector: ", i)
        

    @staticmethod
    def load_labels(
            train_dataset_path : str,
            test_dataset_path : str  
    ):
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

        df_train = pd.read_csv(train_dataset_path)
        df_test = pd.read_csv(test_dataset_path)


        df_train['sentence_embedding'] = np.array([np.fromstring(embedding.strip(' []'), sep=',') for embedding in df_train['sentence_embedding']]).tolist()
        df_test['sentence_embedding'] = np.array([np.fromstring(embedding.strip(' []'), sep=',') for embedding in df_test['sentence_embedding']]).tolist()

        #We reduce the dimension of the sentence embedding to a 2D vector
        ############################################################
        # Convert the "sentence_embedding" column to a 2D NumPy array
        X_train = np.array([embedding for embedding in df_train['sentence_embedding']])
        X_test = np.array([embedding for embedding in df_test['sentence_embedding']])

        # Initialize and fit the PCA model
        pca = PCA(n_components=2)  # Specify the desired number of components
        pca.fit(X_train)
        print('PCA explained variance:', pca.explained_variance_ratio_.sum())

        # Transform the data to the reduced dimension for both training and test sets
        reduced_embeddings_train = pca.transform(X_train)
        reduced_embeddings_test = pca.transform(X_test)

        # Update the DataFrames with the reduced embeddings
        df_train['sentence_embedding'] = list(reduced_embeddings_train)
        df_test['sentence_embedding'] = list(reduced_embeddings_test)

        #Preprocess labels
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(df_train['class'])

        df_train['class'] = label_encoder.transform(df_train['class'])
        df_test['class'] = label_encoder.transform(df_test['class'])

        X_train, y_train, X_test, y_test = reduced_embeddings_train, df_train['class'], reduced_embeddings_test, df_test['class']
        
        return X_train, X_test, y_train.values, y_test.values

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
    def compute_minimum_distances(distances, k_values):
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
        closest_indexes : list[int]
            List with the indexes of the k closest vectors
        
        Returns
        -------
        label : int
            Most frequent label among the k closest vectors
        """
        label_list = []

        for closest_indexes in closest_indexes_list:
            closest_labels = []
            for i in closest_indexes:
                closest_labels.append(self.labels[i])
            c = Counter(closest_labels)
            label, count = c.most_common()[0]

            label_list.append(label)

        return label_list





    