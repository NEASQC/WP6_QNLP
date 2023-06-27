### Given a normalised two-dimensional set of vectors, perform classical k means clustering into number_of_clusters clusters
import numpy as np
import pandas as pd 
import json

from sklearn.neighbors import KNeighborsClassifier



class ClassicalKNearestNeighbours():
    """
    Class for implementing the K Nearest Neighbors algorithm 
    using the classical distance
    """
    
    def __init__(
            self, dataset_dir : str, vectors_train_dir :str,
            vectors_test_dir : str, k : int
        )-> None:
        """
        Initialises the class. 

        Parameters
        ----------
        dataset_dir : str
            Directory where the dataset with the
            training  labels is stored
        vectors_train_dir : str
            Directory containing the vectors for training
        vectors_test_dir : str
            Directory containing the vectors for testing
        k : int
            Number of neighbors of the algorithm
        """
        self.labels = self.load_labels(dataset_dir)
        
        with open (vectors_train_dir) as file:
            self.vectors_train = json.load(file)
        with open(vectors_test_dir) as file:
            self.vectors_test = json.load(file)
            
        self.k = k
        
        neigh = KNeighborsClassifier(n_neighbors=self.k)
        neigh.fit(self.vectors_train, self.labels)
        
        self.predictions = neigh.predict(self.vectors_test).tolist()
        


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









