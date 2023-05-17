from QuantumDistance import QuantumDistance as qd

class QuantumKNearestNeighbors:
    """
    Class for implementing the K Nearest Neighbors algorithm 
    using the quantum distance
    """
    def __init__(self, train_data, test_data, k):
        self.train_data = train_data
        self.test_data = test_data
        self.k = k

    

    def compute_distances(self, sample):
        distances = []
        for data in(self.train_data):
            distances.append(qd(sample, data).dist)
        return distances

    def compute_minimum_distances(self, distances):
        

    