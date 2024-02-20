import os 
import sys 
import unittest 

import numpy as np 
from parameterized import parameterized_class

# The two lines below will be removed when converting the library to a package.
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/beta/")
from beta_1 import QuantumKNearestNeighbours as qkn
from utils import normalise_vector, pad_vector_with_zeros

test_args = {
    'n_runs' : 5,
    'seed' : 30031935,
    'n_vectors_train' : 25,
    'n_vectors_test' : 10,
    'vectors_limit_value' : 1000,
    'n_classes_limit' : 5,
    'vectors_size_limit' : 8,
    'k_limit' : 5
}

def set_up_class_attributes(test_args : dict)-> list:
    """
    Generate parameters for different test runs. 

    Parameters
    ----------
    test_args : dict
        Dictionary with test arguments to generate the test parameters of the 
        different runs. Its keys are the following: 
            n_runs : Number of test runs.
            seed : Random seed for generating the different parameters.
            n_vectors_train : Number of train vectors. 
            n_vectors_test : Number of test vectors.
            vectors_limit_value : Limit value that the components of the
            vectors can have.
            n_classes_limit : Limit to the number of random classes
            that can be generated.
            vectors_size_limit : Limit to the size of the generated random
            vectors.
            k_limit : Limit to the generated random number of k neighbours of 
            the algorithm. 
    
    Return
    ------
    params_list : list 
        List with the parameters to be used 
        on each run.
    """
    params_list = []
    np.random.seed(test_args['seed'])
    for _ in range(test_args['n_runs']):
        params_run = []
        vector_length = np.random.randint(1,test_args['vectors_size_limit'])
        n_classes = np.random.randint(2,test_args['n_classes_limit'])
        k = np.random.randint(1,test_args['k_limit'])
        x_train = [np.random.uniform(
            -test_args['vectors_limit_value'],
            test_args['vectors_limit_value'],
            size=vector_length
        ) for _ in range(
                test_args['n_vectors_train'])]
        x_test = [np.random.uniform(
            -test_args['vectors_limit_value'],
            test_args['vectors_limit_value'],
            size=vector_length
        ) for _ in range(
                test_args['n_vectors_test'])]
        for i,xtr in enumerate(x_train):
            xtr = normalise_vector(xtr)
            x_train[i] = pad_vector_with_zeros(xtr)
        for j,xte in enumerate(x_test):
            xte = normalise_vector(xte)
            x_test[j] = pad_vector_with_zeros(xte)
        y_train = np.random.randint(
            0, n_classes, size = test_args['n_vectors_train']
        )
        y_test = np.random.randint(
            0, n_classes, size = test_args['n_vectors_train']
        )
        beta_1_model = qkn(
            x_train, x_test, y_train, y_test, k
        )
        beta_1_model.compute_test_train_distances()
        beta_1_model.compute_closest_vectors_indexes()
        beta_1_model.compute_test_predictions()
        params_run.append(beta_1_model)
        params_run.append(test_args['n_vectors_train'])
        params_run.append(k)
        params_list.append(params_run)
    return params_list

names_parameters = ('beta_1_model', 'n_vectors_train', 'k')
@parameterized_class(
    names_parameters,
    set_up_class_attributes(test_args)
)
class TestBeta1(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the class for testing. The attributes are defined 
        with the decorator on line 93.
        """
    pass
    
    def test_number_of_test_train_distances_is_correct(self)-> None:
        """
        Check that for each train instance, its distance with all the 
        test vectors is computed, i.e., the total number of items in
        self.beta_1.test_train_distances is equal to 
        the number of train instances * number of test instances.
        """
        for dist_train in self.beta_1_model.test_train_distances:
            self.assertEqual(
                len(dist_train), self.n_vectors_train)

    def test_number_of_closest_vectors_indexes_is_correct(self)-> None:
        """
        Check that for each train instance, the length of the list 
        with the closest vectors indexes has length equal to k. 
        """
        for indexes in self.beta_1_model.closest_vectors_indexes:
            self.assertEqual(
                len(indexes), self.k
            )

    def test_train_distances_dont_change_when_saving_and_loading(self)-> None:
        """
        Check usage of save and load train_test_distances, by ensuring
        that the distances have the same value before and after saving.
        """
        distances_before_saving = self.beta_1_model.test_train_distances
        self.beta_1_model.save_test_train_distances(
            'test_train_distances', './'
        )
        self.beta_1_model.load_test_train_distances(
            'test_train_distances.pickle', './'
        )
        self.assertEqual(
            distances_before_saving, self.beta_1_model.test_train_distances
        )

    def test_predictions_are_integers(self)-> None:
        """
        Check that the predictions output by the model are integers.
        """
        for pred in self.beta_1_model.test_predictions:
            self.assertIs(type(pred), np.int64)
        

if __name__ == '__main__':
    unittest.main()