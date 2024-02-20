import os 
import sys 
import unittest 

import numpy as np 
from parameterized import parameterized_class

# The two lines below will be removed when converting the library to a package.
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/beta/")
from beta_2 import QuantumKMeans as qkm

test_args = {
    'n_runs' : 5,
    'seed' : 1906,
    'n_vectors_train' : 25,
    'vectors_limit_value' : 1000,
    'vectors_size_limit' : 8,
    'k_limit' : 5,
    'tolerance_range' : [1e-05, 1e-01],
    'itermax_limit' : 200
}

def set_up_test_parameters(test_args : dict)-> list:
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
            vectors_limit_value : Limit value that the components of the
            vectors can have.
            vectors_size_limit : Limit to the size of the generated random
            vectors.
            k_limit : Limit to the generated random number of k neighbours of 
            the algorithm. 
            tolerance_range : Range of values between the tolerance for
            stopping the algorithn can be. 
            itermax_limit : Limit to the generated random maximum number
            of iterations. 
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
        vector_length = np.random.randint(1, test_args['vectors_size_limit'])
        print(vector_length)
        k = np.random.randint(1, test_args['k_limit'])
        tol = np.random.uniform(
            test_args['tolerance_range'][0],
            test_args['tolerance_range'][1]
        )
        itermax = np.random.randint(1,test_args['itermax_limit'])
        random_cluster_initialiser = np.random.randint(0,1)
        vectors_train = [np.random.uniform(
            -test_args['vectors_limit_value'],
            test_args['vectors_limit_value'],
            size=vector_length
            ) for _ in range(
            test_args['n_vectors_train'])
        ]
        beta_2_model = qkm(
            vectors_train, k, tol, itermax
        )
        if random_cluster_initialiser == 0 :
            beta_2_model.initialise_cluster_centers(
                random_initialisation = True,
                seed = test_args['seed']
            )
        else : 
            beta_2_model.initialise_cluster_centers(
                random_initialisation = False
            )
        beta_2_model.train_k_means_algorithm()
        beta_2_model.compute_wce()
        beta_2_model.compute_predictions()
        params_run.append(beta_2_model)
        params_run.append(vectors_train)
        params_run.append(k)
        params_list.append(params_run)
    return params_list
    
names_parameters = ('beta_2_model', 'vectors_train', 'k')
@parameterized_class(names_parameters, set_up_test_parameters(test_args))
class TestBeta2(unittest.TestCase):
    @classmethod
    def setUpClass(cls)-> None:
        """
        Set up the class for testing. The attributes are defined 
        with the decorator on line 90.
        """
        pass

    def test_number_of_train_predictions_is_correct(self)-> None:
        """
        Check that the number of predictions output by the model is correct.
        """
        for i in range(self.beta_2_model.n_its):
            self.assertEqual(
                len(self.beta_2_model.predictions[i]),
                len(self.vectors_train)
            )

    def test_number_of_clusters_is_correct(self)-> None:
        """
        Check that the number of clusters output by the model is correct.
        """
        for i in range(self.beta_2_model.n_its):
            self.assertEqual(
                len(self.beta_2_model.clusters[i]),
                (self.k)
            )
    
    def test_number_of_centeres_is_correct(self)-> None:
        """
        Check that the number of centers output by the model is correct.
        """
        for i in range(self.beta_2_model.n_its):
            self.assertEqual(
                len(self.beta_2_model.centers[i]),
                self.k
            )

    def test_wce_is_float(self)-> None:
        """
        Check that the within clusters sum of errors values (WCE):
        $[error=\sum_{i=0}^{N}quantum_distance(x_{i}-center(x_{i}))$]
        are floats.
        """
        for i in range(self.beta_2_model.n_its):
            self.assertIs(
                type(self.beta_2_model.total_wce[i]),
                float
            )


if __name__ == '__main__':
    unittest.main()