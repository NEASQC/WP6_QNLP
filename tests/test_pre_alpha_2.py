import os
import random 
import sys 
import unittest

import lambeq
import numpy as np
import torch
from parameterized import parameterized_class

# The two lines below will be removed when converting the library to a package.
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/pre_alpha_2/")
from pre_alpha_2 import PreAlpha2 as pre_alpha_2
from utils import get_labels_one_hot_encoding, load_dataset

test_args = {
    'train_dataset_path' : "./../neasqc_wp61/data/datasets/toy_datasets/multiclass_toy_train.tsv",
    'val_dataset_path' : "./../neasqc_wp61/data/datasets/toy_datasets/multiclass_toy_validation.tsv",
    'test_dataset_path' : "./../neasqc_wp61/data/datasets/toy_datasets/multiclass_toy_test.tsv",
    'seed' : 1924, 
    'n_qubits_noun_range' : [1,2],
    'n_qubits_sentence_range' : [3,4],
    'n_layers_range' : [1,2],
    'n_single_qubit_params_range' : [1,2],
    'epochs_range' : [1,10],
    'lr_range' : [1e-05, 1e-01],
    'batch_size_range' : [1,4]
}

def set_up_test_parameters(test_args : dict)-> list:
    """
    Generate parameters for different test runs. 

    Parameters
    ----------
    test_args : dict
        Dictionary with test arguments to generate the test parameters of the 
        different runs. Its keys are the following: 
           train_dataset_path : Path of the train dataset used for testing.
           val_dataset_path : Path of the val dataset used for testing.
           test_dataset_path : Path of the test dataset used for testing.
           seed : Random seed for generating the different parameters. 
           n_qubits_noun_range : Range of values of the number of qubits per
        noun type.
            n_qubits_sentence_range : Range of values of the number of qubits
        per sentence type.
            n_layers_range : Range of values of the number of layers of
        the circuits.
            n_single_qubit_params_range : Range of values of the number
        of rotational parameters per single qubit.
            epochs_range : Range of values of the number of epochs in training.
            lr_range : Range of values of the learning rate of the 
        optimiser.
            batch_size_range : Range of values of the batch size in training.
        Its lower value has to be less than the number of sentences of the
        validation dataset.
    Return
    ------
    params_list : list 
        List with the parameters to be used 
        on each run.
    """
    params_list = []
    np.random.seed(test_args['seed'])
    sentences_train, labels_train = load_dataset(
        test_args['train_dataset_path']
    )
    sentences_val, labels_val = load_dataset(
        test_args['val_dataset_path']
    )
    sentences_test, labels_test = load_dataset(
        test_args['test_dataset_path']
    )
    labels_one_hot_encoding = get_labels_one_hot_encoding(
        labels_train, labels_val, labels_test
    )[0]
    n_labels = get_labels_one_hot_encoding(
        labels_train, labels_val, labels_test
    )[1]
    sentences = [sentences_train, sentences_val, sentences_test]
    optimisers_list = [
        torch.optim.Adadelta, torch.optim.Adagrad, torch.optim.Adam,
        torch.optim.Adamax, torch.optim.AdamW, torch.optim.ASGD,
        torch.optim.NAdam, torch.optim.RAdam, torch.optim.RMSprop,
        torch.optim.Rprop, torch.optim.SGD
    ]
    ansatze_list = [
        lambeq.IQPAnsatz, lambeq.Sim14Ansatz, 
        lambeq.Sim15Ansatz, lambeq.StronglyEntanglingAnsatz
    ]
    for optimiser in optimisers_list:
        for ansatz in ansatze_list:
            print('optimiser = ', optimiser)
            print('anstaz = ', ansatz)
            params_run = []
            n_qubits_noun = np.random.randint(
                test_args['n_qubits_noun_range'][0],
                test_args['n_qubits_noun_range'][1]
            )
            n_qubits_sentence = np.random.randint(
                test_args['n_qubits_sentence_range'][0],
                test_args['n_qubits_sentence_range'][1]
            )
            n_layers = np.random.randint(
                test_args['n_layers_range'][0],
                test_args['n_layers_range'][1]
            )
            n_single_qubit_params = np.random.randint(
                test_args['n_single_qubit_params_range'][0],
                test_args['n_single_qubit_params_range'][1]
            )
            epochs = np.random.randint(
                test_args['epochs_range'][0],
                test_args['epochs_range'][1]
            )
            lr = np.random.uniform(
                test_args['lr_range'][0],
                test_args['lr_range'][1]
            )
            batch_size = np.random.randint(
                test_args['batch_size_range'][0],
                test_args['batch_size_range'][1]
            )
            seed = np.random.randint(
                0,1e10
            )
            pre_alpha_2_model = pre_alpha_2(
                sentences, labels_one_hot_encoding, n_labels,
                ansatz, n_qubits_noun, n_qubits_sentence,
                n_layers, n_single_qubit_params, optimiser,
                epochs, batch_size,
                optimiser_args = {'lr' : lr}, seed = seed
            )
            pre_alpha_2_model.fit()
            pre_alpha_2_model.compute_probabilities()
            pre_alpha_2_model.compute_predictions()
            preds = [
                pre_alpha_2_model.preds_train,
                pre_alpha_2_model.preds_val
            ]
            probs = [
                pre_alpha_2_model.probs_train,
                pre_alpha_2_model.probs_val
            ]
            params_run.append(pre_alpha_2_model)
            params_run.extend([
                sentences, labels_one_hot_encoding, n_labels,
                ansatz, n_qubits_noun, n_qubits_sentence,
                n_layers, n_single_qubit_params, optimiser,
                epochs, batch_size, lr, seed, preds, probs
            ])
            params_list.append(params_run)
    return params_list

names_parameters = (
    'pre_alpha_2_model', 'sentences', 'labels_one_hot_encoding',
    'n_labels', 'ansatz', 'n_qubits_noun', 'n_qubits_sentence',
    'n_layers', 'n_single_qubit_params', 'optimiser',
    'epochs', 'batch_size', 'lr', 'seed', 'preds', 'probs'
)

@parameterized_class(names_parameters, set_up_test_parameters(test_args))
class TestPreAlpha2(unittest.TestCase):
    
    def test_model_raises_error_if_not_enough_sentence_qubits(self)-> None:
        """
        Test that the model raises an error when thre are not enough sentence qubits.
        """
        with self.assertRaises(ValueError):
            pre_alpha_2_ = pre_alpha_2(
            self.sentences, self.labels_one_hot_encoding, self.n_labels, 
            self.ansatz, self.n_qubits_noun, 0,
            self.n_layers, self.n_single_qubit_params, self.optimiser,
            self.epochs, self.batch_size,
            optimiser_args = {'lr' : self.lr}, seed = self.seed
            )

    def test_cross_entropy_loss_returns_positive_values(self)->None:
        """
        Test that the cross entropy loss return positive values.
        """
        cross_entropy = self.pre_alpha_2_model.cross_entropy_loss_wrapper()
        for _ in range (1024):
            y = torch.zeros(self.batch_size,self.n_labels)
            random_index = random.randint(0, self.n_labels - 1)
            y[:,random_index] = 1
            size_y_hat = [2] * self.n_qubits_sentence
            y_hat = torch.rand(self.batch_size,*size_y_hat)
            self.assertGreater(cross_entropy(y_hat, y)[0], 0)
    
    def test_number_of_preds_and_probs_is_correct(self):
        """
        Test that the number of predictions and probabilities is equal
        to the the number of epochs in traininig.
        """
        for probs,preds in zip(self.probs, self.preds):
            with self.subTest(probs=probs):
                self.assertEqual(self.epochs, len(probs))
            with self.subTest(pred=preds):
                self.assertEqual(self.epochs, len(preds))
    
    def test_preds_are_integers(self):
        """
        Test that the values predicted are integers.
        """
        preds_train_list = self.preds[0]
        preds_val_list = self.preds[1]
        for preds_single_iteration_train, preds_single_iteration_val in zip(
            preds_train_list, preds_val_list
        ):
            for pred_value_train, pred_value_val in zip(
                preds_single_iteration_train, preds_single_iteration_val
            ):
                with self.subTest(
                    pred_value_train = pred_value_train,
                    pred_value_val = pred_value_val
                ):
                    self.assertIs(type(pred_value_train), int)
                    self.assertIs(type(pred_value_val), int)
    
    def test_probs_add_up_to_1(self):
        """
        Test that the probabilities predicted for each instance add up to one.
        """
        probs_train_list = self.probs[0]
        probs_val_list = self.probs[1]
        for probs_single_iteration_train, probs_single_iteration_val in zip(
            probs_train_list, probs_val_list
        ):
            for prob_value_train, prob_value_val in zip(
                probs_single_iteration_train, probs_single_iteration_val
            ):
                with self.subTest(
                    pred_value_train = prob_value_train,
                    pred_value_val = prob_value_val
                ):
                        self.assertLessEqual(
                            abs(sum(prob_value_train) -1), 1e-06
                        )
                        self.assertLessEqual(
                            abs(sum(prob_value_val) -1), 1e-06
                        )

    
if __name__ == '__main__':
    unittest.main()