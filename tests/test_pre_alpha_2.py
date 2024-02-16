import argparse 
import os
import random 
import sys 
import unittest

import torch

# The two lines below will be removed when converting the library to a package.
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/pre_alpha_2/")
from pre_alpha_2 import PreAlpha2 as pre_alpha_2
from utils import get_labels_one_hot_encoding, load_dataset

class TestPreAlpha2(unittest.TestCase):
    @classmethod
    def setUpClass(cls)-> None:
        """
        Set up the class for testing.
        """
        sentences_train, labels_train = load_dataset(args.train_dataset_path)
        sentences_val, labels_val = load_dataset(args.val_dataset_path)
        sentences_test, labels_test = load_dataset(args.test_dataset_path)
        cls.labels_one_hot_encoding = get_labels_one_hot_encoding(
            labels_train, labels_val, labels_test
        )[0]
        cls.n_labels = get_labels_one_hot_encoding(
            labels_train, labels_val, labels_test
        )[1]
        cls.sentences = [sentences_train, sentences_val, sentences_test]
        cls.pre_alpha_2 = pre_alpha_2(
            cls.sentences, cls.labels_one_hot_encoding, cls.n_labels, 
            args.ansatz, args.n_qubits_noun, args.n_qubits_sentence,
            args.n_layers, args.n_single_qubit_parameters,args.optimiser,
            args.epochs, args.batch_size,
            optimiser_args = {'lr' : args.learning_rate}, seed = args.seed
        )
        cls.pre_alpha_2.fit()
        cls.pre_alpha_2.compute_probabilities()
        cls.pre_alpha_2.compute_predictions()
        cls.probabilities = [
            cls.pre_alpha_2.probs_train,
            cls.pre_alpha_2.probs_val,
        ]
        cls.predictions = [
            cls.pre_alpha_2.preds_train,
            cls.pre_alpha_2.preds_val,
        ]
        torch.manual_seed(args.seed)

    def test_model_raises_error_if_not_enough_sentence_qubits(self)-> None:
        """
        Test that the model raises an error when thre are not enough sentence qubits.
        """
        with self.assertRaises(ValueError):
            pre_alpha_2_ = pre_alpha_2(
            self.sentences, self.labels_one_hot_encoding, self.n_labels, 
            args.ansatz, args.n_qubits_noun, 0,
            args.n_layers, args.n_single_qubit_parameters,args.optimiser,
            args.epochs, args.batch_size,
            optimiser_args = {'lr' : args.learning_rate}, seed = args.seed
            )

    def test_cross_entropy_loss_returns_positive_values(self)->None:
        """
        Test that the cross entropy loss return positive values.
        """
        cross_entropy = self.pre_alpha_2.cross_entropy_loss_wrapper()
        for _ in range (1024):
            y = torch.zeros(args.batch_size,self.n_labels)
            random_index = random.randint(0, self.n_labels - 1)
            y[0,random_index] = 1
            size_y_hat = [2] * args.n_qubits_sentence
            y_hat = torch.rand(args.batch_size,*size_y_hat)
            sum_dims = y_hat.view(1, -1).sum(dim=1, keepdim=True)
            y_hat_normalised = y_hat / sum_dims
            self.assertGreater(cross_entropy(y_hat_normalised, y)[0], 0)
    
    def test_number_of_preds_and_probs_is_correct(self):
        """
        Test that the number of predictions and probabilities is equal
        to the the number of epochs in traininig.
        """
        for probs,preds in zip(self.probabilities, self.predictions):
            with self.subTest(probs=probs):
                self.assertEqual(self.pre_alpha_2.epochs, len(probs))
            with self.subTest(pred=preds):
                self.assertEqual(self.pre_alpha_2.epochs, len(preds))
    
    def test_preds_are_integers(self):
        """
        Test that the values predicted are integers.
        """
        preds_train_list = self.predictions[0]
        preds_val_list = self.predictions[1]
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
        probs_train_list = self.probabilities[0]
        probs_val_list = self.probabilities[1]
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tr", "--train_dataset_path", type = str,
        help = "Path of the train dataset.",
        default = "./../neasqc_wp61/data/datasets/toy_datasets/multiclass_toy_train.tsv"
    )
    parser.add_argument(
        "-val", "--val_dataset_path", type = str,
        help = "Path of the validation dataset.",
        default = "./../neasqc_wp61/data/datasets/toy_datasets/multiclass_toy_validation.tsv"
    )
    parser.add_argument(
        "-te", "--test_dataset_path", type = str,
        help = "Path of the test dataset.",
        default = "./../neasqc_wp61/data/datasets/toy_datasets/multiclass_toy_test.tsv"
    )
    parser.add_argument(
        "-an", "--ansatz", type = str,
        help = "Ansatz to use.",
        default = "IQP"
    )
    parser.add_argument(
        "-qn", "--n_qubits_noun", type = int,
        help = "Number of qubits to use per noun type.",
        default = 1
    )
    parser.add_argument(
        "-qs", "--n_qubits_sentence", type = int,
        help = "Number of qubits to use per sentence type.",
        default = 3
    )
    parser.add_argument(
        "-nl", "--n_layers", type = int,
        help = "Number of layers in the circuit.",
        default = 1
    )
    parser.add_argument(
        "-np", "--n_single_qubit_parameters", type = int,
        help = "Number of single qubit parameteres.",
        default = 1
    )
    parser.add_argument(
        "-op", "--optimiser", type = str,
        help = "Optimiser to use.",
        default = "Adam"
    )
    parser.add_argument(
        "-ep", "--epochs", type = int,
        help = "Number of training epochs to perform.",
        default = 10
    )
    parser.add_argument(
        "-bs", "--batch_size", type = int,
        help = "Batch size to use.",
        default = 2
    )
    parser.add_argument(
        "-lr", "--learning_rate", type = float,
        help = "Learning rate to use in optimisation.",
        default = 0.001
    )
    parser.add_argument(
        "-s", "--seed", type = int,
        help = "Random initial seed.",
        default = 3003
    )
    args, remaining = parser.parse_known_args()
    remaining.insert(0, sys.argv[0])
    unittest.main(argv=remaining)