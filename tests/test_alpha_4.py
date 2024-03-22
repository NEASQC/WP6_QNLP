import os
import random
import sys
import unittest

import numpy as np
import pennylane as qml
import torch
from parameterized import parameterized_class


# The two lines below will be removed when converting the library to a package.
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/")
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/alpha_3_4/")
import circuit as circ
from alpha_3_4 import Alpha4 as alpha_4
from utils import get_labels_one_hot_encoding

test_args = {
    "seed": 888,
    "n_vectors_train": 15,
    "n_vectors_val": 5,
    "n_vectors_test": 5,
    "n_classes_limit": 4,
    "n_layers_circuit_limit": 3,
    "vectors_limit_value": 1000,
    "vectors_size_limit": 8,
    "epochs_limit": 10,
    "lr_range": [1e-05, 1e-01],
    "batch_size_limit": 4,
}


def set_up_test_parameters(test_args: dict) -> list:
    """
    Generate parameters for different test runs.
    """
    params_list = []
    np.random.seed(test_args["seed"])
    torch.manual_seed(test_args["seed"])
    optimisers_list = [
        torch.optim.Adadelta,
        torch.optim.Adagrad,
        torch.optim.Adam,
        torch.optim.Adamax,
        torch.optim.AdamW,
        torch.optim.ASGD,
        torch.optim.NAdam,
        torch.optim.RAdam,
        torch.optim.RMSprop,
        torch.optim.Rprop,
        torch.optim.SGD,
    ]
    ansatze_list = [circ.Sim14, circ.Sim15, circ.StronglyEntangling]
    observables_list = [qml.PauliX, qml.PauliY, qml.PauliZ, qml.Hadamard]
    axis_embedding_list = ["X", "Y", "Z"]
    initialisations_list = [torch.nn.init.uniform_, torch.nn.init.normal_]
    n_classes = np.random.randint(2, test_args["n_classes_limit"])
    for optimiser in optimisers_list:
        for ansatz in ansatze_list:
            for init in initialisations_list:
                params_run = []
                vector_length = np.random.randint(
                    3, test_args["vectors_size_limit"]
                )
                n_layers = np.random.randint(
                    1, test_args["n_layers_circuit_limit"]
                )
                vectors = []
                labels = []
                for key in (
                    "n_vectors_train",
                    "n_vectors_val",
                    "n_vectors_test",
                ):
                    vectors.append(
                        [
                            np.random.uniform(
                                -test_args["vectors_limit_value"],
                                test_args["vectors_limit_value"],
                                size=vector_length,
                            )
                            for _ in range(test_args[key])
                        ]
                    )
                    labels.append(
                        np.random.randint(
                            0, n_classes, size=test_args[key]
                        ).tolist()
                    )
                labels_one_hot_encoding = get_labels_one_hot_encoding(
                    labels[0], labels[1], labels[2]
                )[0]
                n_classes = get_labels_one_hot_encoding(
                    labels[0], labels[1], labels[2]
                )[1]
                n_qubits_circuit = vector_length
                obsevables_circuit_keys = [i for i in range(n_qubits_circuit)]
                observables = {
                    key: random.choice(observables_list)
                    for key in obsevables_circuit_keys
                }
                axis_embedding = random.choice(axis_embedding_list)
                circuit = ansatz(
                    n_qubits_circuit,
                    n_layers,
                    axis_embedding,
                    observables,
                    output_probabilities=True,
                )
                epochs = np.random.randint(1, test_args["epochs_limit"])
                lr = np.random.uniform(
                    test_args["lr_range"][0], test_args["lr_range"][1]
                )
                batch_size = np.random.randint(
                    1, test_args["batch_size_limit"]
                )
                alpha_4_model = alpha_4(
                    vectors,
                    labels_one_hot_encoding,
                    n_classes,
                    circuit,
                    optimiser,
                    epochs,
                    batch_size,
                    torch.nn.CrossEntropyLoss,
                    {"lr": lr},
                    "cpu",
                    test_args["seed"],
                    init,
                )
                alpha_4_model.train()
                alpha_4_model.compute_preds_probs_test()
                loss_train = alpha_4_model.loss_train
                loss_val = alpha_4_model.loss_val
                preds_train = alpha_4_model.preds_train
                preds_val = alpha_4_model.preds_val
                preds_test = alpha_4_model.preds_test
                probs_train = alpha_4_model.probs_train
                probs_val = alpha_4_model.probs_val
                probs_test = alpha_4_model.probs_test
                params_run.append(alpha_4_model)
                params_run.extend(
                    [
                        vectors,
                        labels_one_hot_encoding,
                        n_classes,
                        circuit,
                        optimiser,
                        epochs,
                        batch_size,
                        lr,
                        test_args["seed"],
                        init,
                        vector_length,
                        ansatz,
                        n_layers,
                        axis_embedding,
                        observables,
                        loss_train,
                        loss_val,
                        preds_train,
                        preds_val,
                        preds_test,
                        probs_train,
                        probs_val,
                        probs_test,
                    ]
                )
                params_list.append(params_run)
    return params_list


names_parameters = (
    "alpha_4_model",
    "vectors",
    "labels_one_hot_encoding",
    "n_classes",
    "circuit",
    "optimiser",
    "epochs",
    "batch_size",
    "lr",
    "seed",
    "init_qubits",
    "vector_length",
    "ansatz",
    "n_layers",
    "axis_embedding",
    "observables",
    "loss_train",
    "loss_val",
    "preds_train",
    "preds_val",
    "preds_test",
    "probs_train",
    "probs_val",
    "probs_test",
)


@parameterized_class(names_parameters, set_up_test_parameters(test_args))
class TestAlpha4(unittest.TestCase):
    def test_model_raises_errors_if_number_of_qubits_is_not_correct(
        self,
    ) -> None:
        """
        Test that the model raises an error if the number of qubits is
        different from the length of circuit inputs.
        """
        observables = self.observables.copy()
        del observables[self.circuit.n_qubits - 1]
        circuit = self.ansatz(
            self.vector_length - 1,
            self.n_layers,
            self.axis_embedding,
            observables,
            output_probabilities=True,
        )
        with self.assertRaises(ValueError):
            alpha_4_model = alpha_4(
                self.vectors,
                self.labels_one_hot_encoding,
                self.n_classes,
                circuit,
                self.optimiser,
                self.epochs,
                self.batch_size,
                torch.nn.CrossEntropyLoss,
                {"lr": self.lr},
                "cpu",
                self.seed,
                self.init_qubits,
            )

    def test_model_raises_error_if_out_probabilities_false_in_circuit(
        self,
    ) -> None:
        """
        Test that the model raises an error if output_probabilities = False
        in the circuit.
        """
        circuit = self.ansatz(
            self.vector_length,
            self.n_layers,
            self.axis_embedding,
            self.observables,
            output_probabilities=False,
        )
        with self.assertRaises(ValueError):
            alpha_4_model = alpha_4(
                self.vectors,
                self.labels_one_hot_encoding,
                self.n_classes,
                circuit,
                self.optimiser,
                self.epochs,
                self.batch_size,
                torch.nn.CrossEntropyLoss,
                {"lr": self.lr},
                "cpu",
                self.seed,
                self.init_qubits,
            )

    def test_model_raises_error_if_not_enough_observables(self) -> None:
        """
        Test that the model raises an error if log2(n_measured_qubits)
        is less than the number of classes in our dataset.
        """
        observables = {0: qml.PauliZ}
        circuit = self.ansatz(
            self.vector_length,
            self.n_layers,
            self.axis_embedding,
            observables,
            output_probabilities=False,
        )
        with self.assertRaises(ValueError):
            alpha_4_model = alpha_4(
                self.vectors,
                self.labels_one_hot_encoding,
                self.n_classes,
                circuit,
                self.optimiser,
                self.epochs,
                self.batch_size,
                torch.nn.CrossEntropyLoss,
                {"lr": self.lr},
                "cpu",
                self.seed,
                self.init_qubits,
            )

    def test_loss_values_are_floats(self) -> None:
        """
        Test that the loss values have float type.
        """
        for loss_train_instance, loss_val_instance in zip(
            self.loss_train, self.loss_val
        ):
            self.assertIs(type(loss_train_instance), float)
            self.assertIs(type(loss_val_instance), float)

    def test_number_of_preds_is_correct(self) -> None:
        """
        Test that the number of predictions is correct,
        i.e. its lengths for train and val are equal to the
        number of epochs, and within each epoch, the length
        is equal to the length of train/val vectors.
        """
        for i, preds in enumerate([self.preds_train, self.preds_val]):
            self.assertEqual(len(preds), self.epochs)
            for j in range(len(preds)):
                self.assertEqual(len(preds[j]), len(self.vectors[i]))
        self.assertEqual(len(self.preds_test), len(self.vectors[2]))

    def test_number_of_probs_is_correct(self) -> None:
        """
        Test that the number of probabilities is correct,
        i.e. its lengths for train and val are equal to the
        number of epochs, and within each epoch, the length
        is equal to the length of train/val vectors.
        """
        for i, probs in enumerate([self.probs_train, self.probs_val]):
            self.assertEqual(len(probs), self.epochs)
            for j in range(len(probs)):
                self.assertEqual(len(probs[j]), len(self.vectors[i]))
        self.assertEqual(len(self.probs_test), len(self.vectors[2]))

    def test_preds_are_integers(self) -> None:
        """
        Test that the predicitons are integers.
        """
        for i, preds in enumerate([self.preds_train, self.preds_val]):
            for j in range(len(preds)):
                for prediction in preds[j]:
                    self.assertIs(type(prediction), int)
        for prediction in self.preds_test:
            self.assertIs(type(prediction), int)

    def test_probs_add_up_to_one(self) -> None:
        """
        Test that the predicitons are integers.
        """
        for i, probs in enumerate([self.probs_train, self.probs_val]):
            for j in range(len(probs)):
                for probabilities in probs[j]:
                    self.assertLessEqual(abs(sum(probabilities) - 1), 1e-06)
        for probs in self.probs_test:
            self.assertLessEqual(abs(sum(probabilities) - 1), 1e-06)


if __name__ == "__main__":
    unittest.main()
