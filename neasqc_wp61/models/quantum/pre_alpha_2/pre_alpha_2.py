"""
PreAlpha2
=========
Module containing the class for PreAlpha2 model, 
which classifies sentences using lambeq sofware.
"""
import pickle 
from typing import Callable

import lambeq
import torch 


class PreAlpha2:
    """
    A class to implement sentence classification 
    using lambeq library.
    """
    def __init__(
        self, sentences : list[list[str]],
        labels : list[list[int]],
        n_classes : int,
        ansatz : lambeq.ansatz.CircuitAnsatz,
        n_qubits_noun : int,
        n_qubits_sentence : int,
        n_layers : int,
        n_single_qubits_params : int,
        optimiser : torch.optim.Optimizer,
        epochs : int,
        batch_size : int,
        loss_function : Callable = None, 
        optimiser_args : dict = {'lr' : 0.001},
        device : int = -1,
        seed : int = 30031935
    )-> None:
        """
        Initialise the pre_alpha_2 class.

        Parameters
        ----------
        sentences : list[list[str]]
            List with the train, validation and test sentences.
        labels : list[list[int]]
            List with the train, validation and test labels.
        n_classes : int
            Total number of classes in our datasets.
        n_qubits_noun : int
            Number of qubits per noun type.
        n_qubits_sentence : int
            Number of qubits per sentence type.
        n_layers : int
            Number of layers of the circuits.
        n_single_qubit_params : int
            Number of rotations per single qubit.
        optimiser : torch.optim.Optimizer
            Optimiser to use for training the model.
        epochs : int
            Number of epochs to be done in training. 
        batch_size : int
            Batch size to use in training.
        loss_function : Callable
            Loss function to use in training. If None,
            a class default method (cross_entropy_loss_wrapper)
            defined in the class will be used. If other function
            is chosen, a wrapper similar to the one 
            in cross_entropy_loss_wrapper should be applied 
            to the cost function.
        optimiser_args : dict
            Optional arguments for the optmiser. 
        device : int
            CUDA device ID used for tensor operation speed-up.
            -1 will use the CPU.
        seed : int
            Random seed for generating the initial parameters. 
        """    
        if n_classes > 2 ** n_qubits_sentence:
            raise ValueError(
                "Base 2 logarithm of the number of sentence qubits"\
                "must be greater than the number of classes.")
        self.sentences_train = sentences[0]
        self.sentences_val = sentences[1]
        self.sentences_test = sentences[2]
        self.labels_train = labels[0]
        self.labels_val = labels[1]
        self.labels_test = labels[2]
        self.n_classes = n_classes
        self.n_qubits_sentence = n_qubits_sentence
        self.ansatz = ansatz(
            {
                lambeq.AtomicType.NOUN : n_qubits_noun,
                lambeq.AtomicType.SENTENCE : n_qubits_sentence
            },
            n_layers = n_layers,
            n_single_qubit_params = n_single_qubits_params
        )
        self.optimiser = optimiser
        self.epochs = epochs
        self.batch_size = batch_size
        if loss_function == None:
            self.loss_function = self.cross_entropy_loss_wrapper()
        else:
            self.loss_function = loss_function
        self.optimiser_args = optimiser_args
        self.device = device
        self.seed = seed

    def cross_entropy_loss_wrapper(self)-> None:
        """
        Wrapper for computing the cross entropy loss.
        """
        def cross_entropy_loss(y_hat,y):
            y_hat = self.reshape_output_lambeq_model(y_hat)
            entropies = y * torch.log(y_hat)
            loss = -torch.sum(entropies)/len(y)
            return loss, y_hat
        return cross_entropy_loss
    
    def reshape_output_lambeq_model(
        self, probs : torch.tensor, epsilon : float = 1e-13
    )-> None:
        """
        Reshape the probabilities ouput from a lambeq circuit
        so that it has the correct shape to be input in the 
        cross_entropy function. It also clips the vector so that
        no overflow is found in the logarithms. 

        Parameters
        ----------
        probs : torch.tensor
            Tensor with probabilities output from a lambeq circuit.
        epsilon : float
            Value to clip x with.
        """
        probs = probs.view(probs.shape[0], 2 ** self.n_qubits_sentence)
        probs = probs[:,:self.n_classes]
        probs = torch.nn.functional.normalize(probs, p=1)
        probs = torch.clip(probs, epsilon, 1 - epsilon)
        return probs
            
    def create_diagrams(
        self, kwargs_parser : dict = {}, kwargs_diagrams : dict = {}
    )-> None:
        """
        Create train, validation and test sentence diagrams. 

        Parameters
        ----------
        kwargs_parser : dict
            Dictionary with keyword arguments for a lambeq.BobcatParser object.
        kwargs_diagrams : dict
            Dictionary with keyword arguments for the method converting the
            sentences to diagrams. 
        """
        parser = lambeq.BobcatParser(**kwargs_parser)
        self.diagrams_train = parser.sentences2diagrams(
            self.sentences_train, **kwargs_diagrams
        )
        self.diagrams_val = parser.sentences2diagrams(
            self.sentences_val, **kwargs_diagrams
        )
        self.diagrams_test = parser.sentences2diagrams(
            self.sentences_test, **kwargs_diagrams
        )

    def save_diagrams(self, filename : str, filepath : str)-> None:
        """
        Save the diagrams in a given path. It must be called after having
        created the diagrams. 

        Parameters
        ----------
        filename : str
            Name of the file where to save the diagrams. 
        filepath : str
            Path where to save the diagrams to. 
        """
        diagrams = [
            self.diagrams_train, self.diagrams_val, self.diagrams_test
        ]
        for i, dataset in enumerate(('train', 'val', 'test')):
            with open(
                filepath + filename + f'_{dataset}.pickle', 'wb') as file:
                pickle.dump(diagrams[i], file)

    def load_diagrams(self, filename : str, filepath : str)-> None:
        """
        Load the diagrams from a given path and assigns them as instance
        attributes.

        Parameters
        ----------
        filename : str
            Name of the file where to save the diagrams. 
        filepath : str
            Path where to save the diagrams to. 
        """
        diagrams = []
        for dataset in ('train', 'val', 'test'):
            with open(
                filepath + filename + f'_{dataset}.pickle', 'rb') as file:
                dg = pickle.load(file)
                diagrams.append(dg)
        self.diagrams_train = diagrams[0]
        self.diagrams_val = diagrams[1]
        self.diagrams_test = diagrams[2]

    def create_circuits(self)-> None:
        """
        Create train, validation and test lambeq circuits.
        """
        self.circuits_train = [
            self.ansatz(diagram) for diagram in self.diagrams_train
        ]
        self.circuits_val = [
            self.ansatz(diagram) for diagram in self.diagrams_val
        ]
        self.circuits_test = [
            self.ansatz(diagram) for diagram in self.diagrams_test
        ]
    
    def create_dataset(self)-> None:
        """
        Create train, validation and test lambeq datasets.
        """
        self.dataset_train = lambeq.Dataset(
            self.circuits_train, self.labels_train,
            self.batch_size, shuffle = False
        )
        self.dataset_val = lambeq.Dataset(
            self.circuits_val, self.labels_val,
            self.batch_size, shuffle = False
        )
        self.dataset_test = lambeq.Dataset(
            self.circuits_test, self.labels_test,
            self.batch_size, shuffle = False
        )

    def create_model(self, backend_config : dict = None)-> None:
        """
        Create lambeq pennylane model. 

        Parameters
        ----------
        backend_config : dict
            Configuration for hardware simulator to be used. If None, uses
            default.qubit Pennylane simulator analitically, with normalized
            outputs.
        """
        if backend_config == None:
            self.model = lambeq.PennyLaneModel.from_diagrams(
                self.circuits_train + self.circuits_val + self.circuits_test
            )
        else:
            self.model = lambeq.PennyLaneModel.from_diagrams(
                self.circuits_train + self.circuits_val + self.circuits_test,
                backend_config
            )

    def create_trainer(self, kwargs)-> None:
        """
        Create lambeq trainer. 

        Parameters
        ----------
        kwargs : dict
            Keyword arguments for lambeq.PytorchTrainer object.
        """
        self.trainer = lambeq.PytorchTrainer(
            model = self.model, loss_function = self.loss_function, 
            epochs = self.epochs, optimizer = self.optimiser,
            device = self.device, optimizer_args = self.optimiser_args,
            seed = self.seed, **kwargs
        )

    def fit(
        self, kwargs_parser : dict = {}, kwargs_diagrams : dict = {},
        kwargs_trainer : dict  = {}, backend_config : dict = None
    )-> None:
        """
        Create diagrams, circuits, datasets, model, trainer. 
        After that fit the trainer with train and val datasets.

        Parameters
        ----------
        kwargs_parser : dict
            Dictionary with keyword arguments for a lambeq.BobcatParser object.
        kwargs_diagrams : dict
            Dictionary with keyword arguments for the method converting the
            sentences to diagrams. 
        kwargs_trainer : dict
            Keyword arguments for lambeq.PytorchTrainer object.
        backend_config : dict
            Configuration for hardware simulator to be used. If None, uses
            default.qubit Pennylane simulator analitically, with normalized
            outputs.
        """
        self.create_diagrams(kwargs_parser, kwargs_diagrams)
        self.create_circuits()
        self.create_dataset()
        self.create_model(backend_config)
        self.create_trainer(kwargs_trainer)
        self.trainer.fit(
            self.dataset_train, self.dataset_val)
        self.loss_train = self.trainer.train_epoch_costs
        self.loss_val = self.trainer.val_costs
        
    def compute_probabilities(self)-> None:
        """
        Compute train, validation and test probabilities.
        """
        self.probs_train = self.trainer.train_probabilities
        self.probs_val =  self.trainer.val_probabilities       
        probs_test_raw = self.model.get_diagram_output(self.circuits_test)
        self.probs_test = self.reshape_output_lambeq_model(probs_test_raw)
        self.probs_test = self.probs_test.tolist()
    
    def compute_predictions(self)-> None:
        """
        Compute train, validation and test predictions.
        """
        self.preds_train = self.trainer.train_predictions
        self.preds_val = self.trainer.val_predictions
        self.preds_test = []
        for i in range(len(self.probs_test)):
            self.preds_test.append(self.probs_test[i].index(max(self.probs_test[i])))




