import pandas as pd 
import lambeq
import pytket.extensions.qiskit as pyt
import numpy as np
import discopy
import torch


class PreAlphaLambeq:
    """
    A class to implement the lambeq version of pre-alpha
    """

    @staticmethod
    def load_dataset(
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
        sentences : list[str]
            List with the sentences of the dataset
        labels: list[list[int]]
            List with the labels of the dataset.
            [1, 0] False, and [0, 1] True
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
                labels.append([0,1])
            else:
                labels.append([1,0])

        return sentences[:20], labels[:20]

    @staticmethod
    def create_diagrams(
        data : list[str],
    ) -> list[discopy.rigid.Diagram]:
        """
        Creates diagrams for the sentences in our dataset and 
        simplifies them, removing the unnecesary cups

        Parameters
        ----------
        data : list[str]
            List with the sentences of the dataset

        Returns
        -------
        diagrams : list[discopy.rigid.Diagram]
            A list containing the simplified diagrams
        """
        parser = lambeq.BobcatParser()
        raw_diagrams = parser.sentences2diagrams(data)
        diagrams = [
            lambeq.remove_cups(diagram) for diagram in raw_diagrams]
        return diagrams
        
    @staticmethod    
    def create_circuits(
        diagrams : list[discopy.rigid.Diagram],
        ansatz : str = 'IQP', qn : int = 1, qs : int = 1, 
        n_layers : int = 1,
        n_single_qubits_params : int = 3,
    ) -> list[discopy.quantum.circuit.Circuit]:
        """
        Creates quantum circuits from the train and test diagrams

        Parameters
        ----------
        diagrams : list[discopy.rigid.Diagram]
            List with our discopy diagrams
        ansatz : str , default : IQP
            Type of ansatz to use on our quantum circuits (IQP, Sim14, Sim15,
            StronglyEntangling)
        qn : int , default : 1
            Number of qubits assigned to NOUN type
        qs : int , default : 1
            Number of qubits assigned to SENTENCE type
        n_layers : int , default : 1
            Number of layers in the circuit
        n_single_qubit_params : int , default : 3
            Number of variational parameters assigned to each qubit

        Returns
        -------
        circuits : list[discopy.quantum.circuit.Circuit]
            A list containing the quantum circuits
        """
        if ansatz == "IQP":
            ansatz = lambeq.IQPAnsatz(
                {lambeq.AtomicType.NOUN: qn, lambeq.AtomicType.SENTENCE: qs},
                n_layers=n_layers,
                n_single_qubit_params=n_single_qubits_params
                )
        elif ansatz == "Sim14":
            ansatz = lambeq.Sim14Ansatz(
                {lambeq.AtomicType.NOUN: qn, lambeq.AtomicType.SENTENCE: qs},
                n_layers=n_layers,
                n_single_qubit_params=n_single_qubits_params
            )
        elif ansatz == "Sim15":
            ansatz = lambeq.Sim15Ansatz(
                {lambeq.AtomicType.NOUN: qn, lambeq.AtomicType.SENTENCE: qs},
                n_layers=n_layers,
                n_single_qubit_params=n_single_qubits_params
                )
        elif ansatz == "StronglyEntangling":
            ansatz = lambeq.StronglyEntanglingAnsatz(
                {lambeq.AtomicType.NOUN: qn, lambeq.AtomicType.SENTENCE: qs},
                n_layers=n_layers,
                n_single_qubit_params=n_single_qubits_params
            )
        circuits = [
            ansatz(diagram) for diagram in diagrams
        ]

        return circuits

    @staticmethod
    def create_dataset(
        circuits : list[discopy.quantum.circuit.Circuit],
        labels : list[list[int]]
    ) -> lambeq.Dataset:
        """
        Creates a Dataset class for the training of the lambeq model

        Parameters
        ----------
        circuits : list[discopy.quantum.circuit.Circuit]
            List containing quantum circuits
        labels : list[list[int]]
            List containing our labels

        Returns
        -------
        dataset : lambeq.Dataset
            A lambeq dataset that can be used for training 
        """
        dataset = lambeq.Dataset(
            circuits, labels
        )
        return dataset
   
    @staticmethod
    def create_model(
        all_circuits : list[discopy.quantum.circuit.Circuit]
    ) -> lambeq.PennyLaneModel:
        """
        Creates a model that can be used for training

        Parameters
        ----------
        all_circuits : list[discopy.quantum.circuit.Circuit]
            List with both training and testing circuits.
        Returns
        -------
        model : lambeq.training
            Model that can be use in training 
        """
        model = lambeq.PennyLaneModel.from_diagrams(
            all_circuits
        )
        return model

    @staticmethod    
    def create_trainer(
        model : lambeq.PennyLaneModel, loss_function : torch.nn.functional,
        optimizer : torch.optim.Optimizer, epochs : int,
        learning_rate : float = 0.001, optimizer_args : dict = None,
        seed : int = 18051967
        ) -> lambeq.QuantumTrainer:
        """
        Creates a lambeq trainer 

        Parameters
        ----------
        model : lambeq.PennyLaneModel
            Model to be used in training
        loss_function : torch.nn.functional
            Loss function to be used in training
        optimizer : torch.optim.Optimizer
            Optimizer to be used in training
        epochs : int
            Epochs to be performed in optimization
        learning_rate : float, default : 0.001
            Learning rate provided to the optimizer
        optimizer_args : dict, default : None
            Optional arguments to be passed to the optimizer
        seed : int, default : 18051967
            Seed used in the optimizer

        Returns
        -------
        trainer : lambeq.QuantumTrainer
            A lambeq trainer for our set of sentences. 
        """
        trainer = lambeq.PytorchTrainer(
        model = model,
        loss_function = loss_function,
        epochs = epochs,
        optimizer = optimizer,
        learning_rate = learning_rate,
        optimizer_args = optimizer_args,
        seed=seed
        )
        return trainer

    @staticmethod
    def post_selected_output(
        circuit : discopy.quantum.circuit.Circuit,
        model : lambeq.PennyLaneModel
    ) -> np.array:
        """
        For a given circuit, outputs the normalised
        probabilities of each state of the postselected qubits

        Parameters
        ----------
        circuit : discopy.quantum.circuit.Circuit
            Quantum Circuit we want to analyse
        model : lambeq.PennyLaneModel
            A lambeq model storing the quantum parameters
            assigned per word. 

        Returns 
        -------
        post_selected_output : np.array
            Array containig the proabilities of each state 
        """
        circuit = circuit.to_pennylane()
        circuit.initialise_concrete_params(
            model.symbols, model.weights
        )
        post_selected_output = circuit.eval().detach().numpy()
        post_selected_output = post_selected_output / np.sum(post_selected_output)
        return post_selected_output
    
    @staticmethod
    def predicted_label(
        post_selected_output : np.array
    ) -> int:
        """
        Assigns a label depeding on the probabilities of the post-selected output

        Parameters
        ----------
        post_selected_output : np.array
            Array containing the probabilities of each state
        
        Returns
        -------
        prediction : int
            Predicted label 
        """
        if post_selected_output[0] > 0.5:
            prediction = 0
            return prediction
        else : 
            prediction = 1
            return prediction
