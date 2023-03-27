from module.dataset_wrapper import *
from module.parameterised_quantum_circuit import *


import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

import time

class alpha_model(nn.Module):
    """Defines a neural network that takes sentence BERT embeddings as input and maps them to the parameters in a parameterised quantum circuit.

    ........

    Attributes
    ----------
    
    seed: int
        Random seed setting
    wrapper: dataset_wrapper
        Generates dataset_wrapper object using filename
    sentences: list
        list of sentences in the dataset
    sentence_types: list
        list of sentence types(grammatical structures)
    sentence_labels: list
        list of sentence classification labels
    bert_embeddings: list[list[floats]]
        The BERT embeddings for each word in each sentence.
    BertDim: int
        Dimension of the BERT embedding vectors
    pre_net: nn.Linear
        Network that takes Bert Embedding input and maps it to an intermediate dimension layer.
    pre_net_max_params: nn.Linear
        Network that maps intermediate diimension to a layer with dimension equal to an estimated maximum number of parameters required for the paramterised quantum circuits.
    cascade = nn.ParameterList(nn.Linear)
        Network that maps the maximum parameters kayer to a layer with dimension equal to the number of parameters required for a particular quantum circuit.

    """
    
    def __init__(self,filename:str, seed: int):
        """Initialises alpha_trainer.

        Defines the attributes in alpha_trainer, inculding the neural network.
        

        Parameters
        ----------
        filename : str
            Input data json file
        seed: int
            Random seed setting for reproducibility
            

        """
        
        super().__init__()
        
        #Set random seed
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        ###dataset_wrapper parses the data and finds the bert embeddings for each sentence
        #self.reduced_word_embedding_dimension = 146
        self.reduced_word_embedding_dimension = 22
        self.wrapper = dataset_wrapper(filename, self.reduced_word_embedding_dimension)
        self.sentences = self.wrapper.sentences
        self.sentence_types = self.wrapper.sentence_types
        self.sentence_labels = self.wrapper.sentence_labels
        self.bert_embeddings = self.wrapper.bert_embeddings
        self.BertDim = self.wrapper.reduced_word_embedding_dimension
        
        ###Define the noprmalisation factor for normalised cross entropy loss
        #p = (sum(np.array(self.sentence_labels)==[1,0])/len(self.sentence_labels))[0]
        #p = (sum(np.array(self.sentence_labels[0:10])==[1,0])/len(self.sentence_labels[0:10]))[0]
        #self.normalisation_factor = len(self.sentence_labels[0:10])*(p*np.log(p) +(1-p)*np.log(1-p))
        #self.normalisation_factor = len(self.sentences)*100
        
        
        ###Initialise the circuits class
        self.pqc_sentences = parameterised_quantum_circuit(self.sentences)
        
        ###Defining the network
        intermediate_dimension= 20
        max_param = 10
        min_param = 1
        
        #Create a neural network    
        self.pre_net = nn.Linear(self.reduced_word_embedding_dimension, intermediate_dimension)
        self.pre_net_max_params = nn.Linear(intermediate_dimension, max_param)
        self.cascade = nn.ParameterList()
        for layer in range(max_param,min_param,-1):
            self.cascade.append(nn.Linear(layer, layer-1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, specific_sentence):
        """Performs a forward step in the model training.

        Parameters
        ----------
        specific_sentence: str
            sentence that the model trains on
        
        Returns
        -------
        output: list[floats]
            [x,1-x] binary classification output.
        """   
        #Takes in bert embeddings, assigns them to correct transformation, then outputs results for running the circuit
        # Requires pqc parameter numbers
        sentence_index = self.sentences.index(specific_sentence)
        
        circuit, parameters, word_number_of_parameters = self.pqc_sentences.create_tket_circuit(sentence_index)
        
        counter = 0
        sentence_q_params = []
        for i, embedding in enumerate(self.bert_embeddings[sentence_index]):
            embedding = list(map(float, embedding))
            
            pre_out = self.pre_net.float()(torch.tensor(embedding))
            pre_out = self.dropout(pre_out)
            pre_out = self.pre_net_max_params(pre_out)
            pre_out = self.relu(pre_out)
            for j, layer in enumerate(self.cascade):
                layer_n_out = layer.out_features
                if word_number_of_parameters[i] <= layer_n_out:
                    pre_out = self.cascade[j](pre_out)
                    pre_out = self.relu(pre_out)
            q_in= torch.tanh(pre_out) * np.pi / 2.0
            sentence_q_params.append(q_in)
        qparams = torch.cat(sentence_q_params)
        
        circuit_output = torch.Tensor(self.pqc_sentences.run_circuit(circuit, qparams.clone()))
        
        #reshape arrays
        qparams = torch.reshape(qparams, (1,qparams.size(dim=0)))
        circuit_output = torch.reshape(circuit_output, (1,circuit_output.size(dim=0)))
        
        # find torch linear transformation that represents pqc transforamtion
        transformation = self.linear_transformer(qparams.clone(),circuit_output)
        
        output = qparams@transformation
        output = torch.reshape(output, (output.size(dim=1),))
        
        return torch.clamp(output, min=0.0, max=1.0)
        
    
    def linear_transformer(self,array1, array2):
        """Finds linear transformation between two arrays as torch tensor
        """
        
        transformation = torch.linalg.lstsq(array1,array2).solution
        
        return transformation