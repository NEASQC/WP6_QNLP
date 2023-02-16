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

class alpha_trainer(nn.Module):
    """Trains a neural network that takes sentence BERT embeddings as input and maps them to the parameters in a parameterised quantum circuit.

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
        self.wrapper = dataset_wrapper(filename)
        self.sentences = self.wrapper.sentences
        self.sentence_types = self.wrapper.sentence_types
        self.sentence_labels = self.wrapper.sentence_labels
        self.bert_embeddings = self.wrapper.bert_embeddings
        
        ###Defining the network
        self.BertDim = 768
        intermediate_dimension= 10
        max_param = 9
        min_param = 4
        
        #Create a neural network    
        self.pre_net = nn.Linear(self.BertDim, intermediate_dimension)
        self.pre_net_max_params = nn.Linear(intermediate_dimension, max_param)
        self.cascade = nn.ParameterList()
        for layer in range(max_param,min_param,-1):
            self.cascade.append(nn.Linear(layer, layer-1))
        

        
    def train(self, number_of_epochs):
        """Trains model.

        Uses a binary cross entropy loss criterion and stochastic gradient descent optimiser to train the model of a number of epochs.:

        Parameters
        ----------
        number_of_epochs: int
            the number of epochs over which you loop over the data and train the model
        
        Returns
        -------
        Dataset: np.array()
            array of loss for each epoch
            
        """
        ###Training the model
        
        criterion = nn.BCELoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        
        # generation loop
        loss_array = []
        
        for epoch in range(number_of_epochs):
            tic = time.perf_counter()
            # sentence loop
            running_loss = 0
            for specific_sentence in self.sentences:
                sentence_index = self.sentences.index(specific_sentence)
                sentence_label = self.sentence_labels[sentence_index]
                       
                # 1. forward step: takes sentence embeddings and outputs input parameters to pqc --> parameters, run_circuit(parameters) --> output=[x,1-x]
                output = self.forward(specific_sentence)
                
                # 3. compute loss(compare output to sentence label)
                
                loss = criterion(input=torch.Tensor(output), target=torch.Tensor(sentence_label))
                loss = torch.autograd.Variable(loss, requires_grad = True)

                # 4. backward step --> updated network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                #print stats
                running_loss += loss.item()
            
            toc = time.perf_counter()
            print("Epoch = ", epoch+1)
            print(f"Epoch time taken: {toc - tic:0.4f} seconds")
            print("Loss = ", running_loss/len(self.sentences))
            loss_array.append(running_loss/len(self.sentences))
                
        return np.array(loss_array)
    
    
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
        
        pqc = parameterised_quantum_circuit(specific_sentence)
        word_number_of_parameters = pqc.word_number_of_parameters
        
        counter = 0
        sentence_q_params = []
        for i, embedding in enumerate(self.bert_embeddings[sentence_index][0]):
            pre_out = self.pre_net.float()(torch.tensor(embedding))
            pre_out = self.pre_net_max_params(pre_out)
            for j, layer in enumerate(self.cascade):
                layer_n_out = layer.out_features
                if word_number_of_parameters[i] <= layer_n_out:
                    pre_out = self.cascade[j](pre_out)
            q_in = torch.tanh(pre_out) * np.pi / 2.0 
            sentence_q_params.append(q_in)
        self.qparams = torch.cat(sentence_q_params)
        
        output = pqc.run_circuit(self.qparams)
            
        return output
    

                
        