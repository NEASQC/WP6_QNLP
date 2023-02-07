from module_fix.dataset_wrapper import *
from module_fix.parametrised_quantum_circuit import *

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class alpha_trainer(nn.Module):
    
    
    
    def __init__(self,filename:str, sentence:str, sentence_type:str):
        super().__init__()
        ###dataset_wrapper parses the data and finds the bert embeddings for each sentence
        self.wrapper = dataset_wrapper(filename)
        self.sentences = self.wrapper.sentences
        self.sentence_types = self.wrapper.sentence_types
        self.sentence_labels = self.wrapper.sentence_labels
        self.bert_embeddings = self.wrapper.bert_embeddings

        ###Defining the fixed parametrised_quantum_circuit
        self.sentence = sentence
        self.sentence_type = sentence_type
        self.pqc = parametrised_quantum_circuit(self.sentence, self.sentence_type)
        self.word_number_of_parameters = self.pqc.word_number_of_parameters
        self.number_of_parameters = sum(self.word_number_of_parameters)
        
        ### obtaining sentence type boolean arrays
        self.sentence_type_inputs = self.get_word_order()
        
        ###Defining the network
        self.BertDim = 768
        intermediate_dimension= 20
        max_param = 15
        min_param = 1
        
        #Create a network for each word in the general sentence type
        general_word_types = self.sentence_type.split("-")
        self.number_of_network_elements = len(general_word_types)
        
        self.pre_net = nn.ParameterList()
        self.pre_net_max_params = nn.ParameterList()
        self.cascade = nn.ParameterList()
        for i in range(self.number_of_network_elements):
            self.pre_net.append(nn.Linear(self.BertDim, intermediate_dimension))
            self.pre_net_max_params.append(nn.Linear(intermediate_dimension, max_param))
            cascade_layers = []
            for layer in range(max_param,self.word_number_of_parameters[i],-1):
                cascade_layers.append(nn.Linear(layer, layer-1))
            self.cascade.append(cascade_layers)  

        
    def train(self, number_of_epochs):
        ###Training the model
        
        criterion = nn.BCELoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        
        # generation loop
        loss_array = [0]
        
        for epoch in range(number_of_epochs):
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
                loss.backward()
                optimizer.step()

                #print stats
                running_loss += loss.item()
                print("Running Loss = ",running_loss)
            loss_array.append(running_loss/len(self.sentences))
                
        return loss_array
    
    
    def forward(self, specific_sentence):
        #Takes in bert embeddings, assigns them to correct transformation, then outputs results for running the circuit
        # Requires pqc parameter numbers
        sentence_index = self.sentences.index(specific_sentence)
        
        zero_input = [0.0]*self.BertDim
        
        
        counter = 0
        sentence_q_params = []
        for i in range(self.number_of_network_elements):
            if self.sentence_type_inputs[sentence_index][i] == 1:
                embedding = self.bert_embeddings[sentence_index][0][counter]
                counter+=1
            else:
                embedding = zero_input
            pre_out = self.pre_net[i].float()(torch.tensor(embedding))
            pre_out = self.pre_net_max_params[i](pre_out)
            for j, layer in enumerate(self.cascade[i]):
                pre_out = self.cascade[i][j](pre_out)
            q_in = torch.tanh(pre_out) * np.pi / 2.0 
            sentence_q_params.append(q_in)
        self.qparams = torch.cat(sentence_q_params)
        
        self.pqc = parametrised_quantum_circuit(self.sentence, self.sentence_type)
        output = self.pqc.run_circuit(self.qparams)
            
        return output
    
    def get_word_order(self):
        word_orders = []
        general_word_types = self.sentence_type.split("-")
        number_of_elements = len(general_word_types)
        for sentence_type_input in self.sentence_types:
            word_order = [0]*number_of_elements
            word_types = sentence_type_input.split("-")
            word_type_index_min=0
            for word_type in word_types:
                order_index = 0
                for general_word_type in general_word_types[word_type_index_min:]:
                    if word_type == general_word_type:
                        word_order[word_type_index_min+order_index] = 1
                        order_index += 1
                        word_type_index_min+=1
                        break
                    else:
                        order_index += 1
            word_orders.append(word_order)
        return word_orders
    

                
        