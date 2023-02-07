from module.dataset_wrapper import *
from module.parametrised_quantum_circuit import *
from module.alpha_trainer import *

from lambeq.ansatz.circuit import IQPAnsatz

import numpy as np

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

###Choosing the dataset
filename = "../../../data/Complete_dataset.json"

###dataset_wrapper parses the data and finds the bert embeddings for each sentence
wrapper = dataset_wrapper(filename)

###Defining the fixed parametrised_quantum_circuit
sentence_type = 'NOUN-IVERB-TVERB-PREP-NOUN'

#sentence_type = 'NOUN-IVERB-NOUN-TVERB-NOUN'
sentence= "dog ate cat addressing audience"
#sentence_type = "dog ate cat"
pqc = parametrised_quantum_circuit(sentence, sentence_type)

print(pqc.word_number_of_parameters)

parameters = np.random.rand(sum(pqc.word_number_of_parameters))

print(parameters)

output = pqc.run_circuit(parameters)

print(output)

output = pqc.run_circuit(parameters)

print(output)

###Training the model

trainer = alpha_trainer(filename, sentence, sentence_type)

number_of_epochs = 5

loss_array = trainer.train(number_of_epochs)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(loss_array)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()