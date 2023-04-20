from module.dataset_wrapper import *
from module.parameterised_quantum_circuit import *
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

#filename = "../../../data/datasets/Complete_dataset.json"
filename = "../../../data/datasets/amazonreview_reduced_bert_train.json"

###Training the model

#Set random seed
seed = 100

print("Initialisation Begun \n")
trainer = alpha_trainer(filename, seed)
print("Initialisation Ended \n")
#How many generations(epochs) to be ran?
number_of_epochs = 100

# Run the training number_of_runs times and average over the results
number_of_runs = 1
for i in range(number_of_runs):
    print("run = ", i+1, "\n")
    if i==0:
        loss_array, accuracy_array, prediction_array = trainer.train(number_of_epochs)
    else:
        loss_temp_array, accuracy_temp_array, prediction_temp_array  = trainer.train(number_of_epochs)
        loss_array += loss_temp_array
        accuracy_array += accuracy_temp_array
        prediction_array += prediction_temp_array 
loss_array = loss_array/number_of_runs
accuracy_array = accuracy_array/number_of_runs
prediction_array = prediction_array/number_of_runs

#Plotting the results
import matplotlib.pyplot as plt

plt.figure()
plt.plot(loss_array, label="Loss")
plt.plot(accuracy_array, label = "Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

"""
plt.figure()
plt.plot(log_loss_array)
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.show()
"""