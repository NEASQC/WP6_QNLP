from module.dataset_wrapper import *
from module.parameterised_quantum_circuit import *
from module.alpha_trainer import *

#from lambeq.ansatz.circuit import IQPAnsatz

import numpy as np

#import torch
#from torch.autograd import Function
#from torchvision import datasets, transforms
#import torch.optim as optim
#import torch.nn as nn
#import torch.nn.functional as F

###Choosing the dataset

filename = "../../../data/datasets/Complete_dataset.json"
#filename = "../../../data/datasets/amazon_filtered_dataset.json"

###Training the model

#Set random seed
seed = 0

print("Initialisation Begun")
trainer = alpha_trainer(filename, seed)
print("Initialisation Ended")
#How many generations(epochs) to be ran?
number_of_epochs = 2

# Run the training number_of_runs times and average over the results
number_of_runs = 2
for i in range(number_of_runs):
    print("run = ", i+1)
    if i==0:
        loss_array = trainer.train(number_of_epochs)
    else:
        loss_array += trainer.train(number_of_epochs)
loss_array = loss_array/number_of_runs

# Normalising the array
def normalise_array(array):
    return -array/sum(array)

loss_array = normalise_array(loss_array)
epoch_array = np.arange(0,len(loss_array))

save_array = [list(epoch_array), list(loss_array)]

print(save_array)

#Save save_array to results/raw

