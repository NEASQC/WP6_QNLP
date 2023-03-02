import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../models/quantum/alpha/")
import json
import numpy as np
from module.dataset_wrapper import *
from module.parameterised_quantum_circuit import *
from module.alpha_trainer import *

########################################
#Set experiment number
experiment = "002"
#How many generations(epochs) to be ran?
number_of_epochs = 2
# Run the training number_of_runs times and average over the results
number_of_runs = 2
#Set random seed
seed = 0
########################################
filename = "../../../data/datasets/Complete_dataset.json"
#filename = "../../../data/datasets/amazon_filtered_dataset.json"

###Training the model



print("Initialisation Begun")
trainer = alpha_trainer(filename, seed)
print("Initialisation Ended")

for i in range(number_of_runs):
    print("run = ", i+1)
    if i==0:
        loss_array = trainer.train(number_of_epochs)
    else:
        loss_array += trainer.train(number_of_epochs)
loss_array = loss_array/number_of_runs

# Normalising the array
def normalise_array(array):
    return -array/np.sqrt(sum(array**2))

loss_array = normalise_array(loss_array)
epoch_array = np.arange(0,len(loss_array))

save_array = [list(epoch_array), list(loss_array)]

#Save save_array to results/raw
jsonString = json.dumps(save_array)
jsonFile = open(f"../../../benchmarking/results/raw/results_{experiment}.json", "w")
jsonFile.write(jsonString)
jsonFile.close()