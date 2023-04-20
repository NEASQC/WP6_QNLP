import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "models/quantum/alpha/")
import json
import numpy as np
from models.quantum.alpha.module.dataset_wrapper import *
from models.quantum.alpha.module.parameterised_quantum_circuit import *
from models.quantum.alpha.module.alpha_trainer import *
from models.quantum.alpha.module.alpha_model import *

########################################
#Set experiment number
experiment = "002"
#How many generations(epochs) to be ran?
number_of_epochs = 2
# Run the training number_of_runs times and average over the results
number_of_runs = 1
#Set random seed
seed = 100
########################################
filename = "data/datasets/amazonreview_reduced_bert_train.json"
###Training the model


print("Initialisation Begun \n")
trainer = alpha_trainer(filename, seed)
print("Initialisation Ended \n")

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

save_array = list(prediction_array)

#Save save_array to results/raw           
file = open(f"benchmarking/results/raw/results_alpha_{experiment}.json", "w")
for item in save_array:
    file.write(f"{int(item)}"+"\n")
file.close()               