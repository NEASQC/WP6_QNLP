from module.data_preparation import *
from module.training import *

"""main python file prepares data and trains a model on it.

"""

filename = "../../../data/Complete_dataset.json"

#preparing the data
dataset = data_preparation(filename)

# training the dressed quantum net
model = training(dataset)