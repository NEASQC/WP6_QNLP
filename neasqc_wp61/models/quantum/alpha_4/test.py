import numpy as np 
import torch 
import os 
import sys
import pennylane as qml  

from utils import *

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../")
import circuit as circ 
from alpha_4 import Alpha4 as alpha4
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


torch.manual_seed(901)
n_qubits = 4
n_layers = 2 
axis_embedding = 'Y'
n_classes = 3
observables = {0 : qml.PauliX, 1 : qml.PauliZ, 2 : qml.PauliZ}
mycirc = circ.Sim14(
    n_qubits, n_layers, axis_embedding, 
    observables, output_probabilities = True
)

labels = get_labels_one_hot_encoding(
    y_train.tolist(), y_val.tolist(), y_test.tolist()
)[0]
n_classes = get_labels_one_hot_encoding(
   y_train.tolist(), y_val.tolist(), y_test.tolist()
)[1]

inputs = torch.randn(3,3)

alpha4_model = alpha4(
        sentence_vectors = [X_train, X_val, X_test],
        labels = labels,
        n_classes = n_classes,
        circuit = mycirc,
        optimiser = torch.optim.Adam,
        epochs = 20,
        batch_size = 3,
        loss_function = torch.nn.CrossEntropyLoss,
        optimiser_args = {'lr' : 0.1},
        device = 'cpu',
        seed = 300
)

print(alpha4_model.forward(inputs))
alpha4_model.train()
alpha4_model.compute_preds_probs_test()
print(alpha4_model.preds_test)
print(alpha4_model.probs_test)
