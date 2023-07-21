from torch import nn
from lambeq import PennyLaneModel
import torch
import pennylane as qml
from pennylane import numpy as np

# inherit from PennyLaneModel to use the PennyLane circuit evaluation
class Alpha_pennylane_model(PennyLaneModel):
    def __init__(self, n_qubits, q_delta, device): 
        """
        Definition of the *dressed* layout.
        """

        super().__init__()

        self.n_qubits = n_qubits
        self.q_delta = q_delta
        self.device = device

        self.pre_net = nn.Linear(768, self.n_qubits)
        self.q_params = nn.Parameter(self.q_delta * torch.randn((self.n_qubits + 2) * self.n_qubits))
        self.post_net = nn.Linear(self.n_qubits, 2)
        self.sigmoid = nn.Sigmoid()

        dev = qml.device('default.qubit', wires=self.n_qubits)

        self.quantum_net = qml.QNode(quantum_net_from_paper, dev, interface="torch")

    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0
        #q_in = torch.tanh(pre_out) * np.pi

        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(0, self.n_qubits)
        q_out = q_out.to(self.device)
        for elem in q_in:
            q_out_elem = torch.hstack(self.quantum_net(elem, self.q_params, self.n_qubits)).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))

        # return the two-dimensional prediction from the postprocessing layer
        return self.sigmoid(self.post_net(q_out))
    
    

def quantum_net_from_paper(q_input_features, q_weights, n_qubits):
    #https://onlinelibrary.wiley.com/doi/epdf/10.1002/qute.201900070
    #Circuit [6] but adapted for n_qubits = 3

    # Embed features in the quantum node
    RX_layer(q_input_features)

    parameter_index = 0

    RZ_layer(q_weights[parameter_index : parameter_index+n_qubits])

    parameter_index += n_qubits

    for k in range(0, n_qubits):
        for j in range(0, n_qubits):
            if k != j:
                qml.CRX(phi=q_weights[parameter_index], wires=[k, j])
                parameter_index += 1


    RX_layer(q_weights[parameter_index : parameter_index+n_qubits])
    parameter_index += n_qubits

    RZ_layer(q_weights[parameter_index : parameter_index+n_qubits])
    parameter_index += n_qubits

    exp_vals_Y = [qml.expval(qml.PauliY(position)) for position in range(n_qubits)]

    return tuple(exp_vals_Y)


def RX_layer(w):
    """Layer of parametrized qubit rotations around the x axis.
    """
    for idx, element in enumerate(w):
        qml.RX(element, wires=idx)

def RZ_layer(w):
    """Layer of parametrized qubit rotations around the z axis.
    """
    for idx, element in enumerate(w):
        qml.RZ(element, wires=idx)