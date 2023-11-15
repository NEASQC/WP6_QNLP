from torch import nn
from lambeq import PennyLaneModel
import torch
import pennylane as qml
from pennylane import numpy as np

# inherit from PennyLaneModel to use the PennyLane circuit evaluation
class Alpha_3_multiclass_model(PennyLaneModel):
    def __init__(self, n_qubits, q_delta, n_classes, device): 
        """
        Definition of the *dressed* layout.
        """

        super().__init__()

        self.n_qubits = n_qubits
        self.q_delta = q_delta
        self.device = device
        self.n_classes = n_classes

        self.pre_net = nn.Linear(768, self.n_qubits)
        self.q_params = nn.Parameter(self.q_delta * torch.randn((self.n_qubits + 2) * self.n_qubits))
        self.post_net = nn.Linear(self.n_qubits, self.n_classes)
        self.softmax = nn.Softmax()

        
        dev = qml.device('default.qubit', wires=self.n_qubits)

        self.quantum_net = qml.QNode(self.quantum_net_from_paper, dev, interface="torch")


    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

       # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 768 to self.n_qubits
        pre_out = self.pre_net(input_features)  #Here input_features.shape=(batch_size, 768) and pre_out.shape=(batch_size, self.n_qubits)
        q_in = torch.tanh(pre_out) * np.pi / 2.0    #Here q_in.shape=(batch_size, self.n_qubits) and have values between -pi/2 and pi/2
        
        
        q_out = self.quantum_net(q_in, self.q_params) #Output q_out a list with the shape (self.n_qubits, batch_size)
        q_out = torch.stack(q_out)  # Stack outputs into one tensor of shape (self.n_qubits, batch_size)
        q_out = q_out.transpose(0, 1).float()  # Transpose dimensions 0 and 1 to get a tensor of shape (batch_size, self.n_qubits)

        # return the two-dimensional prediction from the postprocessing layer
        return self.softmax(self.post_net(q_out))
    


    def quantum_net_from_paper(self, inputs, q_weights):
        #https://onlinelibrary.wiley.com/doi/epdf/10.1002/qute.201900070
        #Circuit [6] but adapted for n_qubits


        qml.AngleEmbedding(features=inputs, wires=range(self.n_qubits), rotation='X')

        parameter_index = 0

        #RZ_layer:
        for idx, element in enumerate(q_weights[parameter_index : parameter_index+self.n_qubits]):
            qml.RZ(element, wires=idx)

        parameter_index += self.n_qubits

        for k in range(0, self.n_qubits):
            for j in range(0, self.n_qubits):
                if k != j:
                    qml.CRX(phi=q_weights[parameter_index], wires=[k, j])
                    parameter_index += 1

        #RX_layer:
        for idx, element in enumerate(q_weights[parameter_index : parameter_index+self.n_qubits]):
            qml.RX(element, wires=idx)

        parameter_index += self.n_qubits

        #RZ_layer:
        for idx, element in enumerate(q_weights[parameter_index : parameter_index+self.n_qubits]):
            qml.RZ(element, wires=idx)

        parameter_index += self.n_qubits

        exp_vals_Y = [qml.expval(qml.PauliY(position)) for position in range(self.n_qubits)]

        
        return exp_vals_Y