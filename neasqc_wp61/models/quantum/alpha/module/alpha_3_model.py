from torch import nn
from lambeq import PennyLaneModel
import torch
import pennylane as qml
from pennylane import numpy as np

# inherit from PennyLaneModel to use the PennyLane circuit evaluation
class Alpha_3_model(PennyLaneModel):
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

        self.quantum_net = qml.QNode(self.quantum_net_from_paper, dev, interface="torch")


    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0
        
        
        q_out = self.quantum_net(q_in, self.q_params)
        q_out = torch.stack(q_out)
        q_out = q_out.transpose(0, 1).float()  # Transpose dimensions 0 and 1

        # return the two-dimensional prediction from the postprocessing layer
        return self.sigmoid(self.post_net(q_out))
    


    def quantum_net_from_paper(self, inputs, q_weights):
        #https://onlinelibrary.wiley.com/doi/epdf/10.1002/qute.201900070
        #Circuit [6] but adapted for n_qubits = 3


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