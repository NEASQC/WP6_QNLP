import torch
import torch.optim as optim
import torch.nn as nn
    
from module.DressedQuantumNet import DressedQuantumNet
from module.Qsentence import Qsentence

def training(Dataset: list)->DressedQuantumNet:
    """Trains Dressed Quantum Neural Network Classfier.

    Takes in a list of Qsentence types and trains a Dressed Quantum Network using PyTorch.:

    Parameters
    ----------
    Dataset : list
        list of Qsentence types.
   

    Returns
    -------
    DressedNet: DressedQuantumNet
        Trained dressed quantum netowrk model

    """
    
    #DressedNet = DressedQuantumNet(Qsentence("Alice repairs car",n_dim=1, s_dim=1, depth = 1))
    DressedNet = DressedQuantumNet(Dataset[0])
    
    criterion = nn.BCELoss()
    optimizer = optim.SGD(DressedNet.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        for count,sentence in enumerate(Dataset):
            label = sentence.label

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            net = DressedQuantumNet(sentence)
            outputs = torch.Tensor(net.forward())
            print("count = ",count,"  sentence = ", sentence)
            print("outputs = ", outputs)
            loss = criterion(input=outputs, target=torch.Tensor(label))
            loss = torch.autograd.Variable(loss, requires_grad = True)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print("Count = ",count)
            print("Running Loss = ",running_loss/2000)
            #if i % 2000 == 1999:    # print every 2000 mini-batches
                #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                #running_loss = 0.0
    print('Finished Training')
    print(DressedNet.state_dict())
    return DressedNet