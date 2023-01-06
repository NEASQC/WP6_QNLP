import numpy as np
import matplotlib.pyplot as plt

import nltk

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit, IBMQBackend
from pytket.qasm import circuit_to_qasm_str

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit, IBMQBackend

import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *

from pytket.qasm import circuit_to_qasm_str

import itertools

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from pytket import Circuit, Qubit, Bit
from pytket.extensions.qiskit import AerBackend
from pytket.utils import probs_from_counts


from module.Qsentence import *

class DressedQuantumNet(nn.Module):


    """Neural Network Classifier with Forward Method.

    Defines the Dressed Quantum Neural Network Classifier for a Sentence. A feed forward step is also defined for later training along with supporting methods.
    This class implements the pre processing neural network needed to reduce the dimensionality of BERT embeddings. A dimension for an intermediate representation, as well as the max and min numbers of parameters that are expected to appear in the circuits is also provided. A different number of layers will be applied depending of the numbers of parameters needed to encode that word.
    When the circuit is run using pytket and AerBackend(), the tensor network nature of the parameters is lost, and all we is a classical bitstring and some probabilities. Alternative options like using differentiable Pennylane circuits could solve this issue. An interface between Pennylane and tket exists and it is worth looking into it. 

    Attributes
    ----------
    Sentence : str
        Input Sentence
    QNParamWords : list
        Number of parameters for each word.

    """

    def __init__(self, Sentence):
        """Initialises DressedQuantumNet.

        Defines a neural network implemented before the paremtrised quantum circuit.
       

        Parameters
        ----------
        Sentence : str
            Input sentence.

        """
        super().__init__()
        BertDim = 768
        intermediate_dimension= 20
        max_param = 5
        min_param = 1
        self.Sentence = Sentence
        self.QNParamWords = Sentence.GetNParamsWord()
        self.pre_net = nn.Linear(BertDim, intermediate_dimension)
        self.pre_net_max_params = nn.Linear(intermediate_dimension, max_param)
        self.cascade_layers = []
        
        #CASCADE LAYERS: Consecutive layers reduce the number of parameters until getting the desired number
        #for that specific word
        
        for layer in range(max_param,min_param,-1):
            self.cascade_layers.append(nn.Linear(layer, layer-1))
            
            
    def forward(self):
        """Performs forward step in neural network.

        Takes a list of sentences and find a Bert embedding for each.:


        Returns
        -------
        result_dict.values(): list
            Outputs a two-dimensional list of floats that represents the classification of the Neural Network. True corresponds to [1,0] and False correseponds to [0,1]
            
        """
        sentence_q_params = []
        for i, embedding in enumerate(self.Sentence.embeddings[0]):
            n_q_params = self.QNParamWords[i]
            pre_out = self.pre_net(torch.tensor(embedding))
            pre_out = self.pre_net_max_params(pre_out)
            for j, layer in enumerate(self.cascade_layers):
                layer_n_out = layer.out_features
                if n_q_params <= layer_n_out:
                    pre_out = self.cascade_layers[j](pre_out)
            q_in = torch.tanh(pre_out) * np.pi / 2.0  
            sentence_q_params.append(q_in)
        self.qparams = torch.cat(sentence_q_params)
        parameter_names = self.Sentence.tk_circuit.free_symbols()
        self.parameter_names = parameter_names
        param_dict = {p: q for p, q in zip(self.parameter_names, self.qparams)}
        MyCirc = self.Sentence.tk_circuit
        s_qubits = self.Measure_s_qubits(MyCirc)
        MyCirc.symbol_substitution(param_dict)
        backend = AerBackend()
        #backend.get_compiled_circuits([MyCirc])

        handle = backend.process_circuits(backend.get_compiled_circuits([MyCirc]), n_shots=2000)
        counts = backend.get_result(handle[0]).get_counts()
        result_dict = self.get_norm_circuit_output(counts, s_qubits)
        all_bitstrings = self.calculate_bitstring(s_qubits)
        for bitstring in all_bitstrings:
            if bitstring not in result_dict.keys():
                result_dict[bitstring] = 0
        return list(result_dict.values())
    
    def Measure_s_qubits(self, Circuit):
        """Obtains unmeasured qubits meausrements.

        In the DisCoCat pytket circuits the sentence qubits are not measured, and thus additional measurements
        need to be performed. Otherwise, we will get bitsrings shorter than the number of qubits of the circuits, 
        corresponding only to the post selected ones.:


        Returns
        -------
        sen_c_regs: list
            list of measurements.
            
            
        """
        s_qubits=[]
        for qubit in Circuit.qubits:
            if qubit not in list(Circuit.qubit_readout.keys()):
                s_qubits.append(qubit.index[0])
        n_post_select = len(Circuit.bit_readout.keys())
        for i, s_qubit in enumerate(s_qubits):
            Circuit.add_bit(Bit("c", n_post_select+i))
        sen_c_regs = list(Circuit.bit_readout.keys())[n_post_select:]
        for i, qubit in enumerate(s_qubits):
            bit = list(Circuit.bit_readout.keys()).index(sen_c_regs[i])
            Circuit.Measure(qubit, bit)
        return sen_c_regs

    
    def satisfy_post_selection(self,post_selection, result):
        """Checks post selection criteria for circuit.

        This is used to tell if the output bitstrings satify the post selection conditions given by the ciruit.:

        Parameters
        -------
        post_selection: iterable
        
        result: iterable
        
        Returns
        -------
        bool
            
            
        """
        for index, value in enumerate(result):
            if index in post_selection:
                if value != post_selection[index]:
                    return False
        return True

    def list2bitstring(self,bitlist: list):
        """Converts bit list to bit string.

        Parameters
        -------
        bitlist: str
        
        Returns
        -------
        bitstring: str
            
        """
        bitstring=str()
        for i in bitlist:
            bitstring+=str(i)
        return bitstring

    def norm_probs(self,prob_result):
        """Normalises values in dictionary

        Parameters
        -------
        prob_result: dict
        
        Returns
        -------
        prob_result: dict
            
        """
        tot = sum(list(prob_result.values()))
        for bitstring in prob_result.keys():
            prob_result[bitstring]/=tot
        return prob_result

    def get_norm_circuit_output(self, counts, s_qubits):
        """Obtains normalised output of parametrised quantum circuit.

        Parameters
        -------
        counts: 
        
        s_qubits:
        
        Returns
        -------
        self.norm_probs(prob_result): dict
            
        """
        prob_result=dict()
        for bits in probs_from_counts(counts).keys():
            post_selected = self.satisfy_post_selection(self.Sentence.tk_circuit.post_selection, bits)
            if post_selected==True:
                s_qubits_index = []
                for qubit in s_qubits:
                    s_qubits_index.append(qubit.index[0])
                bitlist = list( bits[i] for i in s_qubits_index)
                bitstring = self.list2bitstring(bitlist)
                prob_result[bitstring] = probs_from_counts(counts)[bits]
        return self.norm_probs(prob_result)
    

    def calculate_bitstring(self, s_qubits):
        """Finds bitstrings for each qubit.

        Parameters
        -------
        
        s_qubits: 
        
        Returns
        -------
        all_bitstrings: list
            
        """
        n = len(s_qubits)
        lst = list(itertools.product([0, 1], repeat=n))
        all_bitstrings = []
        for element in lst:
            string = ""
            for bit in element:
                string+=str(bit)
            all_bitstrings.append(string)
        return all_bitstrings
