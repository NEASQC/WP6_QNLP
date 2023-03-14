from discopy import grammar

from lambeq import BobcatParser
from lambeq.ansatz.circuit import IQPAnsatz
from lambeq.core.types import AtomicType

from sympy import default_sort_key

from pytket import Circuit, Qubit, Bit
from pytket.extensions.qiskit import AerBackend
from pytket.utils import probs_from_counts

import itertools
import copy

parser = BobcatParser()



class parameterised_quantum_circuit():
    """Generates a parameterised quantum circuit for a given input sentence.

    ........

    Attributes
    ----------
    
    sentence: str
        Inputted sentence 
    ansatz: IQPansatz
        IQPansatz for the parameterised quantum circuit.
    tk_circuit: pytket circuit
        pytket representation of the circuit.
    parameters: dict
        Parmameters in the parameterised quantum circuit
    word_number_of_parameters: list[ints]
        The number of parameters required for each word in the sentence respectively.

    """
    
    def __init__(self, sentence: str):
        """Initialises parameterised quantum circuit.

        Takes in a sentence and creates a parameterised quantum circuit that represents it's grammatical structure, using Lambeq library.
        

        Parameters
        ----------
        sentence : str
            Input sentence
            

        """
        
        #Defining the sentence
        self.sentence = sentence
        
        #Defining the sentence structure
        #self.sentence_type = sentence_type
        
        #Defining the ansatz
        n = AtomicType.NOUN
        s = AtomicType.SENTENCE
        p = AtomicType.PREPOSITIONAL_PHRASE
        circuit_layers=1
        self.ansatz = IQPAnsatz({n: 1, s: 1, p: 1}, n_layers=circuit_layers)
        
        #Converting circuit to tket
        self.tk_circuit = self.create_circuit()
        
        #Finding the parameters of the circuit
        self.parameters = sorted(self.tk_circuit.free_symbols(), key=default_sort_key)
        #print("\n parameter names = ", self.parameters, "\n")
        
        #Number of parameters per word
        self.word_number_of_parameters = self.GetNParamsWord()
        
    def run_circuit(self,parameters):
        """Runs the parametrised quantum circuit and performs post processing to achieve classification output.

        Given a set of parameters for the parameterised quantum circuit, this function will run the circuit and return a binary classification.:

        Parameters
        ----------
        
        parameters: dict
            parameters required for the pqc.
        
        Returns
        -------
        classification: list[floats]
            two-element list [x,1-x] that represents a binary classification.
            
        """
        #Update parameters(Input torch tensor of parameters)
        parameters_dict = self.update_parameters(parameters)
        
        #Measure s qubits
        #circuit_temp = self.tk_circuit
        #circuit = copy.deepcopy(circuit_temp)
        circuit = self.create_circuit()
        
        s_qubits = self.Measure_s_qubits(circuit)
        
        #Input parameters
        circuit.symbol_substitution(parameters_dict)
        
        #Run circuit and
        #Obtain outputs, apply post-selction and obtain classsification output
        backend = AerBackend()

        handle = backend.process_circuits(backend.get_compiled_circuits([circuit]), n_shots=2000)
        counts = backend.get_result(handle[0]).get_counts()
        result_dict = self.get_norm_circuit_output(counts, s_qubits)
        all_bitstrings = self.calculate_bitstring(s_qubits)
        for bitstring in all_bitstrings:
            if bitstring not in result_dict.keys():
                result_dict[bitstring] = 0
        classification = list(result_dict.values())
        return classification
    
    def create_circuit(self):
        """Given a sentence, this creates a quantum circuit that represents the sentence grammatical structure.

        Uses the Lambeq library to generate a parameterised quantum circuit that represents the sentence grammtical structure:

        Parameters
        ----------
        
        
        Returns
        -------
        tk_circuit: pytket circuit
            A tket circuit representing the grammar of the sentence.
            
        """
        #Sentence to diagram
        diagram = parser.sentence2diagram(self.sentence)
        
        #Discopy circuit
        discopy_circuit = self.ansatz(diagram)
        
        #Converting circuit to tket
        tk_circuit = discopy_circuit.to_tk()
        
        return tk_circuit       
        
    def update_parameters(self,parameters):
        """Updates the parameters of the quantum circuit.

        Joins the parameter names in the pqc with the inputted parameter float values in a dictionary.:

        Parameters
        ----------
        
        parameters: list[floats]
            parameters required for the pqc.
        
        Returns
        -------
        parameters_dict: dict
            dictionary mapping parameter values to parameter names.
            
        """
        parameter_names = self.tk_circuit.free_symbols()
        parameters_dict = {p: q for p, q in zip(parameter_names, parameters)}
        return parameters_dict
        
    def GetNParamsWord(self):
        """Finds the number of quantum parameters corrseponding to each word in the sentence.

       
        Returns
        -------
        params_per_word: list
            Number of parameters for each word.

        """
        w=0
        params_per_word=[0]
        for i, param in enumerate(self.parameters):
            """
            word = param.name.split("__")[0]
            if i==0:
                prev_word=word
            if word==prev_word:
                    params_per_word[w]+=1
            else:
                w+=1
                params_per_word.append(1)
                prev_word=word
            """
           
            word_counter = param.name[-1]
            if i==0:
                params_per_word[w]+=1
            else:
                if word_counter=="0":
                    w+=1
                    params_per_word.append(1)
                else:
                    params_per_word[w]+=1
           
        return params_per_word
    
    def Measure_s_qubits(self, Circuit):
        """Measures sentence qubits.

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
            post_selected = self.satisfy_post_selection(self.tk_circuit.post_selection, bits)
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
        