from discopy import grammar
from pytket.circuit.display import render_circuit_jupyter

#from lambeq.ccg2discocat import DepCCGParser
from lambeq import BobcatParser
from lambeq.ansatz.circuit import IQPAnsatz
from lambeq.core.types import AtomicType

from sympy import default_sort_key

import time


parser = BobcatParser()

from module.bert_text_preparation import *
from module.get_bert_embeddings import *

class Qsentence:
    """Generates Parametrised Quantum Circuit and Bert Word Embedding for a Sentence.

    Obtains parametrised quantum circuit and word embeddings for the sentence. Also contains attributes pertaining to the parameters in the PQC aswell as the 

    Attributes
    ----------
    label : optional, list, bool, NoneType
        Mapping of True:[1,0] and False:[0,1].
    parameters : tk_circuit.free_symbols()
        The parameters in the quantum circuit that correspond to the words in the sentence.
    embeddings : list
        The sentence embedding.

    """
    
    def __init__(self, sentence_string, n_dim, s_dim, depth=1, label=None):
        """Initialises Qsentence.

        Obtains parametrised quantum circuit and word embeddings for the sentence.
       

        Parameters
        ----------
        sentence_string : str
            Input sentence.
        n_dim : int
            Noun dimension for PQC.
        s_dim : int
            Sentence dimension for PQC.
        depth: int
            Number of layers in the IQPansatz.
        label: optional, bool, list
            label classification of the sentence.

        """
        self.label=label
        self.n = AtomicType.NOUN
        self.s = AtomicType.SENTENCE
        self.p = AtomicType.PREPOSITIONAL_PHRASE
        self.string = sentence_string
        
        tic = time.perf_counter()
        #self.parser = DepCCGParser()
        #self.parser = BobcatParser()
        self.diagram = parser.sentence2diagram(self.string)
        toc = time.perf_counter()
        #print(f"Bobcat Parsed sentence in {toc - tic:0.4f} seconds")
        
        tic = time.perf_counter()
        self.ansatz = IQPAnsatz({self.n: 1, self.s: 1, self.p: 1}, n_layers=depth)
        toc = time.perf_counter()
        #print(f"Generated IQPansatz in {toc - tic:0.4f} seconds")
        
        tic = time.perf_counter()
        self.discopy_circuit = self.ansatz(self.diagram)
        toc = time.perf_counter()
        #print(f"Generated Discopy Circuit in {toc - tic:0.4f} seconds")
        
        tic = time.perf_counter()
        self.tk_circuit = self.discopy_circuit.to_tk()
        toc = time.perf_counter()
        #print(f"Converted circuit to tket in {toc - tic:0.4f} seconds")
        
        tic = time.perf_counter()
        self.parameters = sorted(self.tk_circuit.free_symbols(), key=default_sort_key)
        toc = time.perf_counter()
        #print(f"Sorted Parameters in {toc - tic:0.4f} seconds")
        
        tic = time.perf_counter()
        self.embeddings = self.get_sentence_BERT_embeddings()
        toc = time.perf_counter()
        #print(f"Found BERT embeddings in {toc - tic:0.4f} seconds")
        #print("SENTENCE COMPLETE_____________________________\n \n") 
        
        
    
    def get_sentence_BERT_embeddings(self):
        """Returns word embedding for each sentence.

        Takes a list of sentences and find a Bert embedding for each.:


        Returns
        -------
        Sentences_Embeddings: list
            List consisting of word embeddings for each sentence.

        """
        SentenceList = self.string
        Sentences_Embeddings = []
        if type(SentenceList) == str:
            SentenceList = [SentenceList]
        for sentence in SentenceList:
            tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(sentence, tokenizer)
            list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
            nwords = len(sentence.split(" "))

            word_embeddings = []
            for word in sentence.split(" "):
                word_index = tokenized_text.index(word.lower().replace(".",""))
                word_embeddings.append(list_token_embeddings[word_index])

            Sentences_Embeddings.append(word_embeddings)
        return Sentences_Embeddings
        
        

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

            word = param.name.split("__")[0]
            if i==0:
                prev_word=word
            if word==prev_word:
                    params_per_word[w]+=1
            else:
                w+=1
                params_per_word.append(1)
                prev_word=word
        return params_per_word