from transformers import BertTokenizer
from transformers import BertModel
import torch


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True,
                                  )

from discopy import grammar
from pytket.circuit.display import render_circuit_jupyter

from lambeq import BobcatParser
from lambeq.ansatz.circuit import IQPAnsatz
from lambeq.core.types import AtomicType

from sympy import default_sort_key

import time

import re

import json
import pandas as pd

parser = BobcatParser()

class dataset_wrapper():
    """Generates BERT embeddings for each sentence. Also hold sentences, sentence_types, sentence_labels

    ........

    Attributes
    ----------
    
    file: str
        Inputted file name 
    sentences: list
        list of sentences
    sentence_types: list
        list of sentence types(grammatical structures)
    sentence_labels: list
        list of sentence classification labels
    bert_embeddings: list[list[floats]]
        The BERT embeddings for each word in each sentence.

    """
    
    def __init__(self, filename: str):
        """Initialises dataset_wrapper.

        Takes in a dataset of sentences and finds the sentences, sentence structures(types), sentence classification labels and the BERT embeddings for each word in each sentence.
        

        Parameters
        ----------
        filename : str
            Input data json file
            

        """
        self.file=filename
        
        self.sentences, self.sentence_types, self.sentence_labels, self.bert_embeddings = self.data_parser()
        
        #self.bert_embeddings = self.data_preparation()
        
        
    def data_preparation(self):
        """Transforms sentences into Qsentences.

        Takes sentence train and test data along with their repective true or false labels and transforms each sentence into a so-called Qsentence.:

        Parameters
        ----------
        
        Returns
        -------
        Dataset: list[list[floats]]
            Nested list of BERT embeddings for each word in each sentence.
            
        """
    
        Dataset = []  
        for sentence in self.sentences:
            print("Sentence = ", sentence)
            #Dataset.append(self.get_sentence_BERT_embeddings(sentence_string=sentence))
            print("BERT Embedding Obtained")
        return Dataset
    
    def data_parser(self):
        """Parses the elements of the dataset and returns them as lists.

        Takes the dataset and returns a tuple of three lists, the sentences, sentence_types and sentence_labels.:

        Parameters
        ----------

        Returns
        -------
        sentences, sentence_types, sentence_labels: tuple(list,list,list)
           
        """
        with open(self.file) as f:
            data = json.load(f)
        dftrain = pd.DataFrame(data)
        dftrain["class"]= dftrain["class"].map({"2": [1,0], "1": [0,1]})
        #dftest = pd.DataFrame(data['test_data'])
        #dftest["truth_value"]= dftest["class"].map({2: [1,0], 1: [0,1]})
        
        sentences = []
        sentence_types = []
        sentence_labels = []
        word_embeddings = []

        #for sentence, sentence_type, label in zip(dftrain["sentence"], dftrain["structure_tilde"],dftrain["truth_value"]):
        for sentence, sentence_type, label, word_embedding in zip(dftrain["sentence"], dftrain["tree"],dftrain["class"],dftrain["sentence_vectorized"]):
            if sentence == 'Borring as hell':
                sentence = 'Boring as hell'
            sentences.append(sentence)
            sentence_types.append(sentence_type)
            sentence_labels.append(label)
            word_embeddings.append(word_embedding)
        return sentences, sentence_types, sentence_labels, word_embeddings

    