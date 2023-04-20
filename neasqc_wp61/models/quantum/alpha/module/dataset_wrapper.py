import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import torch


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
    
    def __init__(self, filename: str, reduced_word_embedding_dimension: int):
        """Initialises dataset_wrapper.

        Takes in a dataset of sentences and finds the sentences, sentence structures(types), sentence classification labels and the BERT embeddings for each word in each sentence.
        

        Parameters
        ----------
        filename : str
            Input data json file
            

        """
        self.file=filename
        self.reduced_word_embedding_dimension = reduced_word_embedding_dimension
        
        self.sentences, self.sentence_types, self.sentence_labels, self.word_embeddings, self.sentence_lengths = self.data_parser()
         
        
        self.bert_embeddings = self.data_preparation()
        
        
    def data_preparation(self):
        """Performs PCA on word embeddings.
        
        word embedding input looks like: [[word_embedding_1], [word_embedding_2],...]

        Parameters
        ----------
        
        Returns
        -------
        Dataset: list[list[floats]]
            Nested list of BERT embeddings for each word in each sentence.
            
        """
        
        pca = PCA(n_components=self.reduced_word_embedding_dimension).fit(self.word_embeddings)
      
        
        """
        var_ratio = pca.explained_variance_ratio_
        plt.figure(figsize=(10,6))
        plt.plot(np.cumsum(var_ratio))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.title('Cumulative variance plot')
        plt.yticks(np.arange(0,1.1, 0.1))
        plt.xticks(np.arange(0,800, 50))
        plt.grid()
        plt.show()
        
        print("number of components for 90% explained variance: ", sum(np.cumsum(var_ratio)<=0.90))
        print("number of components for 95% explained variance: ", sum(np.cumsum(var_ratio)<=0.95))
        print("number of components for 99% explained variance: ", sum(np.cumsum(var_ratio)<=0.99))
        """
        reduced_word_embeddings = pca.transform(self.word_embeddings)
        ### reduced word embeddings shape = (number of word embeddings, self.reduced_word_embedding_dimension)
        ### we need a shape like sentence lengths
        
        sentence_reshaped_reduced_word_embeddings = []
        counter = 0
        for sentence_length in self.sentence_lengths:
            sentence_embedding_list = []
            for index in range(sentence_length):
                sentence_embedding_list.append(list(reduced_word_embeddings[counter+index]))
            counter+=sentence_length
            sentence_reshaped_reduced_word_embeddings.append(sentence_embedding_list)
        
        return sentence_reshaped_reduced_word_embeddings
    
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
        sentence_lengths = []
        
        
        

        #for sentence, sentence_type, label in zip(dftrain["sentence"], dftrain["structure_tilde"],dftrain["truth_value"]):
        for sentence, sentence_type, label, word_embedding in zip(dftrain["sentence"], dftrain["tree"],dftrain["class"],dftrain["sentence_vectorized"]):
            sentences.append(sentence)
            sentence_types.append(sentence_type)
            sentence_labels.append(label)
            
            sentence_lengths.append(len(word_embedding))
            reshaped_embedding = self.custom_list_reshaper(word_embedding)

            for embedding in reshaped_embedding:
                 word_embeddings.append(embedding)
            
        return sentences, sentence_types, sentence_labels, word_embeddings, sentence_lengths
    
    def custom_list_reshaper(self, array):
        """ reshapes [[[x]],[[y]],...] into [[x],[y],...]"""
        
        new_array = []
        for sub_array in array:
            new_array.append(sub_array[0])
        return new_array

    