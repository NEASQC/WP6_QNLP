
import pandas as pd 
import numpy as np 
import random as rd 

class Pre_alpha_dataset:
    """
    Class containing all the methods to create a pre-alha compatible
    dataset,i.e., a dataset where all the words present on the test dataset,
    are also present on the training dataset. This class will allow to filter
    the values of a dataset in order to verify this condition.
    """
    def __init__(self, directory : str):
        """
        Instantiates an object of the class.

        Parameters
        ----------
        directory : str
            The directory wehre our dataset is placed (as csv)
        """
        self.complete_dataset = pd.read_csv(
        directory, sep='\t+',
        header=None, names=['label', 'sentence', 'structure_tilde'],
        engine='python')
        self.word_dict = {}
        self.indexes_train_test = {
            'train' : [],
            'test' : [],
            'all' : np.arange(len(self.complete_dataset['sentence'])).tolist()}
        # At first we have all sentences in the train dataset. 

    def create_word_dict(self):
        """
        Creates a dictionary with all the words present on the dataset as keys. 
        The values will be dictionaries that will store the indexes of the sentences
        on which each word appears, and if that index is on the train or the test dataset. 
        """
        sentences = self.complete_dataset['sentence']
        words_dict = {}
        for i,s in enumerate(sentences):
            for word in s.lower().split():
                if word not in self.word_dict.keys():
                    self.word_dict[word] = {'train' : [], 'test' : [], 'all' : []}
                    self.word_dict[word]['all'].append(i)
                else:
                    self.word_dict[word]['all'].append(i)

    
    def discard_sentences(self):
        """
        Discards the sentences containing words that only appear
        only once in the dataset.
        """
        for i,s in enumerate(self.complete_dataset['sentence']):
            for word in s.lower().split():
                if (
                    len(self.word_dict[word]['all']) == 1):
                    if i in self.indexes_train_test['all']:
                        self.indexes_train_test['all'].remove(i)
                    if i in self.word_dict[word]['all']:
                        self.word_dict[word]['all'].remove(i)
        print(len(self.indexes_train_test['all']))


                
    def generate_test_indexes(
            self, seed : int , test_size : int,
            equal_distribution : bool = False):
        """
        Sets the indexes of the sentences that will belong to the test dataset.
        To do so, it sequentially randomly picks sentences, and adds its indexes to the 
        test dataset only if all the words in the sentence are also present in other
        sentences of the train dataset. 

        Parameters 
        ----------
        seed : int
            Seed for the sequential random choice of sentences to
            add to the test dataset
        test_size : int
            Number of sentences to be added to the test dataset 
        equal_distribution : bool 
            If True, will seek to place equal number of sentences
            types on the test dataset. (default False)

        """
        count = 0 
        sentence_types_test = {
            key :[] for key in
            self.complete_dataset['structure_tilde'].unique()
        } 
        # A dictionary that stores how many sentences of each type we have.
        # It is useful when equal_distribution = True
        rd.seed(seed)
        while len(self.indexes_train_test['test']) < test_size:
            good_candidate = True
            candidate_idx = rd.choice(self.indexes_train_test['train'])
            sentence = self.complete_dataset['sentence'][candidate_idx]
            sentence_type = self.complete_dataset['structure_tilde'][candidate_idx]
            for word in sentence.lower().split():
                if len(self.word_dict[word]['train']) == 1:
                    good_candidate = False
            if equal_distribution == True and good_candidate == True: 
                if len(sentence_types_test[sentence_type]) == test_size//5:
                    good_candidate = False
                else : 
                    sentence_types_test[sentence_type].append(candidate_idx)

            if good_candidate == True:
                self.indexes_train_test['train'].remove(candidate_idx)
                self.indexes_train_test['test'].append(candidate_idx)
                for word in sentence.lower().split():
                    self.word_dict[word]['train'].remove(candidate_idx)
                    self.word_dict[word]['test'].append(candidate_idx)
            count += 1
            if count == len(self.complete_dataset['sentence']) * 100:
                print(
            'It was not possible to find a test dataset compatible with pre-alpha')
                break
                # For some test sizes and datasets, it may not be possible 
                # to generate a dataset compatible with pre-alpha, so we break
                # the while loop when a certain number of iterations is reached. 

        #self.indexes_train_test['train'] = rd.sample(
        #    self.indexes_train_test['train'], train_size)

    def generate_train_indexes(
            self, seed : int, train_size : int, 
            equal_distribution : bool = False):
        """
        Sets the indexes of the sentences that will belong to 
        the training dataset. 
        
        Parameters
        ----------
        seed : int 
            Seed used to generate samples from our dataset. 
        train_size : int 
            Number of samples to have in the trainig dataset. 
        equal_distribution : bool
            If True, will seek to place equal number of sentences
            types on the test dataset. (default False) 

        """
        sentence_types_test = {
            key :[] for key in
            self.complete_dataset['structure_tilde'].unique()
        } 
        rd.seed(seed)
        if equal_distribution == False:
            while len(self.indexes_train_test['train']) < train_size:
                candidate_idx = rd.choice(self.indexes_train_test['all'])
                sentence = self.complete_dataset['sentence'][candidate_idx]
                self.indexes_train_test['all'].remove(candidate_idx)
                self.indexes_train_test['train'].append(candidate_idx)
                for word in sentence.lower().split():
                    self.word_dict[word]['all'].remove(candidate_idx)
                    self.word_dict[word]['train'].append(candidate_idx)

        else:
            pass

            



        
    def save_train_test_datasets(
            self, path : str, name : str
    ):
        dataset_train = self.complete_dataset.iloc[
            self.indexes_train_test['train']
        ]
        dataset_test = self.complete_dataset.iloc[
            self.indexes_train_test['test']
        ]
        dataset_train.to_csv(
            path + '/' + name + '_train.tsv', header = False,
            sep = '\t', index = False
        )
        dataset_test.to_csv(
            path + '/' + name + '_test.tsv', header = False,
            sep = '\t', index = False
        )
