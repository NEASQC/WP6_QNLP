
import pandas as pd 
import numpy as np 
import random as rd 

class PreAlphaDataset:
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
        self.dataset = pd.read_csv(
        directory, sep='\t+',
        header=None, names=['label', 'sentence', 'structure_tilde'],
        engine='python')
        self.word_dict = self.create_word_dict()
        self.indexes = {
            'train' : [],
            'validation' : [],
            'test' : [],
            'pool' : np.arange(len(self.dataset['sentence'])).tolist()}
        # This dictionary stores the indexes of the sentences going into
        # train, test, or still none of them (pool)
        self.discard_sentences()
        self.number_structures = len(list(
            self.dataset['structure_tilde'].unique()))

    def create_word_dict(self):
        """
        Creates a dictionary with all the words present on the dataset as keys. 
        The values will be dictionaries that will store the indexes of the sentences
        on which each word appears, and if that index is on the train or the test dataset. 
        """
        word_dict = {}
        sentences = self.dataset['sentence']
        for i,s in enumerate(sentences):
            for word in s.lower().split():
                if word not in word_dict.keys():
                    word_dict[word] = {'train' : [], 'validation' : [], 'test' : [], 'pool' : []}
                    word_dict[word]['pool'].append(i)
                else:
                    word_dict[word]['pool'].append(i)
        return word_dict

    
    def discard_sentences(self):
        """
        Discards the sentences containing words that only appear
        only once in the dataset.
        """
        for i,s in enumerate(self.dataset['sentence']):
            for word in s.lower().split():
                if (
                    len(self.word_dict[word]['pool']) == 1):
                    self.word_dict[word]['pool'].remove(i)
                    if i in self.indexes['pool']:
                        self.indexes['pool'].remove(i)
                    # We remove the sentence containing that word
                    # only if it hasnt been removed yet
                        
    def add_remove_indexes(self, idx : int, in_set : str, out_set : str):
        """
        Adds index and word to out_set and removes them from in_set

        Parameters
        ----------
        idx : int
            Index to be added and removed
        in_set : str
            Set from which we are removing the index and word
            (train,test, pool)
        out_set : str 
            Set to which we are adding the index and word
            (train, test, pool)
        """
        sentence = self.dataset['sentence'][idx]
        self.indexes[in_set].remove(idx)
        self.indexes[out_set].append(idx)
        for word in sentence.lower().split():
            self.word_dict[word][in_set].remove(idx)
            self.word_dict[word][out_set].append(idx)

        
    def generate_val_test_indexes(
            self, seed : int , size : int,
            type : str, equal_distribution : bool = False):
        """
        Sets the indexes of the sentences that will belong to the test dataset.
        To do so, it sequentially randomly picks sentences, and adds its indexes to the 
        test dataset only if all the words in the sentence are also present in any other
        sentence of the train dataset. 

        Parameters 
        ----------
        seed : int
            Seed for the sequential random choice of sentences to
            add to the test dataset
        size : int
            Number of sentences to be added to the test/dev dataset 
        type : str 
            Dictates if we are adding sentences to the test or validation dataset
        equal_distribution : bool 
            If True, will seek to place equal number of sentences
            types on the test dataset. (default False)

        """
        count = 0 
        # A counter that will stop after certain number of iterations
        # when the test set can't be generated.
        if equal_distribution == True:
            sentence_types_test = {
                key :[] for key in
                self.dataset['structure_tilde'].unique()
            } 
        # A dictionary that stores how many sentences of each type we have.
        # Will be used for the case equal_distribution = True. 
        rd.seed(seed)
        while len(self.indexes[type]) < size:
            candidate_idx = rd.choice(self.indexes['train'])
            if equal_distribution == False:
                candidate = self.good_candidate(candidate_idx)
            else:
                candidate = self.good_candidate_equal_distribution(
                    candidate_idx, sentence_types_test, size
                )
            if candidate == True:
                self.add_remove_indexes(candidate_idx, 'train', type)
            count += 1
            if count == len(self.dataset['sentence']) * 100:
                print(
            'It was not possible to find a test dataset compatible with pre-alpha')
                break
                # For some test sizes and datasets, it may not be possible 
                # to generate a dataset compatible with pre-alpha, so we break
                # the while loop when a certain number of iterations is reached. 
        

    def good_candidate(self, idx : int) -> bool:
        """
        For a given sentence, decides if it can be added to the test dataset,
        depending on if every words in the sentence are present in at least 
        one sentence of the train dataset.

        Parameters
        ---------
        idx : int 
            Index of the sentence we are analising

        Returns
        -------
        good_candidate : bool
            Dictates if the sentence is added or not to the test datset
        """
        good_candidate = True 
        sentence = self.dataset['sentence'][idx]
        for word in sentence.lower().split():
            if len(self.word_dict[word]['train']) == 1:
                good_candidate = False
            
        return good_candidate

    def good_candidate_equal_distribution(
        self, idx : int, sentence_types_test : dict,
        test_size : int) -> list[bool, dict]:
        """
        For a given sentence, decides if it can be added to the test dataset,
        depending on if every words in the sentence are present in at least 
        one sentence of the train dataset. Also, we restrict the test dataset
        to have equal number of sentences of each type

        Parameters
        ---------
        idx : int 
            Index of the sentence we are analising
        sentence_types_test : dict
            A dictionary storing the sentences_types that have already been
            added to the test dataset
        test_size : int
            Desired test size

        Returns
        -------
        good_candidate : bool
            Dictates if the sentence is added or not to the test datset
        sentence_types_test : dict
            Updated dictionary storing the sentence's type of the test
            dataset
        """
        good_candidate = True 
        sentence = self.dataset['sentence'][idx]
        sentence_type = self.dataset['structure_tilde'][idx]
        for word in sentence.lower().split():
            if len(self.word_dict[word]['train']) == 1:
                good_candidate = False
            else:
                if (
                    len(sentence_types_test[sentence_type]) ==
                    test_size//self.number_structures):
                    good_candidate = False
                else : 
                    sentence_types_test[sentence_type].append(idx)
        return good_candidate, sentence_types_test


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
            self.dataset['structure_tilde'].unique()
        } 
        rd.seed(seed)
        while len(self.indexes['train']) < train_size:
            candidate_idx = rd.choice(self.indexes['pool'])
            if equal_distribution == False:
                self.add_remove_indexes(candidate_idx, 'pool', 'train')
            else:
                sentence_type = self.dataset['structure_tilde'][candidate_idx]
                if (
                    len(sentence_types_test[sentence_type])
                    != train_size//self.number_structures):
                    self.add_remove_indexes(candidate_idx, 'pool', 'train')
                    sentence_types_test[sentence_type].append(candidate_idx)
                    
    def get_train_val_test_datasets(self) -> list[pd.DataFrame, pd.DataFrame]:
        """
        Returns the dataframes with the selected sentences for train
        and test

        Returns
        -------
        dataframes : list[pd.DataFrame, pd.DataFrame]
            List containing pandas dataframes of the selected sentences.
        """
        dataset_train = self.dataset.iloc[
            self.indexes['train']
        ]
        dataset_validation = self.dataset.iloc[
            self.indexes['validation']
        ]
        dataset_test = self.dataset.iloc[
            self.indexes['test']
        ]
        return dataset_train, dataset_validation, dataset_test

    def save_train_val_test_datasets(
            self, path : str, name : str):
        """
        Saves the dataframe with the selected sentences as csv.

        Parameters
        ----------
        path : str
            Path where to save our datasets
        name : str
            Chosen name for the datasets.
        """
        dataset_train = self.dataset.iloc[
            self.indexes['train']
        ]
        dataset_validation = self.dataset.iloc[
            self.indexes['validation']
        ]
        dataset_test = self.dataset.iloc[
            self.indexes['test']
        ]
        dataset_train.to_csv(
            path + '/' + name + '_train.tsv', header = False,
            sep = '\t', index = False
        )
        dataset_validation.to_csv(
            path + '/' + name + '_validation.tsv', header = False,
            sep = '\t', index = False
        )
        dataset_test.to_csv(
            path + '/' + name + '_test.tsv', header = False,
            sep = '\t', index = False
        )
