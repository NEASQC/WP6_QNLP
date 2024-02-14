import pandas as pd 
import torch

def load_dataset(
    dataset_path : str,
) -> list[str,list[int]]:
    """
    Given a dataset path, loads it as a pd.DataFrame and 
    outputs both sentences and labels as lists.
    The .tsv file containing the dataset must contain three columns : 
    label, sentence and sentence structure, in this order, and it can't 
    have headers. The labels must be integers 0,1...,number_of_classes.
    Parameters
    ----------
    dataset_path : str
        Path of the dataset to be loaded 

    Returns
    -------
    sentences : list[str]
        List with the sentences of the dataset
    labels: list[list[int]]
        List with the one-hot encoding labels of the dataset.
    """
    df = pd.read_csv(
        dataset_path, sep='\t+',
        header=None, names=['label', 'sentence', 'structure_tilde'],
        engine='python'
    )
    sentences = df['sentence'].tolist()
    labels = df['label'].tolist()
    return sentences, labels


def get_labels_one_hot_encoding(
    labels_train : list[int], labels_val : list[int],
    labels_test : list[int]
)->list[list[list[int]], int]:
    """
    Compute the one hot encoding of the labels.

    Parameters
    ----------
    labels_train : list[int]
        Traininig labels as integers.
    labels_val : list[int]
        Validation labels as integers.
    labels_test : list[int]
        Test labels as integers.
    
    Returns
    -------
    list[list[list[int]], int]
        List with the one-hot-encoding labels 
        and the number of classes.
    """
    all_labels = labels_train + labels_val + labels_test
    n_labels = len(set(all_labels))
    all_labels_one_hot = pd.get_dummies(all_labels)
    labels_train_one_hot = all_labels_one_hot[
        :len(labels_train)].values.tolist()
    labels_val_one_hot = all_labels_one_hot[
        len(labels_train):len(labels_train) + len(labels_val)].values.tolist()
    labels_test_one_hot = all_labels_one_hot[
        len(labels_train) + len(labels_val):].values.tolist()
    return (
        [labels_train_one_hot, labels_val_one_hot, labels_test_one_hot],
        n_labels
    )










