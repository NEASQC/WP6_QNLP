"""
Utilities functions of beta models 
"""
import ast

import numpy as np 
import pandas as pd 
from sklearn.decomposition import PCA 

def load_sentence_vectors_labels_dataset(dataset_path : str) -> list[np.array]:
    """
    Outputs sentence's vectors for a given dataset path

    Parameters
    ----------
    dataset_path : str
        Path of the dataset
    
    Returns
    -------
    formatted_sentence_vectors : list[np.array]
        List with the vectors of each sentence
    """
    df = pd.read_csv(dataset_path)
    try:
        sentence_vectors = df['sentence_embedding'].tolist()
        labels = df['class']
    except KeyError:
        raise ValueError('Sentence vector/labels not present in the dataset')
    formatted_sentence_vectors = []
    for s in sentence_vectors:
        formatted_sentence_vectors.append(ast.literal_eval(s))
    return formatted_sentence_vectors, labels

def reduce_dimension_list_of_vectors(
        X : list[np.array], out_dimension : int) ->list[np.array]:
    """
    Reduced the dimension of the sentences vectors 
    (to be updated with the modular code)
    """
    pca = PCA(n_components=out_dimension)
    return pca.fit_transform(X)

def normalise_list_of_vectors(X : list[np.array]) -> list[np.array]:
    """
    Normalises a list of vectors so that the sum
    of its squared elements  is equal to 1.

    Parameters
    ----------
    X : list[np.array]
        List of vectors to be normalised

    Returns
    -------
    X_normalised : np.array
        List of normalised vectors
    """
    X_normalised = []
    for sample in X:
        X_normalised.append(sample/np.linalg.norm(sample))
    return X_normalised

def pad_list_of_vectors_with_zeros(X : list[np.array]) -> list[np.array]:
    """
    For a given list of vectors, it pads with zeros until 
    so that the length of the vector is a power of 2
    
    Parameters
    ----------
    X : list[np.array]
        List of vectors to be padded with zeros

    Returns
    -------
    X_padded : list[np.array]
        List of padded vectors
    """
    n = len(X[0])
    X_padded = []
    next_power_2 = 2 ** int(np.ceil(np.log2(n)))
    zero_padding = np.zeros(next_power_2 - n)

    for sample in X:
        X_padded.append(np.concatenate((sample, zero_padding)))
    return X_padded

def load_data_pipeline(
    dataset_path, out_dimension
):
    """
    Full pipeline to load the dataset
    """
    formatted_sentence_vectors = load_sentence_vectors_labels_dataset(
        dataset_path
    )[0]
    labels = load_sentence_vectors_labels_dataset(
        dataset_path
    )[1]
    reduced_sentence_vectors = reduce_dimension_list_of_vectors(
        formatted_sentence_vectors, out_dimension
    )
    normalised_sentence_vectors = normalise_list_of_vectors(
        reduced_sentence_vectors
    )
    if out_dimension <  2 * int(np.ceil(np.log2(out_dimension))):
        padded_sentence_vectors = pad_list_of_vectors_with_zeros(
            normalised_sentence_vectors
        )
        return padded_sentence_vectors, labels
    else:
        return normalised_sentence_vectors, labels
