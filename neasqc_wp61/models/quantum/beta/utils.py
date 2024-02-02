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
    return reduced_sentence_vectors, labels
