"""
Utilities functions of beta models 
"""
import ast

import numpy as np 
import pandas as pd 
from sklearn.decomposition import PCA 


def load_labels(
        train_dataset_path : str,
        test_dataset_path : str, 
        pca_dimension :int
):
    """
    Loads the chosen dataset as pandas dataframe.

    Parameters
    ----------
    train_dataset_path : str
        Path of the train dataset
    test_dataset_path : str
        Path of the test dataset
    pca_dimension : 
        PCA dimension to make the reduction

    Returns
    -------
    labels: list[int]
        List with the labels of the dataset.
        0 False, and 1 True
    """

    df_train = pd.read_csv(train_dataset_path)
    df_test = pd.read_csv(test_dataset_path)


    df_train['sentence_embedding'] = np.array([np.fromstring(embedding.strip(' []'), sep=',') for embedding in df_train['sentence_embedding']]).tolist()
    df_test['sentence_embedding'] = np.array([np.fromstring(embedding.strip(' []'), sep=',') for embedding in df_test['sentence_embedding']]).tolist()

    #We reduce the dimension of the sentence embedding to a 2D vector
    ############################################################
    # Convert the "sentence_embedding" column to a 2D NumPy array
    X_train = np.array(
        [embedding for embedding in df_train['sentence_embedding']])
    X_test = np.array(
        [embedding for embedding in df_test['sentence_embedding']])

    # Initialize and fit the PCA model
    pca = PCA(n_components=pca_dimension)  # Specify the desired number of components

    pca.fit(X_train)
    print('PCA explained variance:', pca.explained_variance_ratio_.sum())

    # Transform the data to the reduced dimension for both training and test sets
    reduced_embeddings_train = pca.transform(X_train)
    reduced_embeddings_test = pca.transform(X_test)

    # Update the DataFrames with the reduced embeddings
    df_train['sentence_embedding'] = list(reduced_embeddings_train)
    df_test['sentence_embedding'] = list(reduced_embeddings_test)

    #Preprocess labels
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df_train['class'])

    df_train['class'] = label_encoder.transform(df_train['class'])
    df_test['class'] = label_encoder.transform(df_test['class'])

    X_train, y_train, X_test, y_test = reduced_embeddings_train, df_train['class'], reduced_embeddings_test, df_test['class']
    
    return X_train, X_test, y_train.values, y_test.values

def load_sentence_vectors_dataset(dataset_path : str) -> list[np.array]:
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
    except KeyError:
        raise ValueError('Sentence vector not present in the dataset')
    formatted_sentence_vectors = []
    for s in sentence_vectors:
        formatted_sentence_vectors.append(ast.literal_eval(s))
    return formatted_sentence_vectors

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