import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import ast
from torch.utils.data import Dataset


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.mps.manual_seed(seed)


def preprocess_train_test_dataset(train_csv_file, val_csv_file, test_csv_file):
    """
    Preprocess function for the dataset with sentence embeddings
    """
    df_train = pd.read_csv(train_csv_file)
    df_val = pd.read_csv(val_csv_file)
    df_test = pd.read_csv(test_csv_file)


    df_train['sentence_embedding'] = np.array([np.fromstring(embedding.strip(' []'), sep=',') for embedding in df_train['sentence_embedding']]).tolist()
    df_val['sentence_embedding'] = np.array([np.fromstring(embedding.strip(' []'), sep=',') for embedding in df_val['sentence_embedding']]).tolist()
    df_test['sentence_embedding'] = np.array([np.fromstring(embedding.strip(' []'), sep=',') for embedding in df_test['sentence_embedding']]).tolist()

    #Preprocess labels
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df_train['class'].append(df_val['class']))

    df_train['class'] = label_encoder.transform(df_train['class'])
    df_val['class'] = label_encoder.transform(df_val['class'])
    df_test['class'] = label_encoder.transform(df_test['class'])

    X_train, y_train, X_val, y_val, X_test, y_test = df_train[['sentence_embedding', 'sentence']], df_train['class'], df_val[['sentence_embedding', 'sentence']], df_val['class'], df_test[['sentence_embedding', 'sentence']], df_test['class']

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_train_test_dataset_words(train_csv_file, val_csv_file, test_csv_file, reduced_word_embedding_dimension):
    """
    Preprocess function for the dataset with word embeddings
    """
    df_train = pd.read_csv(train_csv_file)
    df_val = pd.read_csv(val_csv_file)
    df_test = pd.read_csv(test_csv_file)


    df_train['sentence_vectorized'] = df_train['sentence_vectorized'].apply(ast.literal_eval)
    df_val['sentence_vectorized'] = df_val['sentence_vectorized'].apply(ast.literal_eval)
    df_test['sentence_vectorized'] = df_test['sentence_vectorized'].apply(ast.literal_eval)

    pca = fit_pca(df_train, reduced_word_embedding_dimension)
    df_train['sentence_vectorized'] = df_train['sentence_vectorized'].apply(lambda x: pca.transform(x))
    df_val['sentence_vectorized'] = df_val['sentence_vectorized'].apply(lambda x: pca.transform(x))
    df_test['sentence_vectorized'] = df_test['sentence_vectorized'].apply(lambda x: pca.transform(x))

    #Preprocess labels
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df_train['class'].append(df_val['class']))

    df_train['class'] = label_encoder.transform(df_train['class'])
    df_val['class'] = label_encoder.transform(df_val['class'])
    df_test['class'] = label_encoder.transform(df_test['class'])

    X_train, y_train, X_val, y_val, X_test, y_test = df_train[['sentence_vectorized', 'sentence']], df_train['class'], df_val[['sentence_vectorized', 'sentence']], df_val['class'], df_test[['sentence_vectorized', 'sentence']], df_test['class']

    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_pca(df, n_components):
    """ applies pca to the array and returns the reduced array"""
    # Convert each array to NumPy array
    np_array = df['sentence_vectorized'].apply(np.array)

    # Concatenate the NumPy arrays into a single array
    np_array = np.concatenate(np_array)

    pca = PCA(n_components=n_components)
    pca.fit(np_array)

    print('PCA explained variance:', pca.explained_variance_ratio_.sum())

    return pca


def preprocess_train_test_dataset_for_alpha_3(train_csv_file, val_csv_file, test_csv_file,):
    """
    Preprocess function for the dataset for the alpha 3 model
    """
    df_train = pd.read_csv(train_csv_file)
    df_val = pd.read_csv(val_csv_file)
    df_test = pd.read_csv(test_csv_file)


    df_train['sentence_embedding'] = np.array([np.fromstring(embedding.strip(' []'), sep=',') for embedding in df_train['sentence_embedding']]).tolist()
    df_val['sentence_embedding'] = np.array([np.fromstring(embedding.strip(' []'), sep=',') for embedding in df_val['sentence_embedding']]).tolist()
    df_test['sentence_embedding'] = np.array([np.fromstring(embedding.strip(' []'), sep=',') for embedding in df_test['sentence_embedding']]).tolist()

    enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
    enc.fit(df_train['class'].append(df_val['class']).values.reshape(-1, 1))
    
    df_train['class'] = enc.transform(df_train['class'].values.reshape(-1, 1)).toarray().tolist()
    df_val['class'] = enc.transform(df_val['class'].values.reshape(-1, 1)).toarray().tolist()
    df_test['class'] = enc.transform(df_test['class'].values.reshape(-1, 1)).toarray().tolist()

    X_train, y_train, X_val, y_val, X_test, y_test = df_train['sentence_embedding'], df_train['class'], df_val['sentence_embedding'], df_val['class'], df_test['sentence_embedding'], df_test['class']
    
    return X_train, X_val, X_test, y_train, y_val, y_test


class BertEmbeddingDataset(Dataset):
    """Bert Embedding dataset."""

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return torch.tensor(self.X.iloc[idx]), torch.tensor(self.Y.iloc[idx])
