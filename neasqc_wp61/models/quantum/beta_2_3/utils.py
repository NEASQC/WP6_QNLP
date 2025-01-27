import os
import random
import numpy as np
import torch
import json
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import ast
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class EmbeddingDataset(Dataset):
    """Bert Embedding dataset."""

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Convert features and labels to tensors
        feature = torch.tensor(self.X.iloc[idx], dtype=torch.float)
        label = torch.tensor(self.Y.iloc[idx], dtype=torch.long)
        return feature, label

def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_features_labels(data):
    X = [item['sentence_vectorized'][0] for item in data]
    Y = [item['class'] for item in data]
    return X, Y

def encode_labels(Y):
    if isinstance(Y[0], str):
        le = LabelEncoder()
        Y_encoded = le.fit_transform(Y)
        return Y_encoded, le
    else:
        return Y, None

def create_pandas_series(X, Y):
    X_series = pd.Series(X)
    Y_series = pd.Series(Y)
    return X_series, Y_series

def create_dataloader(X, Y, batch_size=32, shuffle=True):
    dataset = EmbeddingDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.mps.manual_seed(seed)


def preprocess_train_test_dataset(dataset_csv_file, test_csv_file):
    """
    Preprocess function for the dataset with sentence embeddings
    """
    df_train = pd.read_csv(train_csv_file)
    df_val = pd.read_csv(val_csv_file)
    df_test = pd.read_csv(test_csv_file)

    df_train["sentence_embedding"] = np.array(
        [
            np.fromstring(embedding.strip(" []"), sep=",")
            for embedding in df_train["sentence_embedding"]
        ]
    ).tolist()
    df_val["sentence_embedding"] = np.array(
        [
            np.fromstring(embedding.strip(" []"), sep=",")
            for embedding in df_val["sentence_embedding"]
        ]
    ).tolist()
    df_test["sentence_embedding"] = np.array(
        [
            np.fromstring(embedding.strip(" []"), sep=",")
            for embedding in df_test["sentence_embedding"]
        ]
    ).tolist()

    # Preprocess labels
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df_train["class"].append(df_val["class"]))

    df_train["class"] = label_encoder.transform(df_train["class"])
    df_val["class"] = label_encoder.transform(df_val["class"])
    df_test["class"] = label_encoder.transform(df_test["class"])

    X_train, y_train, X_val, y_val, X_test, y_test = (
        df_train[["sentence_embedding", "sentence"]],
        df_train["class"],
        df_val[["sentence_embedding", "sentence"]],
        df_val["class"],
        df_test[["sentence_embedding", "sentence"]],
        df_test["class"],
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_train_test_dataset_words(
    train_csv_file,
    val_csv_file,
    test_csv_file,
    reduced_word_embedding_dimension,
):
    """
    Preprocess function for the dataset with word embeddings
    """
    df_train = pd.read_csv(train_csv_file)
    df_val = pd.read_csv(val_csv_file)
    df_test = pd.read_csv(test_csv_file)

    df_train["sentence_vectorized"] = df_train["sentence_vectorized"].apply(
        ast.literal_eval
    )
    df_val["sentence_vectorized"] = df_val["sentence_vectorized"].apply(
        ast.literal_eval
    )
    df_test["sentence_vectorized"] = df_test["sentence_vectorized"].apply(
        ast.literal_eval
    )

    pca = fit_pca(df_train, reduced_word_embedding_dimension)
    df_train["sentence_vectorized"] = df_train["sentence_vectorized"].apply(
        lambda x: pca.transform(x)
    )
    df_val["sentence_vectorized"] = df_val["sentence_vectorized"].apply(
        lambda x: pca.transform(x)
    )
    df_test["sentence_vectorized"] = df_test["sentence_vectorized"].apply(
        lambda x: pca.transform(x)
    )

    # Preprocess labels
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(df_train["class"].append(df_val["class"]))

    df_train["class"] = label_encoder.transform(df_train["class"])
    df_val["class"] = label_encoder.transform(df_val["class"])
    df_test["class"] = label_encoder.transform(df_test["class"])

    X_train, y_train, X_val, y_val, X_test, y_test = (
        df_train[["sentence_vectorized", "sentence"]],
        df_train["class"],
        df_val[["sentence_vectorized", "sentence"]],
        df_val["class"],
        df_test[["sentence_vectorized", "sentence"]],
        df_test["class"],
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_pca(df, n_components):
    """applies pca to the array and returns the reduced array"""
    # Convert each array to NumPy array
    np_array = df["sentence_vectorized"].apply(np.array)

    # Concatenate the NumPy arrays into a single array
    np_array = np.concatenate(np_array)

    pca = PCA(n_components=n_components)
    pca.fit(np_array)

    print("PCA explained variance:", pca.explained_variance_ratio_.sum())

    return pca


def preprocess_train_test_dataset_for_beta_2(
    run_number,
    dataset_csv_file,
    test_csv_file,
):
    """
    Preprocess function for the dataset for the beta 2 model
    """

    df_dataset = pd.read_csv(dataset_csv_file)
    df_test = pd.read_csv(test_csv_file)

    # Set up for 5-fold with 10 runs
    run_number += 1
    if 1 <= run_number <= 2:
        split_idx = 0
    elif 3 <= run_number <= 4:
        split_idx = 1
    elif 5 <= run_number <= 6:
        split_idx = 2
    elif 7 <= run_number <= 8:
        split_idx = 3
    elif 9 <= run_number <= 10:
        split_idx = 4

    print(
        f"---------------------------\nSplit = {split_idx}\n----------------------------"
    )

    df_train = df_dataset[df_dataset["split"] != split_idx]
    df_val = df_dataset[df_dataset["split"] == split_idx]

    train_reduced_embeddings = df_train[
        f"reduced_embedding_{split_idx}"
    ].apply(eval)
    val_reduced_embeddings = df_val[f"reduced_embedding_{split_idx}"].apply(
        eval
    )
    test_reduced_embeddings = df_test["reduced_embedding"].apply(eval)

    enc = preprocessing.OneHotEncoder(handle_unknown="ignore")
    enc.fit(df_train["class"].append(df_val["class"]).values.reshape(-1, 1))

    df_train["class"] = (
        enc.transform(df_train["class"].values.reshape(-1, 1))
        .toarray()
        .tolist()
    )
    df_val["class"] = (
        enc.transform(df_val["class"].values.reshape(-1, 1)).toarray().tolist()
    )
    df_test["class"] = (
        enc.transform(df_test["class"].values.reshape(-1, 1))
        .toarray()
        .tolist()
    )

    X_train, y_train, X_val, y_val, X_test, y_test = (
        train_reduced_embeddings,
        df_train["class"],
        val_reduced_embeddings,
        df_val["class"],
        test_reduced_embeddings,
        df_test["class"],
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_train_test_dataset_for_beta_3(
    run_number,
    dataset_csv_file,
    test_csv_file,
):
    """
    Preprocess function for the dataset for the alpha 3 model
    """

    df_dataset = pd.read_csv(dataset_csv_file)
    df_test = pd.read_csv(test_csv_file)

    # Set up for 5-fold with 10 runs
    run_number += 1
    if 1 <= run_number <= 2:
        split_idx = 0
    elif 3 <= run_number <= 4:
        split_idx = 1
    elif 5 <= run_number <= 6:
        split_idx = 2
    elif 7 <= run_number <= 8:
        split_idx = 3
    elif 9 <= run_number <= 10:
        split_idx = 4

    print(
        f"---------------------------\nSplit = {split_idx}\n----------------------------"
    )

    df_train = df_dataset[df_dataset["split"] != split_idx]
    df_val = df_dataset[df_dataset["split"] == split_idx]

    train_reduced_embeddings = df_train["reduced_embedding"].apply(eval)
    val_reduced_embeddings = df_val["reduced_embedding"].apply(eval)
    test_reduced_embeddings = df_test["reduced_embedding"].apply(eval)

    enc = preprocessing.OneHotEncoder(handle_unknown="ignore")
    enc.fit(df_train["class"].append(df_val["class"]).values.reshape(-1, 1))

    df_train["class"] = (
        enc.transform(df_train["class"].values.reshape(-1, 1))
        .toarray()
        .tolist()
    )
    df_val["class"] = (
        enc.transform(df_val["class"].values.reshape(-1, 1)).toarray().tolist()
    )
    df_test["class"] = (
        enc.transform(df_test["class"].values.reshape(-1, 1))
        .toarray()
        .tolist()
    )

    X_train, y_train, X_val, y_val, X_test, y_test = (
        train_reduced_embeddings,
        df_train["class"],
        val_reduced_embeddings,
        df_val["class"],
        test_reduced_embeddings,
        df_test["class"],
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
