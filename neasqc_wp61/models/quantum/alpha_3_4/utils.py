import pandas as pd
import torch


def get_labels_one_hot_encoding(
    labels_train: list[int], labels_val: list[int], labels_test: list[int]
) -> list[list[list[int]], int]:
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
        : len(labels_train)
    ].values.tolist()
    labels_val_one_hot = all_labels_one_hot[
        len(labels_train) : len(labels_train) + len(labels_val)
    ].values.tolist()
    labels_test_one_hot = all_labels_one_hot[
        len(labels_train) + len(labels_val) :
    ].values.tolist()
    return (
        [labels_train_one_hot, labels_val_one_hot, labels_test_one_hot],
        n_labels,
    )


class Dataset(torch.utils.data.Dataset):
    """
    Wrapper of torch.utils.data.Dataset class.
    It is needed to create DatasetLoaders in the Alpha3
    model.
    """

    def __init__(self, vectors: list[torch.tensor], labels: list[int]) -> None:
        """
        Initialiser of the class.

        Parameters
        ----------
        vectors : list[torch.tensor]
            Sentence vectors to be loaded in the dataset.
        labels :
            Labels to be loaded in the dataset.
        """
        self.vectors = vectors
        self.labels = labels

    def __len__(self) -> None:
        """
        Define a __len__ method for the class.
        """
        return len(self.vectors)

    def __getitem__(self, idx: int) -> None:
        """
        Define a __getitem__ method for the class.

        Parameters
        ---------
        idx : int
            Index of the item we want to get.
        """
        return (
            torch.tensor(self.vectors[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


import pandas as pd
import torch


def get_labels_one_hot_encoding(
    labels_train: list[int], labels_val: list[int], labels_test: list[int]
) -> list[list[list[int]], int]:
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
        : len(labels_train)
    ].values.tolist()
    labels_val_one_hot = all_labels_one_hot[
        len(labels_train) : len(labels_train) + len(labels_val)
    ].values.tolist()
    labels_test_one_hot = all_labels_one_hot[
        len(labels_train) + len(labels_val) :
    ].values.tolist()
    return (
        [labels_train_one_hot, labels_val_one_hot, labels_test_one_hot],
        n_labels,
    )


class Dataset(torch.utils.data.Dataset):
    """
    Wrapper of torch.utils.data.Dataset class.
    It is needed to create DatasetLoaders in the Alpha3
    model.
    """

    def __init__(self, vectors: list[torch.tensor], labels: list[int]) -> None:
        """
        Initialiser of the class.

        Parameters
        ----------
        vectors : list[torch.tensor]
            Sentence vectors to be loaded in the dataset.
        labels :
            Labels to be loaded in the dataset.
        """
        self.vectors = vectors
        self.labels = labels

    def __len__(self) -> None:
        """
        Define a __len__ method for the class.
        """
        return len(self.vectors)

    def __getitem__(self, idx: int) -> None:
        """
        Define a __getitem__ method for the class.

        Parameters
        ---------
        idx : int
            Index of the item we want to get.
        """
        return (
            torch.tensor(self.vectors[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )
