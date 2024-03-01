import ast
import os
import random
import sys

import numpy as np
import pandas as pd
import unittest

from itertools import product
from typing import Union

current_path = os.path.dirname(os.path.abspath(__file__))
embedder_path = current_path + "/../neasqc_wp61/models/quantum/"

sys.path.append(embedder_path)

import embedder as emb

dataset_path = (
    current_path + "/../neasqc_wp61/data/datasets/amazonreview_train.tsv"
)


# Define helper functions


def random_dataset_sample(dataset: pd.DataFrame, nrows: int) -> pd.DataFrame:
    """
    Returns a randomly selected subset of a dataset.

    Parameters
    ----------
    dataset: pd.DataFrame
        The dataset containing the natural language data we wish to
        sample from in order to perform our tests.
    nrows: int
        The number of rows that we want to randomly select. This will be
        the number of rows of the output DataFrame.

    Returns
    -------
    random_subset:
        A Dataframe consisting of a randomly selected subset of rows
        from the input dataset.
    """
    shuffled_df = dataset.sample(frac=1)
    random_subset = shuffled_df.head(nrows)

    return random_subset


def check_dimension(
    vectors: list, is_sentence_embedding: bool, dim: int
) -> bool:
    """
    Checks that the dimensions of a sentence or word embedding are
    correct.

    Parameters
    ----------
    vectors: list
        The vectorised sentences whose dimensions we want to check. Can
        be a list of float values (sentence embedding) or a list of
        lists of float values (vector of word embeddings).
    is_sentence_embedding: bool
        Specifies whether we are checking for the dimension of a
        sentence embedding vector or a vector of word embeddings.
    dim: int
        The dimension we wish to check matches the size of our vectors.

    Returns
    -------
    bool
        True if all vectors have the correct dimension, False otherwise.
    """
    for vector in vectors:
        if is_sentence_embedding:
            if len(vector) != dim:
                return False
        else:
            for element in vector:
                if len(element) != dim:
                    return False
    return True


def check_sentence_embedding_type(x: Union[list, pd.Series]) -> bool:
    """
    Checks if a list (or a pandas Series casted to a list) is of
    type list[float], the expected type for a sentence embedding.

    Parameters
    ----------
    x: Union[list, pd.Series]
        The list or pandas Series to check.

    Returns
    -------
    bool
        True if x (or x.tolist() if x is a pandas Series) is of type
        list[list[float]], False otherwise.
    """
    x = x.tolist()
    for embedding in x:
        if isinstance(embedding, list):
            return all(isinstance(element, float) for element in embedding)

    return False


def check_word_embedding_type(x: Union[list, pd.Series]) -> bool:
    """
    Checks if a list (or a pandas Series casted to a list) is of type
    list[list[float]], the expected type for a word embedding.

    Parameters
    ----------
    x: Union[list, pd.Series]
        The list or pandas Series to check.

    Returns
    -------
    bool
        True if x (or x.tolist() if x is a pandas Series) is of type
        list[list[float]], False otherwise.
    """
    x = x.tolist()
    for vector in x:
        if isinstance(vector, list):
            if all(isinstance(sublist, list) for sublist in vector):
                return all(
                    isinstance(element, float)
                    for sublist in vector
                    for element in sublist
                )

    return False


def parse_series(s: pd.Series):
    """
    Parses a pandas Series object to a Python list object. Thanks to
    ast.literal_eval, we can read off embedding vectors as lists of
    floats rather than lists of strings, which is how Series are read by
    default from a CSV/TSV file.

    Parameters
    ----------
    s: pd.Series
        The pandas series to be parsed.

    Returns
    -------
    Union[list, list[float], list[list[float]]]
        An empty list if there are errors in parsing, a list of floats
        if the series is comprised of sentence embeddings and a list of
        lists of floats if the series is comprised of word embeddings.
    """
    try:
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        # Return an empty list if there's an error in parsing
        return []


# Define unit tests


class TestEmbedder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Generates a test dataset and a list of all different embedder
        objects in order to perform unit tests on them.
        """
        full_dataset = pd.read_csv(dataset_path, delimiter="\t")

        # Generate a random subset of the full dataset, from which we
        # will generate further random subsets for our tests
        cls.dataset = random_dataset_sample(full_dataset, nrows=100)
        embedding_type_params = [True, False]
        casing_params = [True, False]
        embedder_params = [
            (x, y) for x, y in product(embedding_type_params, casing_params)
        ]
        # Bert embedder objects to be tested
        cls.bert_object_list = [
            emb.Bert(
                dataset=random_dataset_sample(cls.dataset, nrows=5),
                is_sentence_embedding=params[0],
                cased=params[1],
            )
            for params in embedder_params
        ]
        # FastText embedder objects to be tested
        cls.fasttext_object_list = [
            emb.FastText(
                dataset=random_dataset_sample(cls.dataset, nrows=5),
                is_sentence_embedding=params[0],
                cased=params[1],
            )
            for params in embedder_params
        ]
        # FastText embedder objects with reduced dimensions to be tested
        cls.fasttext_dim_object_list = [
            emb.FastText(
                dataset=random_dataset_sample(cls.dataset, nrows=5),
                is_sentence_embedding=params[0],
                cased=params[1],
                dim=random.randint(1, 299),
            )
            for params in embedder_params
        ]
        # Ember embedder objects to be tested
        cls.ember_object_list = [
            emb.Ember(
                dataset=random_dataset_sample(cls.dataset, nrows=5),
                is_sentence_embedding=param,
            )
            for param in embedding_type_params
        ]

    def testDimensionOfBertEmbeddingsIsCorrect(self):
        """
        Test that the dimension of the BERT embeddings is 768.
        """
        dim_bert = 768
        for embedder in self.bert_object_list:
            with self.subTest(embedder=embedder):
                embedder.compute_embeddings()
                vectorised_df = embedder.dataset
                vectors = vectorised_df["sentence_vectorised"]
                self.assertTrue(
                    check_dimension(
                        vectors, embedder.is_sentence_embedding, dim_bert
                    )
                )

    def testDimensionOfFastTextEmbeddingsIsCorrect(self):
        """
        Test that the dimension of the FastText embeddings is 300.
        """
        dim_ft = 300
        for embedder in self.fasttext_object_list:
            with self.subTest(embedder=embedder):
                embedder.compute_embeddings()
                vectorised_df = embedder.dataset
                vectors = vectorised_df["sentence_vectorised"]
                self.assertTrue(
                    check_dimension(
                        vectors, embedder.is_sentence_embedding, dim_ft
                    )
                )

    def testDimensionOfEmberEmbeddingsIsCorrect(self):
        """
        Test that the dimension of the ember-v1 embeddings is 1024.
        """
        dim_ember = 1024
        for embedder in self.ember_object_list:
            with self.subTest(embedder=embedder):
                embedder.compute_embeddings()
                vectorised_df = embedder.dataset
                vectors = vectorised_df["sentence_vectorised"]
                self.assertTrue(
                    check_dimension(
                        vectors, embedder.is_sentence_embedding, dim_ember
                    )
                )

    def testFastTextReducesDimensionOfEmbeddingsCorrectly(self):
        """
        Test that FastText's inbuilt dimensionality reduction tool
        outputs embeddings of the desired target dimension dimension.
        """
        for embedder in self.fasttext_dim_object_list:
            with self.subTest(embedder=embedder):
                embedder.compute_embeddings()
                vectorised_df = embedder.dataset
                vectors = vectorised_df["sentence_vectorised"]
                self.assertTrue(
                    check_dimension(
                        vectors, embedder.is_sentence_embedding, embedder.dim
                    )
                )

    def testSavedEmbeddingsCanBeReadCorrectly(self):
        """
        Tests that our vectorised sentences are saved correctly on the
        TSV files by ensuring that what we are reading from the file is
        indeed a sentence or word embedding vector.
        """
        embedders = (
            self.bert_object_list
            + self.fasttext_object_list
            + self.ember_object_list
        )
        for embedder in embedders:
            with self.subTest(embedder=embedder):
                filename = "test_file"
                file_path = os.path.join(current_path, filename + ".tsv")
                embedder.save_embedding_dataset(
                    path=current_path, filename=filename
                )
                saved_df = pd.read_csv(
                    file_path,
                    sep="\t",
                    header=0,
                )

                if os.path.exists(file_path):
                    os.remove(file_path)

                vectors = saved_df["sentence_vectorised"].apply(parse_series)
                if embedder.is_sentence_embedding:
                    self.assertTrue(check_sentence_embedding_type(vectors))
                else:
                    self.assertTrue(check_word_embedding_type(vectors))


if __name__ == "__main__":
    unittest.main()
