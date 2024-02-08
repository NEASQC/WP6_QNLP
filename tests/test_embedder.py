import os
import random
import sys

import numpy as np
import pandas as pd
import unittest

current_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = current_path + "/../neasqc_wp61/data/datasets/"
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/")

from embedder import *


# Define helper functions


def is_sentence_embedding(x: list) -> bool:
    """
    Checks if a list is of type list[float], the expected type for
    a sentence embedding.

    Parameters
    ----------
    x: list
        The list to check.

    Returns
    -------
    bool
        True if x is of type list[list[float]], False otherwise.
    """
    if isinstance(x, list):
        return all(isinstance(element, float) for element in x)
    return False


def is_word_embedding(x: list) -> bool:
    """
    Checks if a list is of type list[list[float]], the expected type for
    a word embedding.

    Parameters
    ----------
    x: list
        The list to check.

    Returns
    -------
    bool
        True if x is of type list[list[float]], False otherwise.
    """

    if isinstance(x, list):
        if all(isinstance(sublist, list) for sublist in x):
            return all(
                isinstance(element, float) for sublist in x for element in sublist
            )
    return False


def str_to_sentence_embedding(x: str) -> list[float]:
    """
    Converts a string into a list ofq floats. Helps read sentence
    embeddings off a TSV file.

    Parameters
    ----------
    x: str
        The str that we wish to convert to a readable sentence
        embedding.

    Returns
    -------
    embedding: list[float]
        The resulting embedding with the desired type.
    """
    x = x.strip("[]")
    x_i = x.split(",")
    embedding = [float(i) for i in x_i]
    return embedding


def str_to_word_embedding(x: str) -> list[list[float]]:
    """
    Converts a string into a list of lists of floats. Helps read word
    embeddings off a TSV file.

    Parameters
    ----------
    x: str
        The str that we wish to convert to a readable word embedding.

    Returns
    -------
    embedding: list[list[[float]]
        The resulting embedding with the desired type.
    """
    x = x.strip("[]")
    x_lists = x.split("], [")
    float_lists = []
    for sublist_str in x_lists:
        sublist_str = sublist_str.strip("[]")
        float_strings = sublist_str.split(",")
        float_sublist = [float(float_str) for float_str in float_strings]
        float_lists.append(float_sublist)

    return float_lists


# Define unit tests


class TestEmbedder(unittest.TestCase):

    def setUp(self):
        """
        Load test dataset to perform our tests on.
        """

        self.dataset = pd.read_csv(
            dataset_path + "amazonreview_train.tsv", delimiter="\t", nrows=5
        )

    def testBertSentenceUncasedDim(self):
        """
        Test that the dimension of the BERT uncased model sentence
        embeddings is 768.
        """
        dim_bert = 768
        sentence_embeddings_df = Bert(self.dataset).compute_embeddings()
        embeddings_list = list(sentence_embeddings_df["sentence_vectorised"])
        for embedding in embeddings_list:
            self.assertEqual(len(embedding), dim_bert)

    def testBertSentenceCasedDim(self):
        """
        Test that the dimension of the BERT cased model sentence
        embeddings is 768.
        """
        dim_bert = 768
        sentence_embeddings_df = Bert(self.dataset, cased=True).compute_embeddings()
        embeddings_list = list(sentence_embeddings_df["sentence_vectorised"])
        for embedding in embeddings_list:
            self.assertEqual(len(embedding), dim_bert)

    def testBertWordUncasedDim(self):
        """
        Test that the dimension of the BERT uncased model word
        embeddings is 768.
        """
        dim_bert = 768
        word_embeddings_df = Bert(
            self.dataset, sentence_embedding=False
        ).compute_embeddings()
        embeddings_list = list(word_embeddings_df["sentence_vectorised"])
        for vector in embeddings_list:
            for embedding in vector:
                self.assertEqual(len(embedding), dim_bert)

    def testBertWordCasedDim(self):
        """
        Test that the dimension of the BERT cased model word embeddings
        is 768.
        """
        dim_bert = 768
        word_embeddings_df = Bert(
            self.dataset, cased=True, sentence_embedding=False
        ).compute_embeddings()
        embeddings_list = list(word_embeddings_df["sentence_vectorised"])
        for vector in embeddings_list:
            for embedding in vector:
                self.assertEqual(len(embedding), dim_bert)

    def testBertSentenceCasedUncased(self):
        """
        Test that the cased and uncased models of Bert generate
        different sentence embeddings for the same sentence.
        """
        uncased_embeddings_df = Bert(self.dataset).compute_embeddings()
        cased_embeddings_df = Bert(self.dataset, cased=True).compute_embeddings()
        uncased_embeddings_list = list(uncased_embeddings_df["sentence_vectorised"])
        cased_embeddings_list = list(cased_embeddings_df["sentence_vectorised"])
        for i in range(len(uncased_embeddings_list)):
            self.assertNotEqual(uncased_embeddings_list[i], cased_embeddings_list[i])

    def testBertWordCasedUncased(self):
        """
        Test that the cased and uncased models of Bert generate
        different word embeddings for the same words.
        """
        uncased_embeddings_df = Bert(
            self.dataset, sentence_embedding=False
        ).compute_embeddings()
        cased_embeddings_df = Bert(
            self.dataset, sentence_embedding=False, cased=True
        ).compute_embeddings()
        uncased_embeddings_list = list(uncased_embeddings_df["sentence_vectorised"])
        cased_embeddings_list = list(cased_embeddings_df["sentence_vectorised"])
        for i in range(len(uncased_embeddings_list)):
            self.assertNotEqual(uncased_embeddings_list[i], cased_embeddings_list[i])

    def testFastTextSentenceDim(self):
        """
        Tests that the generated FastText sentence embeddings have
        dimension = 300, which is the FastText standard output
        dimension. Note that there is no need to try cased and uncased
        scenarios separately as in FastText the pretrained model is
        cased and so the choice of cased or uncased only affects the
        casefolding of the input, and has no impact on the output
        dimension of the embeddings that are generated.
        """
        dim_ft = 300
        sentence_embeddings_df = FastText(self.dataset).compute_embeddings()
        embeddings_list = list(sentence_embeddings_df["sentence_vectorised"])
        for embedding in embeddings_list:
            self.assertEqual(len(embedding), dim_ft)

    def testFastTextWordDim(self):
        """
        Tests that the generated FastText word embeddings have
        dimension = 300, which is the FastText standard output
        dimension. Note that there is no need to try cased and uncased
        scenarios separately as in FastText the pretrained model is
        cased and so the choice of cased or uncased only affects the
        casefolding of the input, and has no impact on the output
        dimension of the embeddings that are generated.
        """
        dim_ft = 300
        word_embeddings_df = FastText(
            self.dataset, sentence_embedding=False
        ).compute_embeddings()
        embeddings_list = list(word_embeddings_df["sentence_vectorised"])
        for vector in embeddings_list:
            for embedding in vector:
                self.assertEqual(len(embedding), dim_ft)

    def testFastTextSentenceDimReduction(self):
        """
        Tests that FastText's tool for reducing the dimension of the
        sentence embeddings works for a random dimension in the range
        [1, 299].
        """
        out_dim = random.randint(1, 299)
        sentence_embeddings_df = FastText(
            self.dataset, dim=out_dim
        ).compute_embeddings()
        embeddings_list = list(sentence_embeddings_df["sentence_vectorised"])
        for embedding in embeddings_list:
            self.assertEqual(len(embedding), out_dim)

    def testFastTextWordDimReduction(self):
        """
        Tests that FastText's tool for reducing the dimension of the
        word embeddings works for a random dimension in the range [1,
        299].
        """
        out_dim = random.randint(1, 299)
        sentence_embeddings_df = FastText(
            self.dataset, sentence_embedding=False, dim=out_dim
        ).compute_embeddings()
        embeddings_list = list(sentence_embeddings_df["sentence_vectorised"])
        for vector in embeddings_list:
            for embedding in vector:
                self.assertEqual(len(embedding), out_dim)

    def testReadSavedSentenceBertEmbeddings(self):
        """
        Tests whether the BERT sentence embeddings in the saved TSV
        dataset can be read as lists without issues.
        """
        path = "./"
        filename = "test_sentence_embeddings"
        embedder = Bert(self.dataset)
        embedder.compute_embeddings()
        embedder.save_embedding_dataset(path, filename)
        saved_sentence_embedding_df = pd.read_csv(
            path + filename + ".tsv", sep="\t", header=0
        )
        sentence_embeddings_list = saved_sentence_embedding_df["sentence_vectorised"]
        for vector in sentence_embeddings_list:
            self.assertTrue(is_sentence_embedding(str_to_sentence_embedding(vector)))

        file_path = path + filename + ".tsv"
        if os.path.exists(file_path):
            os.remove(file_path)

    def testReadSavedWordBertEmbeddings(self):
        """
        Tests whether the BERT word embeddings in the saved TSV
        dataset can be read as lists without issues.
        """
        path = "./"
        filename = "test_word_embeddings"
        embedder = Bert(self.dataset, sentence_embedding=False)
        embedder.compute_embeddings()
        embedder.save_embedding_dataset(path, filename)
        saved_word_embedding_df = pd.read_csv(
            path + filename + ".tsv", sep="\t", header=0
        )
        word_embeddings_list = saved_word_embedding_df["sentence_vectorised"]
        for vector in word_embeddings_list:
            self.assertTrue(is_word_embedding(str_to_word_embedding(vector)))

        file_path = path + filename + ".tsv"
        if os.path.exists(file_path):
            os.remove(file_path)

    def testReadSavedFastTextSentenceEmbeddings(self):
        """
        Tests whether the FastText sentence embeddings in the saved TSV
        datasets can be read as lists without issues.
        """
        path = "./"
        filename = "test_sentence_embeddings"
        embedder = FastText(self.dataset)
        embedder.compute_embeddings()
        embedder.save_embedding_dataset(path, filename)
        saved_sentence_embedding_df = pd.read_csv(
            path + filename + ".tsv", sep="\t", header=0
        )
        sentence_embeddings_list = saved_sentence_embedding_df["sentence_vectorised"]
        for vector in sentence_embeddings_list:
            self.assertTrue(is_sentence_embedding(str_to_sentence_embedding(vector)))

        file_path = path + filename + ".tsv"
        if os.path.exists(file_path):
            os.remove(file_path)

    def testReadSavedFastTextWordEmbeddings(self):
        """
        Tests whether the FastText word embeddings in the saved TSV
        datasets can be read as lists without issues.
        """
        path = "./"
        filename = "test_word_embeddings"
        embedder = FastText(self.dataset, sentence_embedding=False)
        embedder.compute_embeddings()
        embedder.save_embedding_dataset(path, filename)
        saved_word_embedding_df = pd.read_csv(
            path + filename + ".tsv", sep="\t", header=0
        )
        word_embeddings_list = saved_word_embedding_df["sentence_vectorised"]
        for vector in word_embeddings_list:
            self.assertTrue(is_word_embedding(str_to_word_embedding(vector)))

        file_path = path + filename + ".tsv"
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    unittest.main()
