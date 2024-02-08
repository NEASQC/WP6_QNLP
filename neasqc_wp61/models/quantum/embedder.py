"""
Embedder
============
Module containing the base class for transforming sentences in a dataset into embeddings.
"""

from abc import ABC, abstractmethod

import fasttext as ft
import fasttext.util as ftu
import numpy as np
import pandas as pd
import torch

from pandas.core.api import DataFrame as DataFrame
from transformers import BertTokenizer, BertModel


class Embedder(ABC):
    """
    Base class for the generation of embeddings for the sentences
    comprising a dataset.
    """

    def __init__(self, dataset: DataFrame, sentence_embedding: bool = True) -> None:
        """
        Initialises the embedder class.

        Parameters
        ----------
        dataset : pd.DataFrame
            Pandas dataset. Each row of the dataset correspods to a
            sentence.
        sentence_embedding : bool
            A bool controllig whether we want to produce a sentence
            embedding or word embedding vector for each sentence in the
            dataset. True is for sentence embedding, False is for word
            embedding.
        """
        self.dataset = dataset
        self.sentence_embedding = sentence_embedding

    @abstractmethod
    def compute_embeddings(self) -> DataFrame:
        """
        Creates embeddings for sentences in a dataset.

        Returns
        -------
        pd.DataFrame
            A pandas dataframe with the same structure as the original
            dataset, but with an additional column containing the
            embeddings corresponding to the sentence contained in each
            row.
        """
        pass

    @abstractmethod
    def save_embedding_dataset(self, path: str, filename: str) -> None:
        """
        Saves the dataset containing the embeddings as a TSV file.

        Parameters
        ----------
        path : str
            The path to the location where the user wishes to save the
            generated dataset containing the embeddings (it should not
            include a / at the end).
        filename: str
            The name with which the generated dataset containing the
            embeddings will be saved (it should not include .csv or any
            other file extension).
        """
        pass


class Bert(Embedder):
    """
    Class for generating BERT embeddings.
    """

    def __init__(
        self,
        dataset: DataFrame,
        sentence_embedding: bool = True,
        cased: bool = False,
        **kwargs
    ) -> None:
        """
        Initialises the BERT embedder class

        Parameters
        ----------
        dataset : pd.DataFrame
            Pandas dataset. Each row of the dataset correspods to a
            sentence.
        sentence_embedding : bool
            States whether we want to produce a sentence embedding or
            word embedding vector for each sentence in the dataset. True
            is for sentence embedding, False is for word embedding.
        cased: bool
            States whether we want to work with BERT's cased or uncased
            pretrained base model. True is for cased, False is for
            uncased.
        **kwargs
            Additional arguments to be passed to the BertTokenizer
            object. These can be found in
            https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.
        """
        super().__init__(dataset, sentence_embedding)
        self.cased = cased
        self.kwargs = kwargs

    def compute_embeddings(self, **kwargs) -> DataFrame:
        """
        Creates BERT embeddings for sentences in a dataset.

        Parameters
        ----------
        **kwargs
            Additional arguments to be passed to the BertTokenizer
            object. These can be found in
            https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.

        Returns
        -------
        embeddings_df : pd.DataFrame
            A pandas dataframe with the same structure as the original
            dataset, but with an additional column containing the
            BERT embeddings corresponding to the sentence contained in each
            row.
        """
        embeddings_df = self.dataset.copy()
        embeddings_df.columns = ["class", "sentence", "sentence_structure"]

        model = "bert-base-cased" if self.cased else "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model, **kwargs)
        bert_model = BertModel.from_pretrained(model)

        for param in bert_model.parameters():
            param.requires_grad = False
        bert_model.eval()

        if self.sentence_embedding:
            print("Generating sentence embeddings. Please wait...")
            vectorised_sentence_list = []
            for sentence in embeddings_df.sentence.values:
                inputs = tokenizer.encode_plus(sentence).input_ids
                inputs = torch.LongTensor(inputs)

                with torch.no_grad():
                    sentence_embedding = (
                        bert_model(inputs.unsqueeze(0))[1]
                        .squeeze(0)
                        .cpu()
                        .detach()
                        .numpy()
                    )
                vectorised_sentence_list.append(
                    [tensor.item() for tensor in sentence_embedding]
                )

            embeddings_df["sentence_vectorised"] = vectorised_sentence_list

            print("Done!")

        else:
            print("Generating word embeddings. Please wait...")
            vectorised_sentence_list = []
            for sentence in embeddings_df.sentence.values:
                # List storing the word embeddings of each word in a sentence
                sentence_word_embeddings = []
                for word in sentence.split():
                    inputs = tokenizer.encode_plus(word).input_ids
                    inputs = torch.LongTensor(inputs)

                    with torch.no_grad():
                        word_embedding = (
                            bert_model(inputs.unsqueeze(0))[1]
                            .squeeze(0)
                            .cpu()
                            .detach()
                            .numpy()
                        )

                    sentence_word_embeddings.append(word_embedding.tolist())
                vectorised_sentence_list.append(sentence_word_embeddings)

            embeddings_df["sentence_vectorised"] = vectorised_sentence_list
            print("Done!")

        self.dataset = embeddings_df

        return self.dataset

    def save_embedding_dataset(self, path: str, filename: str) -> None:
        """
        Saves the dataset containing the BERT embeddings as a CSV file.

        Parameters
        ----------
        path : str
            The path to the location where the user wishes to save the
            generated dataset containing the embeddings.
        filename: str
            The name with which the generated dataset containing the
            embeddings will be saved (it should not include .csv or any
            other file extension).
        """
        self.dataset.to_csv(path + "/" + filename + ".tsv", index=False, sep="\t")


class FastText(Embedder):
    """
    Class for generating FastText embeddings.
    """

    def __init__(
        self,
        dataset: DataFrame,
        sentence_embedding: bool = True,
        dim: int = 300,
        cased: bool = True,
    ) -> None:
        """
        Initialises the FastText embedder class

        Parameters
        ----------
        dataset : pd.DataFrame
            Pandas dataset. Each row of the dataset correspods to a
            sentence.
        sentence_embedding : bool
            States whether we want to produce a sentence embedding or word
            embedding vector for each sentence in the dataset. True is for
            sentence embedding, False is for word embedding.
        dim : int
            The output dimension of FastText's word/sentence vectors.
            Default value is 300, but can be set to any value below that.
        cased: bool
            Controls whether we casefold our input sentences before
            vectorising them using FastText. FastText's pretrained model is
            case sensitive so casefolding will produce different embeddings.
        """
        super().__init__(dataset, sentence_embedding)
        self.dim = dim
        self.cased = cased

    def compute_embeddings(self) -> DataFrame:
        """
        Creates FastText embeddings for sentences in a dataset

        Returns
        -------
        embeddings_df : pd.DataFrame
            A pandas dataframe with the same structure as the original
            dataset, but with an additional column containing the
            FastText embeddings corresponding to the sentence contained
            in each row.
        """
        ftu.download_model("en", if_exists="ignore")
        model = ft.load_model("cc.en.300.bin")
        if self.dim < 300:
            ftu.reduce_model(model, self.dim)
        elif self.dim > 300:
            print("Error: dimension must be <= 300")
            return

        embeddings_df = self.dataset.copy()
        embeddings_df.columns = ["class", "sentence", "sentence_structure"]

        if self.sentence_embedding:
            print("Generating sentence embeddings. Please wait...")
            vectorised_sentence_list = []
            for sentence in embeddings_df.sentence.values:
                if not self.cased:
                    sentence = sentence.casefold()
                sentence_embedding = model.get_sentence_vector(sentence).tolist()
                vectorised_sentence_list.append(sentence_embedding)

            embeddings_df["sentence_vectorised"] = vectorised_sentence_list
            print("Done!")

        else:
            print("Generating word embeddings. Please wait...")
            vectorised_sentence_list = []
            for sentence in embeddings_df.sentence.values:
                word_embeddings_list = []
                for word in sentence.split():
                    word_embedding = model.get_word_vector(word).tolist()
                    word_embeddings_list.append(word_embedding)
                vectorised_sentence_list.append(word_embeddings_list)

            embeddings_df["sentence_vectorised"] = vectorised_sentence_list
            print("Done!")

        self.dataset = embeddings_df

        return self.dataset

    def save_embedding_dataset(self, path: str, filename: str) -> None:
        """
        Saves the dataset containing the FastText embeddings as a CSV
        file.

        Parameters
        ----------
        path : str
            The path to the location where the user wishes to save the
            generated dataset containing the embeddings.
         filename: str
            The name with which the generated dataset containing the
            embeddings will be saved (it should not include .csv or any
            other file extension).
        """
        self.dataset.to_csv(path + "/" + filename + ".tsv", index=False, sep="\t")
