"""
Embedder
============
Module containing the base class for transforming sentences in a dataset into embeddings
"""

from abc import ABC, abstractmethod
import pandas as pd
from pandas.core.api import DataFrame as DataFrame
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import fasttext as ft


class Embedder(ABC):
    """
    Base class for the generation of embeddings for the sentences
    comprising a dataset.
    """

    def __init__(self, dataset: DataFrame, sentence_or_word: bool = True) -> None:
        """
        Initialises the embedder class

        Parameters
        ----------
        dataset : pd.DataFrame
            Pandas dataset. Each row of the dataset correspods to a
            sentence.
        sentence_or_word : bool
            A bool controllig whether we want to produce a sentence
            embedding or word embedding vector for each sentence in the
            dataset. True is for sentence embedding, False is for word
            embedding.
        """

        self.dataset = dataset

    @abstractmethod
    def compute_embedding(self) -> DataFrame:
        """
        Creates embeddings for sentences in a dataset

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
        Saves the dataset containing the embeddings as a CSV file.

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
        embeddings_df = self.compute_embedding()
        embeddings_df.to_csv(path + "/" + filename + ".csv", index=False)


class Bert(Embedder):
    """
    Class for generating BERT embeddings
    """

    def __init__(
        self,
        dataset: DataFrame,
        sentence_or_word: bool = True,
        cased_or_uncased: bool = False,
        **kwargs
    ) -> None:
        """
        Initialises the BERT embedder class

        Parameters
        ----------
        dataset : pd.DataFrame
            Pandas dataset. Each row of the dataset correspods to a
            sentence.
        sentence_or_word : bool
            States whether we want to produce a sentence embedding or
            word embedding vector for each sentence in the dataset. True
            is for sentence embedding, False is for word embedding.
        case_or_uncased: bool
            States whether we want to work with BERT's cased or uncased
            pretrained base model. True is for cased, False is for
            uncased.
        **kwargs
            Additional arguments to be passed to the BertTokenizer
            object. These can be found in
            https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer
        """
        super().__init__(dataset, sentence_or_word)
        self.cased_or_uncased = cased_or_uncased
        self.kwargs = kwargs

    def compute_embedding(self, **kwargs) -> DataFrame:
        """
        Creates BERT embeddings for sentences in a dataset

        Parameters
        ----------
        **kwargs
            Additional arguments to be passed to the BertTokenizer
            object. These can be found in
            https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer

        Returns
        -------
        embeddings_df : pd.DataFrame
            A pandas dataframe with the same structure as the original
            dataset, but with an additional column containing the
            BERT embeddings corresponding to the sentence contained in each
            row.
        """

        embeddings_df = self.dataset
        embeddings_df.columns = ["class", "sentence", "sentence_structure"]

        if self.cased_or_uncased:
            tokenizer = BertTokenizer.from_pretrained("bert-base-cased", **kwargs)
            bert_model = BertModel.from_pretrained("bert-base-cased")

        else:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", **kwargs)
            bert_model = BertModel.from_pretrained("bert-base-uncased")

        for param in bert_model.parameters():
            param.requires_grad = False
        bert_model.eval()

        if self.sentence_or_word:
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

        return embeddings_df

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
        super().save_embedding_dataset(path, filename)


class FastText(Embedder):
    """
    Class for generating FastText embeddings
    """

    def __init__(
        self,
        dataset: DataFrame,
        sentence_or_word: bool = True,
        dim: int = 300,
        cased_or_uncased: bool = False,
    ) -> None:
        """
        Initialises the FastText embedder class

        Parameters
        ----------
        dataset : pd.DataFrame
            Pandas dataset. Each row of the dataset correspods to a
            sentence.
        sentence_or_word : bool
            States whether we want to produce a sentence embedding or word
            embedding vector for each sentence in the dataset. True is for
            sentence embedding, False is for word embedding.
        dim : int
            The output dimension of FastText's word/sentence vectors.
            Default value is 300, but can be set to any value below that.
        case_or_uncased: bool
            Controls whether we casefold our input sentences before
            vectorising them using FastText. FastText's pretrained model is
            case sensitive so casefolding will produce different embeddings.
        """

        super().__init__(dataset, sentence_or_word)
        self.dim = dim
        self.cased_or_uncased = cased_or_uncased

    def compute_embedding(self) -> DataFrame:
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
        ft.util.download_model("en", if_exists="ignore")
        model = load_model("cc.en.300.bin")
        if dim < 300:
            ft.util.reduce_model(model, dim)
        elif dim > 300:
            print("Error: dimension must be <= 300")
            return

        embeddings_df = self.dataset
        embeddings_df.columns = ["class", "sentence", "sentence_structure"]

        if self.sentence_or_word:
            print("Generating sentence embeddings. Please wait...")
            vectorised_sentence_list = []
            for sentence in embeddings_df.sentence.values:
                if not self.cased_or_uncased:
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

        return embeddings_df

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
        super().save_embedding_dataset(path, filename)
