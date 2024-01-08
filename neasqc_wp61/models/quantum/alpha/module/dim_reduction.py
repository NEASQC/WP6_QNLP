
"""
DimReduction
============
Module containing the base class for performing dimensionality reduction

"""
from abc import ABC, abstractmethod
import pandas as pd
import sklearn.decomposition as skd
import sklearn.manifold as skm
import numpy as np 
import umap

class DimReduction(ABC):
    """
    Base class for dimensionality reduction of 
    vectors representing sentences. 
    """
    def __init__(
        self, dataset : pd.DataFrame, dim_out : int
    )-> None:
        """
        Initialises the dimensionality reduction class
        
        Parameters
        ----------
        dataset : pd.DataFrame
            Pandas dataset. Each row of the dataset corresponds
            to a sentence. The dataset must contain one column named
            'sentence_vector', with the vector representation 
            of each sentence.
        dim_out : int 
            Desired output dimension of the vectors
        """

        self.dataset = dataset
        if 'sentence_vector' not in self.dataset:
            raise ValueError('sentence vector not present in the dataset')
        self.sentence_vectors = dataset['sentence_vector'].to_list()
        self.dim_out = dim_out

    @abstractmethod
    def fit(self) -> None:
        """
        Fits the dataset to output vectors with the desired dimension
        """


class PCA(DimReduction):
    """
    Class for principal component analysis implementation
    """
    def __init__(
        self, dataset : pd.DataFrame, dim_out : int, **kwargs
    )-> None:
        """
        Initialises the PCA dimensionality reduction class

            Parameters
            ----------
            dataset : pd.DataFrame
                Pandas dataset. Each row of the dataset corresponds
                to a sentence. The dataset must contain one column named
                'vectorised_sentence', with the vector representation 
                of each sentence.
            dim_out : int 
                Desired output dimension of the vectors
            **kwargs 
                Arguments passed to the sklearn.decomposition.PCA object
                They can be found in 
                https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        """
        super().__init__(
            dataset = dataset, dim_out = dim_out
        )
        self.pca = skd.PCA(n_components=self.dim_out, **kwargs)
    
    def fit(self):
        """
        Fits the vectorised sentences to obtain the reduced dimension
        sentence vectors
        """
        sentence_vectors_reduced = self.pca.fit_transform(
            self.sentence_vectors)
        self.reduced_dataset = self.dataset.copy()
        self.reduced_dataset[
            'sentence_vector'] = sentence_vectors_reduced.tolist()

class ICA(DimReduction):
    """
    Class for the independent component analysis implementation
    """
    def __init__(
        self, dataset : pd.DataFrame, dim_out : int, **kwargs
    )-> None:
        """
        Initialises the ICA dimensionality reduction class
            Parameters
            ----------
            dataset : pd.DataFrame
                Pandas dataset. Each row of the dataset corresponds
                to a sentence. The dataset must contain one column named
                'vectorised_sentence', with the vector representation 
                of each sentence.
            dim_out : int 
                Desired output dimension of the vectors
            **kwargs 
                Arguments passed to the sklearn.decomposition.FastICA object
                They can be found in 
                https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
        """
        super().__init__(
            dataset = dataset, dim_out = dim_out
        )
        self.ica = skd.FastICA(
            n_components = self.dim_out, **kwargs
        )

    def fit(self):
        """
        Fits the vectorised sentences to obtain the reduced dimension
        sentence vectors
        """
        sentence_vectors_reduced = self.ica.fit_transform(
            self.sentence_vectors
        )
        self.reduced_dataset = self.dataset.copy()
        self.reduced_dataset[
            'sentence_vector'] = sentence_vectors_reduced.tolist()
        
class TSVD(DimReduction):
    """
    Class for truncated SVD dimensionality reduction
    """
    def __init__(
        self, dataset : pd.DataFrame, dim_out : int, **kwargs
    )-> None:
        """
        Initialises the tSVD dimensionality reduction class
            Parameters
            ----------
            dataset : pd.DataFrame
                Pandas dataset. Each row of the dataset corresponds
                to a sentence. The dataset must contain one column named
                'vectorised_sentence', with the vector representation 
                of each sentence.
            dim_out : int 
                Desired output dimension of the vectors
            **kwargs 
                Arguments passed to the sklearn.decomposition.TruncatedSVD object
                They can be found in 
                https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
        """
        super().__init__(
            dataset = dataset, dim_out = dim_out
        )
        self.tsvd = skd.TruncatedSVD(
            n_components = self.dim_out, **kwargs
        )

    def fit(self):
        """
        Fits the vectorised sentences to obtain the reduced dimension
        sentence vectors
        """
        sentence_vectors_reduced = self.tsvd.fit_transform(
            self.sentence_vectors
        )
        self.reduced_dataset = self.dataset.copy()
        self.reduced_dataset[
            'sentence_vector'] = sentence_vectors_reduced.tolist()
        
class UMAP(DimReduction):
    """
    Class for UMAP dimensionality reduction
    """
    def __init__(
        self, dataset : pd.DataFrame, dim_out : int, **kwargs
    )-> None:
        """
        Initialises the UMAP dimensionality reduction class
            Parameters
            ----------
            dataset : pd.DataFrame
                Pandas dataset. Each row of the dataset corresponds
                to a sentence. The dataset must contain one column named
                'vectorised_sentence', with the vector representation 
                of each sentence.
            dim_out : int 
                Desired output dimension of the vectors
            **kwargs 
                Arguments passed to the sklearn.decomposition.UMAP object
                They can be found in 
                https://umap-learn.readthedocs.io/en/latest/parameters.html
        """
        super().__init__(
            dataset = dataset, dim_out = dim_out
        )
        self.umap = umap.UMAP(
            n_components = self.dim_out, **kwargs
        )

    def fit(self):
        """
        Fits the vectorised sentences to obtain the reduced dimension
        sentence vectors
        """
        sentence_vectors_reduced = self.umap.fit_transform(
            self.sentence_vectors
        )
        self.reduced_dataset = self.dataset.copy()
        self.reduced_dataset[
            'sentence_vector'] = sentence_vectors_reduced.tolist()
        
class TSNE(DimReduction):
    """
    Class for truncated TSNE dimensionality reduction
    """
    def __init__(
        self, dataset : pd.DataFrame, dim_out : int, **kwargs
    )-> None:
        """
        Initialises the tSNE dimensionality reduction class
            Parameters
            ----------
            dataset : pd.DataFrame
                Pandas dataset. Each row of the dataset corresponds
                to a sentence. The dataset must contain one column named
                'vectorised_sentence', with the vector representation 
                of each sentence.
            dim_out : int 
                Desired output dimension of the vectors
            **kwargs 
                Arguments passed to the sklearn.decomposition.TSNE object
                They can be found in 
                https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        """
        super().__init__(
            dataset = dataset, dim_out = dim_out
        )
        self.tsne = skm.TSNE(
            n_components = self.dim_out, **kwargs)
    
    def fit(self):
        """
        Fits the vectorised sentences to obtain the reduced dimension
        sentence vectors
        """
        sentence_vectors_reduced = self.tsne.fit_transform(
            np.array(self.sentence_vectors)
        )
        self.reduced_dataset = self.dataset.copy()
        self.reduced_dataset[
            'sentence_vector'] = sentence_vectors_reduced.tolist()
        