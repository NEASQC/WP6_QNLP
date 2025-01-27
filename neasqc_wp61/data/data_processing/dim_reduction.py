"""
DimReduction
============
Module containing the base class for performing dimensionality reduction.

"""
from abc import ABC, abstractmethod

import numpy as np 
import pandas as pd
import sklearn.decomposition as skd
import sklearn.manifold as skm
import umap
import joblib

class DimReduction(ABC):
    """
    Base class for dimensionality reduction of 
    vectors representing sentences. 
    """
    def __init__(
        self, dataset : pd.DataFrame, dim_out : int
    )-> None:
        """
        Initialise the dimensionality reduction class.
        
        Parameters
        ----------
        dataset : pd.DataFrame
            Pandas dataframe where each row corresponds
            to a sentence. It must contain one column named
            'sentence_vector', with the vector representation 
            of each sentence.
        dim_out : int 
            Desired output dimension of the vectors.
        """
        self.dataset = dataset
        try:
            self.sentence_vectors = dataset['sentence_embedding'].to_list()
        except KeyError:
            raise ValueError('Sentence vector not present in the dataset.')
        self.dim_out = dim_out

    @abstractmethod
    def reduce_dimension(self)-> None:
        """
        Fit the dataset to output vectors with the desired dimension.
        """
    
    @abstractmethod
    def apply_pretrained_reduction_model(self, new_dataset: pd.DataFrame)-> pd.DataFrame:
        """
        Fit the dataset to output vectors with the desired dimension using a pre-trained model.
        """

    def save_dataset(
            self, filename :str,
            dataset_path : str)-> None:
        """
        Save the reduced dataset in a given path.

        Parameters
        ----------
        filename : str
            Name of the file to save to.
        dataset_path : str
            Path where to store the dataset.
        """
        filepath =f"{dataset_path}{filename}.tsv"
        self.dataset.to_csv(
            filepath, sep='\t', index = False
        )


class PCA(DimReduction):
    """
    Class for principal component analysis implementation.
    """
    def __init__(
        self, dataset : pd.DataFrame, dim_out : int, **kwargs
    )-> None:
        """
        Initialise the PCA dimensionality reduction class.

        Parameters
        ----------
        dataset : pd.DataFrame
            Pandas dataframe where each row corresponds
            to a sentence. It must contain one column named
            'sentence_vector', with the vector representation 
            of each sentence.
        dim_out : int 
            Desired output dimension of the vectors.
        **kwargs 
            Arguments passed to the sklearn.decomposition.PCA object.
            They can be found in 
            https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html.
        """
        super().__init__(
            dataset = dataset, dim_out = dim_out
        )
        self.pca_sk = skd.PCA(n_components=self.dim_out, **kwargs)
    
    def reduce_dimension(self)-> None:
        """
        Fit the vectorised sentences to obtain the reduced dimension
        sentence vectors.
        """
        sentence_vectors_reduced = self.pca_sk.fit_transform(
            self.sentence_vectors)
        self.dataset[
            'reduced_sentence_embedding'] = sentence_vectors_reduced.tolist()

    def apply_pretrained_reduction_model(self, new_dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the pre-trained dimensionality reduction model to a new dataset.

        Parameters
        ----------
        new_dataset : pd.DataFrame
            New dataset containing a 'sentence_embedding' column.

        Returns
        -------
        pd.DataFrame
            The new dataset with the reduced vectors added.
        """
        # Check if 'sentence_embedding' column exists
        if 'sentence_embedding' not in new_dataset.columns:
            raise ValueError("The new dataset must contain a 'sentence_embedding' column.")
        
        # Extract sentence embeddings
        sentence_vectors = new_dataset['sentence_embedding'].tolist()
        
        # Transform the vectors using the loaded PCA model
        reduced_vectors = self.pca_sk.transform(sentence_vectors)
        
        # Add the reduced vectors to the new dataset
        new_dataset['reduced_sentence_embedding'] = reduced_vectors.tolist()
        
        return new_dataset


class ICA(DimReduction):
    """
    Class for the independent component analysis implementation.
    """
    def __init__(
        self, dataset : pd.DataFrame, dim_out : int, **kwargs
    )-> None:
        """
        Initialise the ICA dimensionality reduction class.

        Parameters
        ----------
        dataset : pd.DataFrame
            Pandas dataframe where each row corresponds
            to a sentence. It must contain one column named
            'sentence_vector', with the vector representation 
            of each sentence.
        dim_out : int 
            Desired output dimension of the vectors.
        **kwargs 
            Arguments passed to the sklearn.decomposition.FastICA object.
            They can be found in 
            https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html.
        """
        super().__init__(
            dataset = dataset, dim_out = dim_out
        )
        self.ica_sk = skd.FastICA(
            n_components = self.dim_out, **kwargs
        )

    def reduce_dimension(self)-> None:
        """
        Fit the vectorised sentences to obtain the reduced dimension
        sentence vectors.
        """
        sentence_vectors_reduced = self.ica_sk.fit_transform(
            self.sentence_vectors
        )
        self.dataset[
            'reduced_sentence_embedding'] = sentence_vectors_reduced.tolist()
        
    def apply_pretrained_reduction_model(
        new_dataset: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply the pre-trained dimensionality reduction model to a new dataset.

        Parameters
        ----------
        new_dataset : pd.DataFrame
            New dataset containing a 'sentence_embedding' column.

        Returns
        -------
        pd.DataFrame
            The new dataset with the reduced vectors added.
        """
        # Check if 'sentence_embedding' column exists
        if 'sentence_embedding' not in new_dataset.columns:
            raise ValueError("The new dataset must contain a 'sentence_embedding' column.")
        
        # Extract sentence embeddings
        sentence_vectors = new_dataset['sentence_embedding'].tolist()
        
        # Transform the vectors using the loaded PCA model
        reduced_vectors = self.ica_sk.transform(sentence_vectors)
        
        # Add the reduced vectors to the new dataset
        new_dataset['reduced_sentence_embedding'] = reduced_vectors.tolist()
        
        return new_dataset

class TSVD(DimReduction):
    """
    Class for truncated SVD dimensionality reduction.
    """
    def __init__(
        self, dataset : pd.DataFrame, dim_out : int, **kwargs
    )-> None:
        """
        Initialise the TSVD dimensionality reduction class.

        Parameters
        ----------
        dataset : pd.DataFrame
            Pandas dataframe where each row corresponds
            to a sentence. It must contain one column named
            'sentence_vector', with the vector representation 
            of each sentence.
        dim_out : int 
            Desired output dimension of the vectors.
        **kwargs 
            Arguments passed to the sklearn.decomposition.TruncatedSVD object.
            They can be found in 
            https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html.
        """
        super().__init__(
            dataset = dataset, dim_out = dim_out
        )
        self.tsvd_sk = skd.TruncatedSVD(
            n_components = self.dim_out, **kwargs
        )

    def reduce_dimension(self)-> None:
        """
        Fit the vectorised sentences to obtain the reduced dimension
        sentence vectors.
        """
        sentence_vectors_reduced = self.tsvd_sk.fit_transform(
            self.sentence_vectors
        )
        self.dataset[
            'reduced_sentence_embedding'] = sentence_vectors_reduced.tolist()
        
    def apply_pretrained_reduction_model(
        new_dataset: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply the pre-trained dimensionality reduction model to a new dataset.

        Parameters
        ----------
        new_dataset : pd.DataFrame
            New dataset containing a 'sentence_embedding' column.

        Returns
        -------
        pd.DataFrame
            The new dataset with the reduced vectors added.
        """
        # Check if 'sentence_embedding' column exists
        if 'sentence_embedding' not in new_dataset.columns:
            raise ValueError("The new dataset must contain a 'sentence_embedding' column.")
        
        # Extract sentence embeddings
        sentence_vectors = new_dataset['sentence_embedding'].tolist()
        
        # Transform the vectors using the loaded PCA model
        reduced_vectors = self.tsvd_sk.transform(sentence_vectors)
        
        # Add the reduced vectors to the new dataset
        new_dataset['reduced_sentence_embedding'] = reduced_vectors.tolist()
        
        return new_dataset

class UMAP(DimReduction):
    """
    Class for UMAP dimensionality reduction.
    """
    def __init__(
        self, dataset : pd.DataFrame, dim_out : int, **kwargs
    )-> None:
        """
        Initialise the UMAP dimensionality reduction class.

        Parameters
        ----------
        dataset : pd.DataFrame
            Pandas dataframe where each row corresponds
            to a sentence. It must contain one column named
            'sentence_vector', with the vector representation 
            of each sentence.
        dim_out : int 
            Desired output dimension of the vectors.
        **kwargs 
            Arguments passed to the sklearn.decomposition.UMAP object.
            They can be found in 
            https://umap-learn.readthedocs.io/en/latest/parameters.html.
        """
        super().__init__(
            dataset = dataset, dim_out = dim_out
        )
        self.umap_sk = umap.UMAP(
            n_components = self.dim_out, **kwargs
        )

    def apply_pretrained_reduction_model(
        new_dataset: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply the pre-trained dimensionality reduction model to a new dataset.

        Parameters
        ----------
        new_dataset : pd.DataFrame
            New dataset containing a 'sentence_embedding' column.

        Returns
        -------
        pd.DataFrame
            The new dataset with the reduced vectors added.
        """
        # Check if 'sentence_embedding' column exists
        if 'sentence_embedding' not in new_dataset.columns:
            raise ValueError("The new dataset must contain a 'sentence_embedding' column.")
        
        # Extract sentence embeddings
        sentence_vectors = new_dataset['sentence_embedding'].tolist()
        
        # Transform the vectors using the loaded PCA model
        reduced_vectors = self.umap_sk.transform(sentence_vectors)
        
        # Add the reduced vectors to the new dataset
        new_dataset['reduced_sentence_embedding'] = reduced_vectors.tolist()
        
        return new_dataset

    def reduce_dimension(self)-> None:
        """
        Fit the vectorised sentences to obtain the reduced dimension
        sentence vectors.
        """
        sentence_vectors_reduced = self.umap_sk.fit_transform(
            self.sentence_vectors
        )
        self.dataset[
            'reduced_sentence_embedding'] = sentence_vectors_reduced.tolist()
        

class TSNE(DimReduction):
    """
    Class for truncated TSNE dimensionality reduction.
    """
    def __init__(
        self, dataset : pd.DataFrame, dim_out : int, **kwargs
    )-> None:
        """
        Initialise the TSNE dimensionality reduction class.
            
        Parameters
        ----------
        dataset : pd.DataFrame
            Pandas dataframe where each row corresponds
            to a sentence. It must contain one column named
            'sentence_vector', with the vector representation 
            of each sentence.
        dim_out : int 
            Desired output dimension of the vectors.
        **kwargs 
            Arguments passed to the sklearn.decomposition.TSNE object.
            They can be found in 
            https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html.
        """
        super().__init__(
            dataset = dataset, dim_out = dim_out
        )
        self.tsne_sk = skm.TSNE(
            n_components = self.dim_out, **kwargs)
    
    def reduce_dimension(self)-> None:
        """
        Fit the vectorised sentences to obtain the reduced dimension
        sentence vectors.
        """
        sentence_vectors_reduced = self.tsne_sk.fit_transform(
            np.array(self.sentence_vectors)
        )
        self.dataset[
            'reduced_sentence_embedding'] = sentence_vectors_reduced.tolist()
        
    def apply_pretrained_reduction_model(
        new_dataset: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply the pre-trained dimensionality reduction model to a new dataset.

        Parameters
        ----------
        new_dataset : pd.DataFrame
            New dataset containing a 'sentence_embedding' column.

        Returns
        -------
        pd.DataFrame
            The new dataset with the reduced vectors added.
        """
        # Check if 'sentence_embedding' column exists
        if 'sentence_embedding' not in new_dataset.columns:
            raise ValueError("The new dataset must contain a 'sentence_embedding' column.")
        
        # Extract sentence embeddings
        sentence_vectors = new_dataset['sentence_embedding'].tolist()
        
        # Transform the vectors using the loaded PCA model
        reduced_vectors = self.tsne_sk.transform(sentence_vectors)
        
        # Add the reduced vectors to the new dataset
        new_dataset['reduced_sentence_embedding'] = reduced_vectors.tolist()
        
        return new_dataset
