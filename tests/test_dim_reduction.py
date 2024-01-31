import unittest 
import argparse

import pandas as pd 
import numpy as np 
import os 
import sys 

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/")
from dim_reduction import *

class TestDimReduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Generate random dataset for our tests
        """
        number_of_sentences = 10 
        data = {
            'label' : np.random.randint(2, size=number_of_sentences),
            'sentence_embedding' : [
                np.random.rand(
                    args.original_dimension) for _ in range(
                        number_of_sentences)]
        }
        cls.df = pd.DataFrame(data)
        cls.dim_out = args.reduced_dimension

    def test_PCA_gets_desired_dimension(self):
        """
        Test of PCA dimensionality reduction. 
        """
        PCA_object = PCA(self.df, self.dim_out, svd_solver = 'full', tol = 9.0)
        PCA_object.reduce_dimension()
        for value in PCA_object.dataset['reduced_sentence_embedding']:
            self.assertEqual(len(value), self.dim_out)
    
    def test_ICA_gets_desired_dimension(self):
        """
        Test of ICA dimensionality reduction. 
        """
        ICA_object = ICA(self.df, self.dim_out, fun = 'cube')
        ICA_object.reduce_dimension()
        for value in ICA_object.dataset['reduced_sentence_embedding']:
            self.assertEqual(len(value), self.dim_out)

    def test_TSVD_gets_desired_dimension(self):
        """
        Test of TSVD dimensionality reduction. 
        """
        TSVD_object = TSVD(self.df, self.dim_out, algorithm = 'arpack')
        TSVD_object.reduce_dimension()
        for value in TSVD_object.dataset['reduced_sentence_embedding']:
            self.assertEqual(len(value), self.dim_out)

    def test_UMAP_gets_desired_dimension(self):
        """
        Test of UMAP dimensionality reduction. 
        """
        UMAP_object = UMAP(self.df, self.dim_out, n_neighbors=2)
        UMAP_object.reduce_dimension()
        for value in UMAP_object.dataset['reduced_sentence_embedding']:
            self.assertEqual(len(value), self.dim_out)  

    def test_TSNE_gets_desired_dimension(self):
        """
        Test of TSNE dimensionality reduction. 
        """
        TSNE_object = TSNE(self.df, self.dim_out, perplexity=5.0)
        TSNE_object.reduce_dimension()
        for value in TSNE_object.dataset['reduced_sentence_embedding']:
            self.assertEqual(len(value), self.dim_out)  

    def test_save_dataset_correctly(self):
        """
        Test save dataset function
        """
        UMAP_object = UMAP(self.df, self.dim_out, n_neighbors=2)
        UMAP_object.reduce_dimension()
        name_dataset = 'umap_dataset'
        UMAP_object.save_dataset(name_dataset, './')
        df = pd.read_csv(name_dataset + '.tsv', sep = '\t')
        ## Convert str to list of floats 
        def str_to_float(x):
            x = x.strip('[]')
            x_i = x.split(',')
            out = [float(i) for i in x_i]
            return out
        df['reduced_sentence_embedding'] = df[
            'reduced_sentence_embedding'].apply(str_to_float)
        
        for value in df['reduced_sentence_embedding']:
            self.assertEqual(len(value), self.dim_out)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-od", "--original_dimension",
        help = "Dimension of the vectors to reduce",
        default = 768
    )
    parser.add_argument(
        "-rd", "--reduced_dimension",
        help = "Desired output reduced version",
        default = 3
    )
    args, remaining = parser.parse_known_args()
    remaining.insert(0, sys.argv[0])

    unittest.main(argv=remaining)