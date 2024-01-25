import unittest 
import pandas as pd 
import numpy as np 
import os 
import sys 
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/")
from dim_reduction import *

class TestDimReduction(unittest.TestCase):
    
    def setUp(self):
        """
        Generate random dataset for our tests
        """
        n_rows = 10 
        data = {
            'label' : np.random.randint(2, size=n_rows),
            'sentence_embedding' : [
                np.random.rand(768) for _ in range(n_rows)]
        }
        self.df = pd.DataFrame(data)


    def testPCA(self):
        """
        Test of PCA dimensionality reduction. 
        """
        dim_out = 4
        PCA_object = PCA(self.df, dim_out, svd_solver = 'full', tol = 9.0)
        PCA_object.fit()
        for value in PCA_object.dataset['reduced_sentence_embedding']:
            self.assertEqual(len(value), dim_out)
    
    def testICA(self):
        """
        Test of ICA dimensionality reduction. 
        """
        dim_out = 2
        ICA_object = ICA(self.df, dim_out, fun = 'cube')
        ICA_object.fit()
        for value in ICA_object.dataset['reduced_sentence_embedding']:
            self.assertEqual(len(value), dim_out)

    def testTSVD(self):
        """
        Test of TSVD dimensionality reduction. 
        """
        dim_out = 2
        TSVD_object = TSVD(self.df, dim_out, algorithm = 'arpack')
        TSVD_object.fit()
        for value in TSVD_object.dataset['reduced_sentence_embedding']:
            self.assertEqual(len(value), dim_out)

    def testUMAP(self):
        """
        Test of UMAP dimensionality reduction. 
        """
        dim_out = 4
        UMAP_object = UMAP(self.df, dim_out, n_neighbors=2)
        UMAP_object.fit()
        for value in UMAP_object.dataset['reduced_sentence_embedding']:
            self.assertEqual(len(value), dim_out)  

    def testTSNE(self):
        """
        Test of TSNE dimensionality reduction. 
        """
        dim_out = 3
        TSNE_object = TSNE(self.df, dim_out, perplexity=5.0)
        TSNE_object.fit()
        for value in TSNE_object.dataset['reduced_sentence_embedding']:
            self.assertEqual(len(value), dim_out)  

    def test_save_dataset(self):
        """
        Test save dataset function
        """
        dim_out = 8
        UMAP_object = UMAP(self.df, dim_out, n_neighbors=2)
        UMAP_object.fit()
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
            self.assertEqual(len(value), dim_out)



if __name__ == '__main__':
    unittest.main()