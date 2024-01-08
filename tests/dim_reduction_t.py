import unittest 
import pandas as pd 
import numpy as np 
import os 
import sys 
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/alpha/module/")
from dim_reduction import *

class TestDimReduction(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(33)
        data = {'labels': np.random.randint(0, 5, 10),
            'sentence_vector': [np.random.rand(16) for _ in range(10)]}
        self.df = pd.DataFrame(data)

    def testPCA(self):
        dim_out = 4
        PCA_object = PCA(self.df, dim_out, svd_solver = 'full', tol = 9.0)
        PCA_object.fit()
        PCA_reduced_dataset = PCA_object.reduced_dataset
        for value in PCA_reduced_dataset['sentence_vector']:
            self.assertEqual(len(value), dim_out)
    
    def testICA(self):
        dim_out = 2
        ICA_object = ICA(self.df, dim_out, fun = 'cube')
        ICA_object.fit()
        ICA_reduced_dataset = ICA_object.reduced_dataset
        for value in ICA_reduced_dataset['sentence_vector']:
            self.assertEqual(len(value), dim_out)

    def testTSVD(self):
        dim_out = 2
        TSVD_object = TSVD(self.df, dim_out, algorithm = 'arpack')
        TSVD_object.fit()
        TSVD_reduced_dataset = TSVD_object.reduced_dataset
        for value in TSVD_reduced_dataset['sentence_vector']:
            self.assertEqual(len(value), dim_out)

    def testUMAP(self):
        dim_out = 4
        UMAP_object = UMAP(self.df, dim_out, n_neighbors=2)
        UMAP_object.fit()
        UMAP_reduced_dataset = UMAP_object.reduced_dataset
        for value in UMAP_reduced_dataset['sentence_vector']:
            self.assertEqual(len(value), dim_out)  

    def testTSNE(self):
        dim_out = 3
        TSNE_object = TSNE(self.df, dim_out, perplexity=5.0)
        TSNE_object.fit()
        TSNE_reduced_dataset = TSNE_object.reduced_dataset
        for value in TSNE_reduced_dataset['sentence_vector']:
            self.assertEqual(len(value), dim_out)  
            
if __name__ == '__main__':
    unittest.main()