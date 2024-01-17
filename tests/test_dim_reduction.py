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
        np.random.seed(33)
        data = {'labels': np.random.randint(0, 5, 10),
            'sentence_vector': [np.random.rand(16) for _ in range(10)]}
        self.df = pd.DataFrame(data)
        """
        dataset_test_path = ('./../neasqc_wp61/data/toy_dataset/' +
        'toy_dataset_bert_sentence_embedding_val.csv')
        self.df = pd.read_csv(dataset_test_path)
        
    def testPCA(self):
        dim_out = 4
        PCA_object = PCA(self.df, dim_out, svd_solver = 'full', tol = 9.0)
        PCA_object.fit()
        for value in PCA.dataset['reduced_sentence_vector']:
            self.assertEqual(len(value), dim_out)
    
    def testICA(self):
        dim_out = 2
        ICA_object = ICA(self.df, dim_out, fun = 'cube')
        ICA_object.fit()
        for value in ICA.dataset['reduced_sentence_vector']:
            self.assertEqual(len(value), dim_out)

    def testTSVD(self):
        dim_out = 2
        TSVD_object = TSVD(self.df, dim_out, algorithm = 'arpack')
        TSVD_object.fit()
        for value in TSVD.dataset['reduced_sentence_vector']:
            self.assertEqual(len(value), dim_out)

    def testUMAP(self):
        dim_out = 4
        UMAP_object = UMAP(self.df, dim_out, n_neighbors=2)
        UMAP_object.fit()
        for value in UMAP.dataset['reduced_sentence_vector']:
            self.assertEqual(len(value), dim_out)  

    def testTSNE(self):
        dim_out = 3
        TSNE_object = TSNE(self.df, dim_out, perplexity=5.0)
        TSNE_object.fit()
        for value in TSNE.dataset['reduced_sentence_vector']:
            self.assertEqual(len(value), dim_out)  
            
if __name__ == '__main__':
    unittest.main()