import argparse
import unittest 

import numpy as np 
import os 
import pandas as pd 
import sys 

# The two lines below will be removed when converting the library to a package.
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/models/quantum/")
import dim_reduction as dr


class TestDimReduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls)-> None:
        """
        Generate a random dataset for our tests.
        """
        number_of_sentences = args.number_of_sentences
        data = {
            'label' : np.random.randint(2, size=number_of_sentences),
            'sentence_embedding' : [
                np.random.rand(
                    args.original_dimension) for _ in range(
                        number_of_sentences)]
        }
        cls.df = pd.DataFrame(data)
        cls.dim_out = args.reduced_dimension

    def test_PCA_outputs_desired_reduced_dimension(self)-> None:
        """
        Test that PCA outputs the desired reduced dimension. 
        """
        PCA_object = dr.PCA(
            self.df, self.dim_out, svd_solver = 'full', tol = 9.0
        )
        PCA_object.reduce_dimension()
        for value in PCA_object.dataset['reduced_sentence_embedding']:
            self.assertEqual(len(value), self.dim_out)
    
    def test_ICA_outputs_desired_reduced_dimension(self)-> None:
        """
        Test that ICA outputs the desired reduced dimension. 
        """
        ICA_object = dr.ICA(self.df, self.dim_out, fun = 'cube')
        ICA_object.reduce_dimension()
        for value in ICA_object.dataset['reduced_sentence_embedding']:
            self.assertEqual(len(value), self.dim_out)

    def test_TSVD_outputs_desired_reduced_dimension(self):
        """
        Test that TSVD outputs the desired reduced dimension. 
        """
        TSVD_object = dr.TSVD(self.df, self.dim_out, algorithm = 'arpack')
        TSVD_object.reduce_dimension()
        for value in TSVD_object.dataset['reduced_sentence_embedding']:
            self.assertEqual(len(value), self.dim_out)

    def test_UMAP_outputs_desired_reduced_dimension(self):
        """
        Test that UMAP outputs the desired reduced dimension. 
        """
        UMAP_object = dr.UMAP(self.df, self.dim_out, n_neighbors=2)
        UMAP_object.reduce_dimension()
        for value in UMAP_object.dataset['reduced_sentence_embedding']:
            self.assertEqual(len(value), self.dim_out)  

    def test_TSNE_outputs_desired_reduced_dimension(self):
        """
        Test that TSNE outputs the desired reduced dimension. 
        """
        TSNE_object = dr.TSNE(self.df, self.dim_out, perplexity=5.0)
        TSNE_object.reduce_dimension()
        for value in TSNE_object.dataset['reduced_sentence_embedding']:
            self.assertEqual(len(value), self.dim_out)  

    def test_save_and_load_dataset_preserves_desired_dimension(self):
        """
        Test that save dataset function preserves the desired reduced
        dimension after being saved and loaded.
        """
        UMAP_object = dr.UMAP(self.df, self.dim_out, n_neighbors=2)
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
        "-ns", "--number_of_sentences", type = int,
        help = "Number of random sentences to generate for testing.",
        default = 10
    )
    parser.add_argument(
        "-od", "--original_dimension", type = int,
        help = "Dimension of the vectors to be reduced.",
        default = 768
    )
    parser.add_argument(
        "-rd", "--reduced_dimension", type = int,
        help = "Desired output reduced dimension.",
        default = 3
    )
    args, remaining = parser.parse_known_args()
    remaining.insert(0, sys.argv[0])
    unittest.main(argv=remaining)
