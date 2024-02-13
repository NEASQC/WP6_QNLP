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
        cls.dim_reduction_funcs = [
            dr.PCA, dr.ICA, dr.TSVD, dr.UMAP, dr.TSNE
        ]
        cls.kwargs_dim_reduction_funcs = [
            {'svd_solver' : 'full', 'tol' : 9.0},
            {'fun' : 'cube'},
            {'algorithm' : 'arpack'},
            {'n_neighbors' : 2},
            {'perplexity' : 5.0}
        ]
        cls.dim_reduction_object_list = []
        for dim_reduction_func, kwargs in zip(
            cls.dim_reduction_funcs, cls.kwargs_dim_reduction_funcs
        ):
            dim_reduction_object = dim_reduction_func(
                    cls.df, cls.dim_out, **kwargs
                )
            dim_reduction_object.reduce_dimension()
            cls.dim_reduction_object_list.append(dim_reduction_object)

    def test_dim_reduction_produces_the_desired_output_dimension(self):
        """
        Test that the available dim reduction techniques output a
        vector with the desired dimension.
        """
        for dim_reduction_object in self.dim_reduction_object_list:
            with self.subTest(
                dim_reduction_object=dim_reduction_object
            ):
                for value in dim_reduction_object.dataset[
                    'reduced_sentence_embedding'
                ]:
                    self.assertEqual(len(value), self.dim_out)

    def test_save_and_load_dataset_preserves_desired_dimension(self):
        """
        Test that save dataset function preserves the desired reduced
        dimension after being saved and loaded.
        """
        ## Convert str to list of floats 
        def str_to_float(x):
            x = x.strip('[]')
            x_i = x.split(',')
            out = [float(i) for i in x_i]
            return out
        for dim_reduction_object in self.dim_reduction_object_list:
            with self.subTest(
                dim_reduction_object=dim_reduction_object
            ):
                name_dataset = 'test_dataset'
                dim_reduction_object.save_dataset(name_dataset, './')
                df = pd.read_csv(name_dataset + '.tsv', sep = '\t')
                df['reduced_sentence_embedding'] = df[
                    'reduced_sentence_embedding'].apply(str_to_float)
                for value in df['reduced_sentence_embedding']:
                    self.assertEqual(len(value), self.dim_out)

    def test_reduced_embedding_is_populated_by_float_values(self):
        """
        Test that the reduced embedding is populated by float values.
        """
        for dim_reduction_object in self.dim_reduction_object_list:
            with self.subTest(
                dim_reduction_object=dim_reduction_object
            ):
                for sentence_vector in dim_reduction_object.dataset[
                    'reduced_sentence_embedding'
                ]:
                    for value in sentence_vector:
                        self.assertIs(type(value), float)

    def test_not_all_elements_are_equal_in_reduced_embedding(self):
        """
        Test that at least two of the values of the reduced embeddings
        are different.
        """
        for dim_reduction_object in self.dim_reduction_object_list:
            with self.subTest(
                dim_reduction_object=dim_reduction_object
            ):
                for sentence_vector in dim_reduction_object.dataset[
                    'reduced_sentence_embedding'
                ]:
                    self.assertNotEqual(
                        len(set(sentence_vector)), 1
                    )


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
