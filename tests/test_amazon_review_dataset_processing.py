import unittest 
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/data/data_processing/")
from amazon_review_dataset_processing import *
import pandas as pd 



class Test_amazon_review_dataset_processing(unittest.TestCase):

    data = load_data()
    filtered_dataset = filter_structures(data)
    filtered_dictionary = dataset_to_dict(filtered_dataset, 0.8)
    reduced_dataset = generate_reduced_dataset(filtered_dataset)


    def test_load_data(self):
        
        self.assertIsInstance(self.data, pd.DataFrame)
        # Verify that our data is a pd.DataFrame
    def test_filter_structures(self):

        selected_structures = [
            's[n[n] (s\\\\n)[((s\\\\n)/(s\\\\n))   (s\\\\n)]]',
            's[n[(n/n)   n] (s\\\\n)[((s\\\\n)/(s\\\\n))   (s\\\\n)]]',
            's[n[n[(n/n)   n]] (s\\\\n)]',
            's[n   (s\\\\n)[((s\\\\n)/n)   n[(n/n)   n]]]',
            's[n   (s\\\\n)[((s\\\\n)/n)   n[(n/n)   n[(n/n)   n]]]]'
            ]
        filtered_structures = list(self.filtered_dataset['structure_tilde'].unique())
        self.assertEqual(selected_structures.sort(), filtered_structures.sort())
        # Verify that the sentence structures on the filtered dataset are the ones
        # that we want 

        number_sentences_structures = [6637, 4189, 3681, 3615, 3033]
        # Number of sentences per structure as appears on Tilde statistics sheet.
        for i,j in enumerate(list(self.filtered_dataset['structure_tilde'].unique())):

            count = list(self.filtered_dataset.loc[
                self.filtered_dataset['structure_tilde'] == j].count())[0]
            self.assertEqual(count, number_sentences_structures[i])
            # Verify that the number of sentences per structure on our filtered dataset 
            # coincides with the number of sentences per structure appearing on Tilde's 
            # statistics sheet. 

    def test_dataset_to_dict(self):

        for i,j in enumerate(["train_data", "test_data"]):
            for item in self.filtered_dictionary[j]:
                df = self.filtered_dataset.loc[
                    self.filtered_dataset["sentence"] == item["sentence"]
                ]
                self.assertEqual(df["structure_tilde"].iloc[0], item["structure_tilde"])
                if df["label"].iloc[0] == 1:
                    self.assertEqual(item["truth_value"] , False)
                if df["label"].iloc[0] == 2:
                    self.assertEqual(item["truth_value"], True)
                # Verify that the values in the dictionary coincide with the ones 
                # in the filtered DataFrame 

    def test_generate_reduced_dataset(self):

        for i in list(self.reduced_dataset['structure_tilde'].unique()):
            count = list(self.reduced_dataset.loc[
                self.reduced_dataset['structure_tilde'] == i].count())[0]
            self.assertEqual(count, 700)
            # Verify that we have 700 sentences of each type.
            


if __name__ == '__main__':

    unittest.main()


