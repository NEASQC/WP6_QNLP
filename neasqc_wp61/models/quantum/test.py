from dim_reduction import *
import pandas as pd 
import numpy as np 

dataset_test_path = ('./../../data/toy_dataset/' +
'toy_dataset_bert_sentence_embedding_val.csv')
df = pd.read_csv(dataset_test_path, sep='\t')
print(type(df['sentence_embedding'][0]))
"""
PCA_object = PCA(df, dim_out=3)
PCA_object.fit()
print(PCA_object.dataset['reduced_sentence_vector'][0])
"""