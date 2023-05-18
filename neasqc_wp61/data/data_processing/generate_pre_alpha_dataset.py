from Pre_alpha_dataset import *
import os
current_path = os.path.dirname(os.path.abspath(__file__))
import numpy as np

directory = current_path + "/../datasets/amazonreview_train_filtered.tsv"
train_datasets = [] 
test_datasets = []
seed = 364
dataset = Pre_alpha_dataset(directory)
dataset.generate_train_indexes(seed, 4000, False)
dataset.generate_test_indexes(seed, 700, False)
dataset.save_train_test_datasets(
    current_path + "/../datasets", "reduced_amazonreview_pre_alpha"
)
