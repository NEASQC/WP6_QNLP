import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/quantum/pre-alpha-lambeq/")
from PreAlphaLambeq import *

def main():
    """
    Saves the diagrams of reduced_amazonreview_pre_alpha_train.tsv and 
    reduced_amazonreview_pre_alpha_test.tsv as pickle objects
    """
    train_path = "./../datasets/reduced_amazonreview_pre_alpha_train.tsv"
    test_path = "./../datasets/reduced_amazonreview_pre_alpha_test.tsv"
    sentences_train = PreAlphaLambeq.load_dataset(train_path)[0]
    sentences_test = PreAlphaLambeq.load_dataset(test_path)[0]
    diagrams_train = PreAlphaLambeq.create_diagrams(sentences_train)
    diagrams_test = PreAlphaLambeq.create_diagrams(sentences_test)
    PreAlphaLambeq.save_diagrams(
        diagrams_train, './', 'diagrams_reduced_amazonreview_train'
    )
    PreAlphaLambeq.save_diagrams(
        diagrams_test, './', 'diagrams_reduced_amazonreview_test'
    )

if __name__ == "__main__":
    main()