import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/quantum/pre-alpha-lambeq/")
from PreAlphaLambeq import *
import argparse

def main():
    """
    Saves the diagrams of reduced_amazonreview_pre_alpha_train.tsv and 
    reduced_amazonreview_pre_alpha_test.tsv as pickle objects
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", help = "Path of train dataset")
    parser.add_argument("-te", "--test", help = "Path of test dataset")
    parser.add_argument("-o", "--output", help = "Output path")
    parser.add_argument("-n", "--name", help = "Name of the output file")   
    args = parser.parse_args()

    sentences_train = PreAlphaLambeq.load_dataset(args.train)[0]
    sentences_test = PreAlphaLambeq.load_dataset(args.test)[0]
    diagrams_train = PreAlphaLambeq.create_diagrams(sentences_train)
    diagrams_test = PreAlphaLambeq.create_diagrams(sentences_test)
    PreAlphaLambeq.save_diagrams(
        diagrams_train, args.output, args.name + '_train'
    )
    PreAlphaLambeq.save_diagrams(
        diagrams_test, args.output, args.name + '_test'
    )

if __name__ == "__main__":
    main()