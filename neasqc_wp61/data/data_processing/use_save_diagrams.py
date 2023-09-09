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
    parser.add_argument("-val", "--validation", help = "Path of dev dataset")
    parser.add_argument("-o", "--output", help = "Output path") 
    args = parser.parse_args()
    train_dataset_name = os.path.basename(args.train)
    train_dataset_name = os.path.splitext(train_dataset_name)[0]
    test_dataset_name = os.path.basename(args.test)
    test_dataset_name = os.path.splitext(test_dataset_name)[0]
    dev_dataset_name = os.path.basename(args.validation)
    dev_dataset_name = os.path.splitext(dev_dataset_name)[0]

    sentences_train = PreAlphaLambeq.load_dataset(args.train)[0]
    sentences_test = PreAlphaLambeq.load_dataset(args.test)[0]
    sentences_dev = PreAlphaLambeq.load_dataset(args.validation)[0]
    diagrams_train = PreAlphaLambeq.create_diagrams(sentences_train)
    diagrams_test = PreAlphaLambeq.create_diagrams(sentences_test)
    diagrams_dev = PreAlphaLambeq.create_diagrams(sentences_dev)

    PreAlphaLambeq.save_diagrams(
        diagrams_train, args.output, 'diagrams_' + train_dataset_name 
    )
    PreAlphaLambeq.save_diagrams(
        diagrams_test, args.output, 'diagrams_' + test_dataset_name 
    )
    PreAlphaLambeq.save_diagrams(
        diagrams_dev, args.output, 'diagrams_' + dev_dataset_name 
    )

if __name__ == "__main__":
    main()