from pre_alpha_reduced_dataset import *
import os
current_path = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help = "Dataset path")
    parser.add_argument("-s", "--seed", help = "Seed for generating the dataset")
    parser.add_argument("-tr", "--train", help = "Number of sentences in train")
    parser.add_argument("-va", "--validation", help = "Number of sentences in validation")
    parser.add_argument("-te", "--test", help = "Number of sentences in test")
    parser.add_argument("-n", "--name", help = "Name of the output dataset")
    parser.add_argument("-o", "--output", help = "Path of the output dataset")
    args = parser.parse_args()

    dataset = PreAlphaDataset(args.dataset)
    dataset.generate_train_indexes(
        args.seed, int(args.train) + int(args.validation) + int(args.test), False)
    dataset.generate_val_test_indexes(
        args.seed, int(args.validation), 'validation', False)
    dataset.generate_val_test_indexes(
        args.seed, int(args.test), 'test', False)
    
    dataset.save_train_val_test_datasets(
        args.output, args.name
    )

if __name__ == "__main__":
    main()