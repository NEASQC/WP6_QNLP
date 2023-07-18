from Pre_alpha_dataset import *
import os
current_path = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help = "Dataset path")
    parser.add_argument("-s", "--seed", help = "Seed for generating the dataset")
    parser.add_argument("-tr", "--train", help = "Number of sentences in train")
    parser.add_argument("-te", "--test", help = "Number of sentences in test")
    parser.add_argument("-de", "--dev", help = "Number of sentences in dev")
    parser.add_argument("-n", "--name", help = "Name of the output dataset")
    parser.add_argument("-o", "--output", help = "Path of the output dataset")
    args = parser.parse_args()

    dataset = Pre_alpha_dataset(args.dataset)
    dataset.generate_train_indexes(
        args.seed, int(args.train) + int(args.test) + int(args.dev), False)
    dataset.generate_test_dev_indexes(
        args.seed, int(args.test), 'test', False)
    dataset.generate_test_dev_indexes(
        args.seed, int(args.dev), 'dev', False)
    
    dataset.save_train_test_dev_datasets(
        args.output, args.name
    )

if __name__ == "__main__":
    main()