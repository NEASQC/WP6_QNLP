import sys
import os
import argparse
import time 
import json
from collections import Counter

import numpy as np

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/quantum/beta/")
from beta_1 import QuantumKNearestNeighbours as qkn
from save_json_output import JsonOutputer
from utils import *

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tr", "--train",
        help = "Directory of the train dataset",
        type = str,
        default = './../datasets/toy_datasets/binary_toy_train_sentence_bert.csv'
    )
    parser.add_argument(
        "-te", "--test",
        help = "Directory of the test dataset",
        type = str,
        default = './../datasets/toy_datasets/binary_toy_test_sentence_bert.csv'
    )
    parser.add_argument(
        "-pca", "--pca_dimension",
        help = "Principal component dimension",
        type= int,  default = 4
    )
    parser.add_argument(
        "-k", "--k",
        nargs='+',
        help = "Number of K neighbors",
        type = int, default = [1, 3]
    )
    parser.add_argument(
        "-o", "--output",
        help = "Output directory with the predictions",
        type = str, default = "../../benchmarking/results/raw/"
    )
    args = parser.parse_args()

    X_train, y_train = load_data_pipeline(
        args.train, args.pca_dimension
    )
    X_test, y_test = load_data_pipeline(
        args.test, args.pca_dimension
    )

    timestr = time.strftime("%Y%m%d-%H%M%S")

    predictions_list = []
    accuracy_test_list = [] #List of accuracies for each k
    time_taken = 0

    t1 = time.time()
    
    beta_model = qkn(
        X_train, X_test, y_train, y_test, args.k
    )
    beta_model.compute_predictions(
        compute_checkpoints = True
    )
    predictions_list = beta_model.get_predictions()
    # For each k, we have a list of predictions
    predictions_list = np.array(predictions_list).T 
    for predictions in predictions_list:
        correct_pred = np.sum(predictions == y_test)
        accuracy_test = correct_pred/len(y_test)
        accuracy_test_list.append(accuracy_test)
            
    t2 = time.time()
    time_taken = t2 - t1 # /!\ Time taken for the whole algorithm, so for all k values

    for i in range(len(args.k)):
        k = args.k[i]
        predictions = predictions_list[i]
        accuracy_test = accuracy_test_list[i]
        model_name = f"beta_1"

        # Create the JsonOutputer object
        json_outputer = JsonOutputer(model_name, timestr, args.output)
        json_outputer.save_json_output_beta_1(
            args, time_taken, y_train.tolist(), y_test.tolist(),
            final_val_acc = accuracy_test,
            test_predictions = predictions.tolist(),
            best_final_val_acc=accuracy_test, k = k)
        
if __name__ == "__main__":
    main()
        