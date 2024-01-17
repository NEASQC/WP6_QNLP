import sys
import os
import argparse
import time 
import json
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/quantum/beta_1/")
from beta_1 import QuantumKNearestNeighbours as qkn
from save_json_output import JsonOutputer
from collections import Counter
import numpy as np


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tr", "--train", help = "Directory of the train dataset", type = str, default = '../datasets/toy_datasets/toy_dataset_bert_sentence_embedding_train.csv')
    parser.add_argument("-te", "--test", help = "Directory of the test dataset", type = str, default = '../datasets/toy_datasets/toy_dataset_bert_sentence_embedding_test.csv')
    parser.add_argument("-pca", "--pca_dimension", type= int, help = "Principal component dimension", default = 8)
    parser.add_argument("-k", "--k",  nargs='+', help = "Number of K neighbors", type = int, default = [1, 3, 5, 7, 9])
    parser.add_argument("-o", "--output", help = "Output directory with the predictions", type = str, default = "../../benchmarking/results/raw/")
    parser.add_argument("-s", "--seed", help = "Random seed for selecting labels in case of tie", type = int)
    args = parser.parse_args()

    print(args)

    X_train, X_test, y_train, y_test = qkn.load_labels(
        args.train, args.test, args.pca_dimension)
    

    X_test = qkn.normalise_vector(X_test)
    X_train = qkn.normalise_vector(X_train)

    if args.pca_dimension <  2 * int(np.ceil(np.log2(args.pca_dimension))):
        X_train = qkn.pad_zeros_vector(X_train)
        X_test = qkn.pad_zeros_vector(X_test)
    # We pad with zeros
    
    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_train shape: ", y_train.shape)
    print("y_test shape: ", y_test.shape)

    k_values = args.k

    timestr = time.strftime("%Y%m%d-%H%M%S")

    predictions_list = []
    accuracies_test = []
    time_taken = 0

    t1 = time.time()
    
    beta_model = qkn(
        X_train, X_test, y_train, y_test, k_values
    )
    beta_model.compute_predictions(
        compute_checkpoints = True
    )
    predictions_list = beta_model.get_predictions()


    #Here predictions is a list of lists, where each list contains the predictions for a given k
    predictions_list = np.array(predictions_list).T #So we have for each k a list of predictions

    accuracy_test_list = [] #List of accuracies for each k

    for predictions in predictions_list:
        #We have the predictions for k=x
        correct_pred = np.sum(predictions == y_test)
        accuracy_test = correct_pred/len(y_test)

        accuracy_test_list.append(accuracy_test)
            

    t2 = time.time()
    time_taken = t2 - t1 # /!\ Time taken for the whole algorithm, so for all k values

    for i in range(len(k_values)):
        k = k_values[i]
        predictions = predictions_list[i]
        accuracy_test = accuracy_test_list[i]

        model_name = f"beta_1_{k}"

        # Create the JsonOutputer object
        json_outputer = JsonOutputer(model_name, timestr, args.output)

        json_outputer.save_json_output(args, predictions.tolist(), time_taken, final_val_acc = [accuracy_test], best_final_val_acc=accuracy_test, k = k)

    # We save the json output 

if __name__ == "__main__":
    main()
        


