import sys
import os
import argparse
import time 
import json
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/quantum/beta/")
from QuantumKNearestNeighbours import QuantumKNearestNeighbours as qkn
from collections import Counter
from save_json_output import JsonOutputer

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help = "Input path with the labels and vectors", type = str
    )
    parser.add_argument(
        "-k", "--k", help = "Number of K neighbors", type = int
    )
    parser.add_argument(
        "-o", "--output", help = "Output path with the predictions", type = str
    )
    args = parser.parse_args()
    model_name = 'beta'
    timestr = time.strftime("%Y%m%d-%H%M%S")
    json_outputer = JsonOutputer(model_name, timestr, args.output)
    f = open(args.input)
    results_pre_alpha = json.load(f)
    seed = results_pre_alpha['input_args']['seed']
    runs = results_pre_alpha['input_args']['runs']
    train_labels_dir = results_pre_alpha['input_args']['train']
    test_labels = qkn.load_labels(results_pre_alpha['input_args']['test'])
    name_file = args.output + f"beta_neighbors_{seed}_{runs}_{args.k}"
    accuracies_test_list = []
    best_test_accuracy = None
    best_test_run = None

    for i in range(runs):
        t1 = time.time()
        train_vectors = results_pre_alpha['vectors_train'][i]
        test_vectors = results_pre_alpha['vectors_test'][i]
        predictions = qkn(
            train_labels_dir, train_vectors, test_vectors, args.k).predictions

        prediction_list = []
        correct_pred = 0
        for i,pred in enumerate(predictions):
            prediction_list.append(pred)
            if pred == test_labels[i]:
                correct_pred += 1
        accuracy_test = correct_pred/len(test_labels)
        accuracies_test_list.append(accuracy_test)
        best_test_accuracy = max(accuracies_test_list)
        best_test_run = accuracies_test_list.index(max(accuracies_test_list))

        t2 = time.time()
        time_taken = (t2 - t1)

        json_outputer.save_json_output_run_by_run(
        args, prediction_list, time_taken,
        best_final_test_acc = best_test_accuracy,
        best_run = best_test_run, test_acc = accuracy_test,
        seed = seed, k = args.k
        )
    # We save the json output 

if __name__ == "__main__":
    main()
        


