import sys
import os
import argparse
import time 
import json
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/quantum/beta/")
from QuantumKNearestNeighbors import QuantumKNearestNeighbors as qkn
from save_json_output import save_json_output
from collections import Counter


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
    f = open(args.input)
    results_pre_alpha = json.load(f)
    seed = results_pre_alpha['input_args']['seed']
    runs = results_pre_alpha['input_args']['runs']
    train_labels_dir = results_pre_alpha['input_args']['train']
    test_labels = qkn.load_labels(results_pre_alpha['input_args']['test'])
    name_file = args.output + f"beta_neighbors_{seed}_{runs}_{args.k}"
    predictions_list = [[] for i in range(len(test_labels))]
    accuracies_test = []
    time_list = []

    for i in range(runs):
        t1 = time.time()
        train_vectors = results_pre_alpha['vectors_train'][i]
        test_vectors = results_pre_alpha['vectors_test'][i]
        predictions = qkn(
            train_labels_dir, train_vectors, test_vectors, args.k).predictions

        correct_pred = 0
        for i,pred in enumerate(predictions):
            predictions_list[i].append(pred)
            if pred == test_labels[i]:
                correct_pred += 1
        accuracies_test.append(correct_pred/len(test_labels))
        t2 = time.time()
        time_list.append(t2 - t1)
    print(predictions_list)
    predictions_majority_vote = []
    for i in range(len(test_labels)):
        c = Counter(predictions_list[i])
        value, count = c.most_common()[0]
        predictions_majority_vote.append(value)
    with open(name_file + "_predictions.txt", "w") as output:
            for pred in predictions_majority_vote:
                output.write(f"{pred}\n")
    best_final_accuracy = max(accuracies_test)    
    best_run = accuracies_test.index(best_final_accuracy)

    save_json_output(
    'beta', args, predictions_majority_vote,
    time_list, args.output, best_final_val_acc = best_final_accuracy,
    best_run = best_run, seed = seed, final_val_acc = accuracies_test,
    k = args.k
    )
    # We save the json output 

if __name__ == "__main__":
    main()
        


