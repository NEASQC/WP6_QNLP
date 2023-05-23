import sys
import os
import argparse
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/quantum/beta/")
from QuantumKNearestNeighbors import QuantumKNearestNeighbors as qkn
from collections import Counter
import json
import pickle

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--labels", help = "Directory of the dataset containing the labels",
        type = str
    )
    parser.add_argument(
        "-tr", "--train", help = "Directory of the training vectors", type = str
    )
    parser.add_argument(
        "-te", "--test", help = "Directory of the test vectors", type = str
    ) 
    parser.add_argument(
        "-k", "--k", help = "Number of K neighbors", type = int
    )
    parser.add_argument(
        "-r", "--runs", help = "Number of runs", type = int
    )
    parser.add_argument(
        "-o", "--output", help = "Output directory with the predictions", type = str
    )
    args = parser.parse_args()
    name_file = args.output + f"beta_neighbors_{args.k}_{args.runs}"
    with open (args.test) as file:
        vectors_test = json.load(file)
    n_test = len(vectors_test)
    predictions = [[] for i in range(n_test)]
    
    for i in range(args.runs):
        pred = qkn(args.labels, args.train, args.test, args.k).predictions
        for j in range(len(pred)):
            predictions[j].append(pred[j])
        with open(name_file + f'_predictions_run_{i}.pickle', 'wb') as file:
            pickle.dump(predictions, file)
    
    predictions_majority_vote = []
    for i in range(n_test):
        c = Counter(predictions[i])
        value, count = c.most_common()[0]
        predictions_majority_vote.append(value)
    
    with open(name_file + "_predictions.txt", "w") as output:
        for pred in predictions_majority_vote:
            output.write(f"{pred}\n")
    for i in range(args.runs):
        os.remove(name_file + f'_predictions_run_{i}.pickle')

if __name__ == "__main__":
    main()
        


