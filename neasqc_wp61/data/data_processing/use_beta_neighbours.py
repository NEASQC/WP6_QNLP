import sys
import os
import argparse
import time 
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/quantum/beta/")
from QuantumKNearestNeighbors import QuantumKNearestNeighbors as qkn
from save_json_output import save_json_output


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--labels", help = "Path of the dataset containing the training labels",
        type = str
    )
    parser.add_argument(
        "-tr", "--train", help = "Path of the training vectors", type = str
    )
    parser.add_argument(
        "-te", "--test", help = "Path of the test vectors", type = str
    ) 
    parser.add_argument(
        "-k", "--k", help = "Number of K neighbors", type = int
    )
    parser.add_argument(
        "-o", "--output", help = "Output path with the predictions", type = str
    )
    args = parser.parse_args()

    name_file = args.output + f"beta_neighbors_{args.k}"
    t1 = time.time()
    predictions = qkn(args.labels, args.train, args.test, args.k).predictions
    t2 = time.time()
    with open(name_file + "_predictions.txt", "w") as output:
        for pred in predictions:
            output.write(f"{pred}\n")

    save_json_output(
    'beta', args, predictions,
    t2 - t1, args.output
    )
    # We save the json output 

if __name__ == "__main__":
    main()
        


