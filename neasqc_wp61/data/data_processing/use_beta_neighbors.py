import sys
import os
import argparse
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/quantum/beta/")
from QuantumKNearestNeighbors import QuantumKNearestNeighbors as qkn


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
    predictions = qkn(args.labels, args.train, args.test, args.k).predictions
    with open(name_file + "_predictions.txt", "w") as output:
        for pred in predictions:
            output.write(f"{pred}\n")


if __name__ == "__main__":
    main()
        


