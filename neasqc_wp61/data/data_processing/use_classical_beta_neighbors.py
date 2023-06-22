import sys
import os
import argparse
import git 
import time 
import json 
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/classical")
from classical_k_means import ClassicalKNearestNeighbors as ckn


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

    name_file = args.output + f"classical_beta_neighbors_{args.k}"
    t1 = time.time()
    predictions = qkn(args.labels, args.train, args.test, args.k).predictions
    t2 = time.time()
    with open(name_file + "classical_predictions.txt", "w") as output:
        for pred in predictions:
            output.write(f"{pred}\n")

########################################################
################ JSON output ###########################
########################################################

    output = {}

    # 1. Commit HASH 

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    output['hash'] = sha

    # 2. Input arguments 

    input_args = {
        'labels' : args.labels, 'train' : args.train,
        'test' : args.test, 'k' : args.k,
        'output' : args.output}
    output['input_args'] = input_args

    # 3. Predictions 

    output['predictions'] = predictions

    # 4. Time taken 

    output['time'] = t2 - t1 

    # Save the results 
    timestr = time.strftime("%Y%m%d-%H%M%S")
    with open (args.output + f'classical_beta_neighbors_{timestr}.json', 'w') as file:
        json.dump(output, file)

if __name__ == "__main__":
    main()
        


