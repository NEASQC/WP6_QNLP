import git 
import argparse
import numpy as np 
import time
import json

def save_json_output(
    model : str, parser_args : argparse.Namespace,
    predictions : list[int], t : float,
    output_path : str, *args : list
):

    json_output = {}

    # 1. Commit hash
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    json_output['hash'] = sha

    # 2. Input arguments
    json_output['input_args'] = vars(parser_args)

    # 3. Predictions 
    json_output['predictions'] = predictions

    # 4. Time taken 
    json_output['time'] = t

    # 5. Loss function and model parameters
    if model == 'pre_alpha' or model == 'pre_alpha_lambeq':
        arrays_cost = [np.array(x) for x in args[0][0]]
        mean_cost = [np.mean(k) for k in zip(*arrays_cost)]
        json_output['cost'] = mean_cost

        arrays_weights = [np.array(x) for x in args[0][1]]
        mean_weights = [np.mean(k) for k in zip(*arrays_weights)]
        json_output['weights'] = mean_weights
    
    # Save the results
    timestr = time.strftime("%Y%m%d-%H%M%S")
    with open (output_path + f'{model}_{timestr}.json', 'w') as file:
        json.dump(json_output, file) 

    


