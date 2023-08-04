import git 
import argparse
import numpy as np 
import time
import json

def save_json_output(
    model : str, parser_args : argparse.Namespace,
    predictions : list[int], t : list[float],
    output_path : str, **kwargs : list
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

    # 5. Other outputs 
    for k,v in kwargs.items():
        json_output[k] = v

    # Save the results
    timestr = time.strftime("%Y%m%d-%H%M%S")
    with open (output_path + f'{model}_{timestr}.json', 'w') as file:
        json.dump(json_output, file, indent=4) 


    