import git 
import argparse
import numpy as np 
import time
import json
import os

class JsonOutputer():
    """An object allowing to save the results of an experiment in a json file"""

    file_name : str # Name of the file to generate
    file_exists : bool  # Whether the file already exists
    fix_keys : list[str]    # Keys that should be written only once
    non_array_keys : list[str]  # Keys that should not be written as arrays

    def __init__(self, model_name, timestr, output_path) -> None:
        self.file_exists = False
        self.fix_keys = ['hash', 'input_args']
        self.non_array_keys = ['best_val_acc', 'best_run']

        self.filepath = f'{model_name}_{timestr}.json'
        self.filepath = os.path.join(output_path, self.filepath)


    def save_json_output_run_by_run(
        self, parser_args: argparse.Namespace, predictions: list[int], 
        t: list[float], **kwargs: list) -> None:
        """
        This function is used to save the results of each run in a json file
        The results of each run are appended to the existing data in the file
        """
        json_output = {}

        # 1. Commit hash
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        json_output['hash'] = sha

        # 2. Input arguments
        json_output['input_args'] = vars(parser_args)

        # 3. Predictions
        json_output['predictions'] = predictions if self.file_exists else [predictions]

        # 4. Time taken
        json_output['time'] = t if self.file_exists else [t]

        # 5. Other outputs
        for k, v in kwargs.items():
            json_output[k] = v if self.file_exists else [v]

        if self.file_exists:
            with open(self.filepath, 'r') as file:
                existing_data = json.load(file)
            # Append the new data to the existing data
            for key, value in json_output.items():
                if key not in existing_data:
                    existing_data[key] = value
                else:
                    if key in self.non_array_keys:
                        existing_data[key] = value
                    elif key not in self.fix_keys:
                        existing_data[key].append(value)
                    else:
                        pass
            json_output = existing_data

        self.file_exists = True

        # Save the results to the file
        with open(self.filepath, 'w') as file:
            json.dump(json_output, file, indent=4)

    def save_json_output(
        self, parser_args: argparse.Namespace, predictions: list[int], 
        t: list[float], **kwargs: list) -> None:
        """
        This function is used to save the results of the whole experiment at once in a json file
        """
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
        for k, v in kwargs.items():
            json_output[k] = v

        # Save the results
        with open(self.filepath, 'w') as file:
            json.dump(json_output, file, indent=4)

