import argparse
import os
import json

"""
This script is used to combine the results of multiple runs of the same experiment into a single JSON file.
NB: The JSON files should have the same input arguments.
"""

def merge_json(json_files):
    json_output = []
    is_first_file = True

    fix_keys = ['hash', 'input_args']   # Keys that should be written only once

    for json_file in json_files:
        with open(json_file, 'r') as file:
            data = json.load(file)

            if is_first_file:
                json_output = data
                is_first_file = False
            else:
                # Append the new json to the existing json
                for key, value in data.items():
                    if key not in json_output:
                        json_output[key] = value
                    else:
                        # Update the best_val_acc and best_run only if > existing value
                        if key == 'best_val_acc':
                            if value > json_output[key]:
                                json_output[key] = value
                                json_output['best_run'] = data['best_run']
                        # Append the number of runs of each experiment
                        elif key == 'input_args':
                            json_output[key]['runs'] += value['runs']
                        elif key == 'best_run':
                            pass
                        elif key not in fix_keys:
                            json_output[key].extend(value)
                        else:
                            pass

    return json_output


def check_experiments(json_files):
    """
    Check if the experiments in the JSON files are the same
    """
    previous_experiment_args = None
    ignore_keys = ['seed', 'runs'] # Keys to ignore when comparing experiments

    for json_file in json_files:
        with open(json_file, 'r') as file:
            data = json.load(file)
            if previous_experiment_args is None:
                previous_experiment_args = data['input_args']
            else:
                equal_dicts(previous_experiment_args, data['input_args'], ignore_keys)
                previous_experiment_args = data['input_args']

    return None


#https://stackoverflow.com/questions/10480806/compare-dictionaries-ignoring-specific-keys
def equal_dicts(d1, d2, ignore_keys):
    ignored = set(ignore_keys)
    for k1, v1 in d1.items():
        if k1 not in ignored and (k1 not in d2 or d2[k1] != v1):
            raise ValueError("Experiments have different inpiut arguments! Please check the json files")
    for k2, v2 in d2.items():
        if k2 not in ignored and k2 not in d1:
            raise ValueError("Experiments have different inpiut arguments! Please check the json files")
    return True



def main():
    parser = argparse.ArgumentParser(description='Combine JSON files into a single JSON file')
    parser.add_argument('--files', nargs='+', help='List of JSON files to merge')
    parser.add_argument('--folder', help='Folder containing JSON files to merge')
    parser.add_argument('--output_name', help='Name of the output JSON file')

    args = parser.parse_args()

    if args.files:
        json_files = args.files
    elif args.folder:
        folder_path = args.folder
        json_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
    else:
        print("Please provide either --files or --folder argument.")
        return

    if not json_files:
        print("No JSON files found.")
        return


    if len(json_files) == 1:
        print("Only one JSON file found. No need to merge.")
        return

    
    check_experiments(json_files) # Check if the experiments in the JSON files are the same

    merged_data = merge_json(json_files)

    if args.output_name:
        output_file = args.output_name
    else:
        output_file = 'merged'
    
    with open(f'{output_file}.json', 'w') as file:
        json.dump(merged_data, file, indent=4)

    print(f'Merged JSON data saved to {output_file}.json')

if __name__ == "__main__":
    main()
