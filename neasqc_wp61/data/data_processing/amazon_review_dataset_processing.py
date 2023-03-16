import pandas as pd 
import os 
import numpy as np 
from sklearn.model_selection import train_test_split
import json

current_path = os.path.dirname(os.path.abspath(__file__))

def load_data():
    data = pd.read_csv(
        current_path + 
        '/../datasets/amazonreview_train_filtered_train.tsv', sep='\t+',
        header=None, names=['label', 'sentence', 'structure_tilde'],
        engine='python')
    
    return data
    

def filter_structures(dataset : pd.DataFrame) -> pd.DataFrame:
    """
    Filters the amazon reviews dataframe with the sentences structures we 
    have chosen to work with. 

    Parameters
    ----------
    dataset : pd.DataFrame
        withtags_amazonreview_train.tsv file as pd.DataFrame
    Returns 
    -------
    amazon_dataset_filtered : pd.DataFrame
        The amazon dataset filtered with our selected sentences. 
    """
    freq_structure = dataset['structure_tilde'].value_counts().to_dict()
    key_list = list(freq_structure.keys())
    selected_items = [25, 50, 59, 61, 74] 
    # These are the sentences structures we have selected

    frames = []
    for s in selected_items:
        df = dataset.loc[
            dataset['structure_tilde'] ==  r"{}".format(key_list[s-1])
            # The selected_items were written starting in number 1 ..
            # .. instead of 0
            ]
        frames.append(df)

    amazon_dataset_filtered = pd.concat(frames)


    return amazon_dataset_filtered

def dataset_to_dict(dataset : pd.DataFrame) -> dict:
    """
    Converts a pd.Dataframe to a dictionary with a key structure suitable
    to be run on pre-alpha mode
    Parameters
    ----------
    dataset : pd.DataFrame
        Dataframe we want to convert
    Returns 
    -------
    generated_dataset : dict
        Our dataset as a dictionary with a structure that fits 
        on pre-alpha model. 
    """
    generated_dataset = {"train_data" : []}
    for k in range(dataset.shape[0]):
        data_value = {}
        data_value["sentence"] = dataset["sentence"].iloc[k]
        data_value["structure_tilde"] = dataset["structure_tilde"].iloc[k]
        if dataset["label"].iloc[k] == 1 :
            data_value["truth_value"] = False
        if dataset["label"].iloc[k] == 2:
            data_value["truth_value"] = True
        generated_dataset["train_data"].append(data_value)

    return generated_dataset


def generate_reduced_dataset(dataset : pd.DataFrame) -> pd.DataFrame:
    """
    Selects 700 sentences for each chosen sentence structure 
    and generates a dataset with a structure that fits on pre-alpha
    model

    Parameters
    ----------
    dataset : pd.Dataframe
        withtags_amazonreview_train.tsv file as pd.DataFrame
    Returns 
    -------
    reduced_dataset : dict
        Our reduced dataset as a dictionary with a structure that fits 
        on pre-alpha model.     
    """
    sentence_structures = dataset['structure_tilde'].unique()
    frames = []
    for s in sentence_structures:
        data = dataset.loc[dataset['structure_tilde'] == s]
        frames.append(data.sample(n = 700, random_state = 18061997))
    reduced_dataset = pd.concat(frames)
    return reduced_dataset


def save_dictionary(
    dataset : dict,
    name : str, path : str):
    """
    Saves our dictionary as json file
    
    Parameters
    ----------
    dataset : dict
        Dataset we want to save
    name :
        Name we want to assign to the saved file
    path :
        Path where we want to save the file
    """
    with open(path + name + '.json', 'w') as file:
        json.dump(dataset, file)


def save_dataset(
        dataset : pd.DataFrame,
        name : str, path : str):
    """
    Saves our dataset as tsv file

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset we want to save
    name :
        Name we want to assign to the saved file
    path :
        Path where we want to save the file
    """
    dataset.to_csv(
        path + name + '.tsv', header = False,
        sep = '\t', index = False
    )




def main():

    data = load_data()
    reduced_dataset = generate_reduced_dataset(data)
    reduced_dict = dataset_to_dict(reduced_dataset)
    save_dictionary(reduced_dict, 'amazonreview_reduced_train_pre_alpha',
                 current_path + '/../datasets/')




if __name__ == "__main__":
    main()