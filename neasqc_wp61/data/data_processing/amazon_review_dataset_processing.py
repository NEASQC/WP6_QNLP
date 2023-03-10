import pandas as pd 
import os 
import numpy as np 
from sklearn.model_selection import train_test_split
import json

current_path = os.path.dirname(os.path.abspath(__file__))

def load_data():
    data = pd.read_csv(
        current_path + 
        '/../datasets/withtags_amazonreview_train.tsv', sep='\t+',
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

def dataset_to_dict(dataset : pd.DataFrame,
                    train_size : float) -> dict:
    """
    Converts a pd.Dataframe to a dictionary with a key structure suitable
    to be run on pre-alpha mode
    Parameters
    ----------
    dataset : pd.DataFrame
        Dataframe we want to convert
    train_size : float 
        Proportion of the dataset included on the train split 
    Returns 
    -------
    generated_dataset : dict
        Our dataset as a dictionary with a structure that fits 
        on pre-alpha model. 
    """
    train, test = train_test_split(
        dataset, train_size=train_size,
        random_state = 30042021)
    list_dataset = [train, test]
    generated_dataset = {"train_data" : [], "test_data" : []}
    for i,j in enumerate(list(generated_dataset.keys())):
        df = list_dataset[i]
        for k in range(df.shape[0]):
            data_value = {}
            data_value["sentence"] = df["sentence"].iloc[k]
            data_value["structure_tilde"] = df["structure_tilde"].iloc[k]
            if df["label"].iloc[k] == 1 :
                data_value["truth_value"] = False
            if df["label"].iloc[k] == 2:
                data_value["truth_value"] = True
            generated_dataset[j].append(data_value)

    return generated_dataset


def generate_reduced_dataset(dataset : pd.DataFrame) -> dict:
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


def save_dataset(
    dataset : dict,
    name : str, path : str):
    """
    Saves our dataset as json file
    
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






def main():

    data = load_data()
    filtered_dataset =  filter_structures(data)
    filtered_dictionary = dataset_to_dict(filtered_dataset, 0.80)
    save_dataset(filtered_dictionary, 'amazon_filtered_dictionary',
    current_path + '/../datasets/')
    reduced_dataset = generate_reduced_dataset(filtered_dataset)
    reduced_dict = dataset_to_dict(reduced_dataset, 0.99)
    save_dataset(reduced_dict, 'reduced_amazon_filtered_dictionary',
                 current_path + '/../datasets/')




if __name__ == "__main__":
    main()