import pandas as pd 
import os 
import numpy as np 
from sklearn.model_selection import train_test_split
import json

current_path = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(
    current_path + 
    '/../datasets/withtags_amazonreview_train.tsv', sep='\t',
    header=None, names=['label', 'sentence', 'structure_tilde']
)

def filter_structures(dataset : pd.DataFrame) -> pd.DataFrame:

    freq_structure = dataset['structure_tilde'].value_counts().to_dict()
    key_list = list(freq_structure.keys())
    item_list = list(freq_structure.items())
    selected_items = [25, 50, 59, 61, 74] 
    # These are the sentences structures we have selected


    frames = []
    for s in selected_items:
        df = dataset.loc[
            dataset['structure_tilde'] ==  key_list[s]
            ]
        frames.append(df)

    amazon_dataset_processed = pd.concat(frames)

    return amazon_dataset_processed

def generate_dataset(dataset : pd.DataFrame) -> dict:

    train, test = train_test_split(dataset, test_size=0.2)
    list_dataset = [train, test]
    generated_dataset = {"train_data" : [], "test_data" : []}
    for i,j in enumerate(list(generated_dataset.keys())):
        df = list_dataset[i]
        for k in range(df.shape[0]):
            data_value = {}
            data_value["sentence"] = df["sentence"].iloc[k]
            data_value["structure_tilde"] = df["structure_tilde"].iloc[k]
            if df["label"].iloc[i] == 1 :
                data_value["truth_value"] = False
            if df["label"].iloc[i] == 2:
                data_value["truth_value"] = True
            generated_dataset[j].append(data_value)

    return generated_dataset


def save_dataset(
    dataset : dict,
    name : str, path : str):

    with open(path + name + '.json', 'w') as file:
        json.dump(dataset, file)






def main():

    filtered_dataset =  filter_structures(data)
    amazon_filtered_dataset = generate_dataset(filtered_dataset)
    save_dataset(amazon_filtered_dataset, 'amazon_filtered_dataset',
    current_path + '/../datasets/')




if __name__ == "__main__":
    main()