import pandas as pd
import json

from module.Qsentence import *

def data_preparation(filename: str)->list:
    """Transforms sentences into Qsentences.

    Takes sentence train and test data along with their repective true or false labels and transforms each sentence into a so-called Qsentence.:

    Parameters
    ----------
    filename : str
        File path to the data to be prepared

    Returns
    -------
    Dataset: list
        List of Qsentence types corresponding to each sentence.
        

    """
    with open(filename) as f:
        data = json.load(f)
    dftrain = pd.DataFrame(data['train_data'])
    dftrain["truth_value"]= dftrain["truth_value"].map({True: [1,0], False: [0,1]})
    dftest = pd.DataFrame(data['test_data'])
    dftest["truth_value"]= dftest["truth_value"].map({True: [1,0], False: [0,1]})


    Dataset = []
    for sentence, label in zip(dftrain["sentence"], dftrain["truth_value"]):
        #print("Sentence: ", sentence, "     label: ", label)
        Dataset.append(Qsentence(sentence_string=sentence, n_dim=1, s_dim=1, depth = 1, label = label))
    return Dataset