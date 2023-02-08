import sys 
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../doc/tutorials/")
from dataset_example_seeded import pre_alpha_classifier_seeded
import numpy as np
import json 


def compute_means(
    x: list[list[float]]) -> list[float]:
    """
    For a list containing the values of cost function 
    during the optimization for several runs
    (with different seed) of the model
    ,it computes the mean of each step of the 
    optimisation across the different runs. 
    
    Parameters
    ----------
    x : [list[list[float]]]
        Nested list containig the values of cost function
        in optimisation across several runs
    Returns
    -------
    x_means : list[float]
        List with mean for each step of the optimisation 
        across different runs.
    """

    x_means = []
    n_runs = len(x)
    n_iterations = len(x[0])
    for i in range(n_iterations):
        x_means.append(np.mean([x[j][i] for j in range(n_runs)]))

    return x_means

def compute_stds(
    x: list[list[float]]) -> list[float]:
    """
    For a list containing the values of cost function 
    during the optimization for several runs
    (with different seed) of the model
    ,it computes the mean of each step of the 
    optimisation across the different runs. 
    
    Parameters
    ----------
    x : [list[list[float]]]
        Nested list containig the values of cost function
        in optimisation across several runs
    Returns
    -------
    x_stds : list[float]
        List with mean for each step of the optimisation 
        across different runs.
    """

    x_stds = []
    n_runs = len(x)
    n_iterations = len(x[0])
    for i in range(n_iterations):
        x_stds.append(np.std([x[j][i] for j in range(n_runs)]))

    return x_stds

def benchmarking_pipeline(
    n_runs: int,
    initial_seed: float = 18061997,
    customised_seed_list: list = None,
    save_results: bool = None, 
    path: str = None, name: str = None):
    """
    Performs the pipeline for pre-alpha benchmarking.
    For a given number of runs and seeds for initial 
    circuit parameters if computes the mean values for 
    the optimisation of the cost function in pre-alpha
    model.

    Parameters
    ----------
    n_runs : int
        Number of runs to be performed
    initial_seed : float, optional
        Sets the random seed for the first run. The random seed will be changing 
        over the different runs of the model, so changing the seed for the first 
        run will also change the seeds of the subsequent runs. 
    customised_seed : list, optional
        Sets a predefined list of seeds. Its length must match the number of 
        runs to be performed. Default is None. 
    save_results : list, optional
        Boolean that decides if the results of the optimisations over
        different runs will be saved. If True, it will save the results 
        as json format. 
    path : str, optional
        When save_results is set to True, specifies the path where to store 
        the results for the optimisations. 
    name : str, optional
        When save_results is set to True, specifies the name of the results 
        that are being saved.
    """
    
    if customised_seed_list != None:
        seed_list = customised_seed_list
    else:
        seed_list =[initial_seed + i for i in range(n_runs)]

    results = []
    for i in range(n_runs):
        results.append(pre_alpha_classifier_seeded(seed_list[i]))

    x = np.arange(len(results[0])).tolist() # Number of steps in optimisation   
    y = compute_means(results)
    y_stds = compute_stds(results)

    if save_results == True:
        with open (path + "/" + name + ".json", "w") as file:
            json.dump([x, y, y_stds], file)

    return x, y, y_stds

    


    
    