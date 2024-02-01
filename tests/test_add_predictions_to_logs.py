import subprocess
import unittest
import glob
import os
import json
import argparse
import sys
import math

import numpy as np 


current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../neasqc_wp61/data/data_processing/")

from compute_metrics_logs import compute_metrics

class TestPredictionsPreAlpha2Alpha3(unittest.TestCase):
    """
    Class for testing the prediction's computations,
    probabilities computations and ML metrics computations
    for preAlpha2 and Alpha3 models
    """
    @classmethod
    def setUpClass(cls) -> None:
        cls.results = {'pre_alpha_2' : [], 'alpha_3' : []}
        cls.n_runs = [int(args.n_runs[i]) for i in range (len(args.n_runs))]
        cls.n_its = [
            int(args.n_iterations[i]) for i in range(len(args.n_iterations))
        ]
        model_parameters = {
            'pre_alpha_2' : {
                'train' : ('../neasqc_wp61/data/datasets/toy_datasets/'
                           'toy_train.tsv'),
                'val' : ('../neasqc_wp61/data/datasets/toy_datasets/'
                           'toy_validation.tsv'),
                'test' : ('../neasqc_wp61/data/datasets/toy_datasets/'
                           'toy_test.tsv')
            },
            'alpha_3' : {
                'train' : ('../neasqc_wp61/data/datasets/toy_datasets/'
                           'multiclass_toy_train_sentence_bert.csv'),
                'val' : ('../neasqc_wp61/data/datasets/toy_datasets/'
                           'multiclass_toy_validation_sentence_bert.csv'),
                'test' : ('../neasqc_wp61/data/datasets/toy_datasets/'
                           'multiclass_toy_test_sentence_bert.csv')
            }
        }
        for it,runs in zip(args.n_iterations, args.n_runs):
            for model in model_parameters.keys():
                ### Run an experiment
                command = ['python',
                    f'../neasqc_wp61/data/data_processing/use_{model}.py',
                    '-i', str(it), '-r', str(runs),
                    '-tr', model_parameters[model]['train'],
                    '-val', model_parameters[model]['val'],
                    '-te', model_parameters[model]['test'],
                    '-o', '../neasqc_wp61/benchmarking/results/raw/'
                    ]
                subprocess.run(command, check=True)
                ### Load the results
                list_of_files = glob.glob(
                    './../neasqc_wp61/benchmarking/results/raw/*.json')
                results_path = max(list_of_files, key=os.path.getctime)
                with open(results_path, 'r') as file:
                    r = json.load(file)
                cls.results[model].append(r)
            

    def test_correct_number_of_preds(self):
        """
        Tests that the number of predictions of models in
        train,val and test partitions is correct
        """
        for model in self.results.keys():
            for i in range(len(self.results[model])):
                for dataset in ('train', 'val', 'test'):
                    self.assertEqual(
                        len(self.results[model][i][
                        f'{dataset}_predictions']),
                        self.n_runs[i]
                    )
                    if dataset != 'test':
                        for j in range(self.n_runs[i]):
                            self.assertEqual(
                            len(self.results[model][i][
                            f'{dataset}_predictions'][j]),
                            self.n_its[i]
                        )
                            
    def test_predictions_are_integers(self):
        """
        Test that all the output predictions are integers
        """
        for model in self.results.keys():
            for i in range(len(self.results[model])):
                for j in range(self.n_runs[i]):
                    for pred in self.results[model][i]['test_predictions'][j]:
                        self.assertIs(type(pred), int)
                    for dataset in ('train', 'val'):
                        for k in range(self.n_its[i]):
                            for pred in self.results[
                                model][i][f'{dataset}_predictions'][j][k]:
                                self.assertIs(type(pred), int)

    def test_probs_add_up_to_one(self):
        """
        Test that all the output predictions are integers
        """
        for model in self.results.keys():
            for i in range(len(self.results[model])):
                for j in range(self.n_runs[i]):
                    for probs in self.results[model][i]['test_probabilities'][j]:
                        self.assertLess(abs(1.0 -sum(probs)), 1e-03)
                    for dataset in ('train', 'val'):
                        for k in range(self.n_its[i]):
                            for probs in self.results[
                                model][i][f'{dataset}_probabilities'][j][k]:
                                self.assertLess(abs(1.0 -sum(probs)), 1e-03)
    
    def test_metrics_are_less_than_one(self):
        """
        Test the computation of metrics with the script
        neasqc_wp61.data.data_processing.compute_metrics_logs.py
        """
        metric_names = [
            'precision', 'accuracy', 'recall',
            'roc_auc', 'f1'
        ]
        for model in self.results.keys():
            print('MODEL = ', model)
            for i in range(len(self.results[model])):
                metric_outputs = compute_metrics(self.results[model][i], average='macro')
                for m in metric_names:
                    for r in range(self.n_runs[i]):
                        ### nan values may occur for some quantities
                        if math.isnan(metric_outputs[f'{m}_test'][r]) != True:
                            self.assertLessEqual(
                                metric_outputs[f'{m}_test'][r], 1)
                        for d in ('train', 'val'):
                            for it in range(self.n_its[i]):
                                if math.isnan(
                                    metric_outputs[f'{m}_{d}'][r][it]) != True:
                                    self.assertLessEqual(
                                        metric_outputs[f'{m}_{d}'][r][it], 1)

class TestPredictionsBeta1(unittest.TestCase):
    """
    Class for testing the prediction's computations,
    probabilities computations and ML metrics computations
    for Beta1 model
    """
    @classmethod
    def setUpClass(cls) -> None:
        ## Run the experiment
        command = ['python',
           f'../neasqc_wp61/data/data_processing/use_beta_1.py',
           '-tr',  ('../neasqc_wp61/data/datasets/toy_datasets/'
           'multiclass_toy_train_sentence_bert.csv') , '-te',
            ('../neasqc_wp61/data/datasets/toy_datasets/'
            'multiclass_toy_test_sentence_bert.csv'),
            '-o', '../neasqc_wp61/benchmarking/results/raw/'        
        ]
        subprocess.run(command, check=True)
        ### Load the results
        list_of_files = glob.glob(
            './../neasqc_wp61/benchmarking/results/raw/*.json')
        results_path = max(list_of_files, key=os.path.getctime)
        with open(results_path, 'r') as file:
            cls.results = json.load(file)          

    def test_metrics_are_less_than_one(self):
        """
        Tests the correct number of predictions in beta1 model
        """ 
        metric_names = [
            'precision', 'accuracy', 'recall', 'f1'
        ]
        metric_outputs = compute_metrics(
            self.results, average='macro'
        )
        for m in metric_names:
            if math.isnan(metric_outputs[f'{m}_test']) != True:
                self.assertLessEqual(
                    metric_outputs[f'{m}_test'], 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nr", "--n_runs", nargs = "+",
        help = "Number of runs of the experiments",
        default = [1,3,5]
    )
    parser.add_argument(
        "-ni", "--n_iterations", nargs = "+",
        help = "Number of iterations of the experiments",
        default = [25,10,5]
    )
    args, remaining = parser.parse_known_args()
    remaining.insert(0, sys.argv[0])
    unittest.main(argv=remaining)