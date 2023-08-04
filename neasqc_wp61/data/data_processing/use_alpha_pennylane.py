import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/quantum/alpha/module/")
import argparse

import json
import numpy as np

import random, os
import numpy as np
import torch
import time
import git

from alpha_pennylane_trainer import Alpha_pennylane_trainer
from save_json_output import save_json_output



parser = argparse.ArgumentParser()

# To chose the model

parser.add_argument("-s", "--seed", help = "Seed for the initial parameters", type = int, default = 0)
parser.add_argument("-i", "--iterations", help = "Number of iterations of the optimiser", type = int, default = 100)
parser.add_argument("-r", "--runs", help = "Number of runs", type = int, default = 1)
parser.add_argument("-tr", "--train", help = "Directory of the train dataset", type = str, default = '../toy_dataset/toy_dataset_bert_sentence_embedding_train.csv')
parser.add_argument("-te", "--test", help = "Directory of the test datset", type = str, default = '../toy_dataset/toy_dataset_bert_sentence_embedding_test.csv')
parser.add_argument("-o", "--output", help = "Output directory with the predictions", type = str, default = "../../benchmarking/results/raw/")

parser.add_argument("-nq", "--n_qubits", help = "Number of qubits in our circuit", type = int, default = 3)
parser.add_argument("-qd", "--q_delta", help = "Initial spread of the parameters", type = float, default = 0.01)
parser.add_argument("-b", "--batch_size", help = "Batch size", type = int, default = 2048)

# Hyperparameters
parser.add_argument("-lr", "--lr", help = "Learning rate", type = float, default = 2e-3)
parser.add_argument("-wd", "--weight_decay", help = "Weight decay", type = float, default = 0.0)
parser.add_argument("-slr", "--step_lr", help = "Step size for the learning rate scheduler", type = int, default = 20)
parser.add_argument("-g", "--gamma", help = "Gamma for the learning rate scheduler", type = float, default = 0.5)

args = parser.parse_args()



def main(args):
    random.seed(args.seed)
    seed_list = random.sample(range(1, int(2**32 - 1)), int(args.runs))
    
    
    all_training_loss_list = []
    all_training_acc_list = []
    all_validation_loss_list = []
    all_validation_acc_list = []

    all_prediction_list = []
    all_time_list = []

    all_best_model_state_dict = []

    best_val_acc_all_runs = 0
    best_run = 0


    for i in range(args.runs):
        t_before = time.time()
        print("\n")
        print("-----------------------------------")
        print("run = ", i+1)
        print("-----------------------------------")
        print("\n")

        trainer = Alpha_pennylane_trainer(args.iterations, args.train, args.test, seed_list[i], args.n_qubits, args.q_delta,
                                          args.batch_size, args.lr, args.weight_decay, args.step_lr, args.gamma)
        
        training_loss_list, training_acc_list, validation_loss_list, validation_acc_list, best_val_acc, best_model = trainer.train()

        t_after = time.time()
        print("Time taken for this run = ", t_after - t_before, "\n")
        all_time_list.append(t_after - t_before)

        all_training_loss_list.append(training_loss_list)
        all_training_acc_list.append(training_acc_list)
        all_validation_loss_list.append(validation_loss_list)
        all_validation_acc_list.append(validation_acc_list)

        prediction_list = trainer.predict()
        all_prediction_list.append(prediction_list.tolist())

        all_best_model_state_dict.append(best_model)


        if best_val_acc > best_val_acc_all_runs:
            best_val_acc_all_runs = best_val_acc
            best_run = i

    
    model_name = "alpha_pennylane"

    all_best_model_state_dict = [convert_to_json_serializable(model) for model in all_best_model_state_dict]


    # Save the results
    save_json_output(model_name, args, all_prediction_list, all_time_list, args.output, best_val_acc = best_val_acc_all_runs, 
                    best_run = best_run, seed_list = seed_list, val_acc = all_validation_acc_list, val_loss = all_validation_loss_list,
                    train_acc = all_training_acc_list, train_loss = all_training_loss_list, weights = all_best_model_state_dict
                    )
   




def convert_to_json_serializable(data):
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.tolist()
        elif isinstance(value, dict):
            data[key] = convert_to_json_serializable(value)
    return data


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)              
    