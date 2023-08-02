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

from lambeq import IQPAnsatz, Sim14Ansatz, Sim15Ansatz, StronglyEntanglingAnsatz
from alpha_pennylane_lambeq_trainer import Alpha_pennylane_lambeq_trainer
from save_json_output import save_json_output




parser = argparse.ArgumentParser()

# To chose the model
parser.add_argument("-v", "--version", help = "Choose between alpha_pennylane_lambeq and alpha_pennylane_lambeq_original", type = str, default = "alpha_pennylane_lambeq")
parser.add_argument("-pca", "--pca", help = "Choose the reduced dimension for the word embeddings", type = int, default = 22)

parser.add_argument("-s", "--seed", help = "Seed for the initial parameters", type = int, default = 0)
parser.add_argument("-i", "--iterations", help = "Number of iterations of the optimiser", type = int, default = 100)
parser.add_argument("-r", "--runs", help = "Number of runs", type = int, default = 1)
parser.add_argument("-tr", "--train", help = "Directory of the train dataset", type = str, default = '../toy_dataset/toy_dataset_bert_sentence_embedding_train.csv')
parser.add_argument("-te", "--test", help = "Directory of the test datset", type = str, default = '../toy_dataset/toy_dataset_bert_sentence_embedding_test.csv')
parser.add_argument("-o", "--output", help = "Output directory with the predictions", type = str, default = "../../benchmarking/results/raw/")
parser.add_argument("-an", "--ansatz", help = "Ansatz to be used in quantum circuits", type = str, default = "IQP")
parser.add_argument("-qn", "--qn", help = "Number of qubits per NOUN type", type = int, default = 1)
parser.add_argument("-qs", "--qs", help = "Number of qubits per SENTENCE type", type = int, default = 1)
parser.add_argument("-nl", "--n_layers", help = "Number of layers for the circuits", type = int, default = 4)
parser.add_argument("-np", "--n_single_qubit_params", help = "Number of parameters per qubit", type = int, default = 3)
parser.add_argument("-b", "--batch_size", help = "Batch size", type = int, default = 2048)

# Hyperparameters
parser.add_argument("-lr", "--lr", help = "Learning rate", type = float, default = 2e-2)
parser.add_argument("-wd", "--weight_decay", help = "Weight decay", type = float, default = 0.0)
parser.add_argument("-slr", "--step_lr", help = "Step size for the learning rate scheduler", type = int, default = 20)
parser.add_argument("-g", "--gamma", help = "Gamma for the learning rate scheduler", type = float, default = 0.5)

args = parser.parse_args()



def main(args):
    random.seed(args.seed)
    seed_list = random.sample(range(1, int(2**32 - 1)), int(args.runs))

    if args.ansatz == "IQP":
        ansatz = IQPAnsatz
    elif args.ansatz == "Sim14":
        ansatz = Sim14Ansatz
    elif args.ansatz == "Sim15":
        ansatz = Sim15Ansatz
    elif args.ansatz == "StronglyEntangling":
        ansatz = StronglyEntanglingAnsatz
    else:
        raise ValueError("The ansatz is not valid")
    

    if args.version == "alpha_pennylane_lambeq":
        version_original = False
        model_name = "alpha_pennylane_lambeq"
    elif args.version == "alpha_pennylane_lambeq_original":
        version_original = True
        model_name = "alpha_pennylane_lambeq_original"
    else:
        raise ValueError("The version is not valid")
    
    
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

        trainer = Alpha_pennylane_lambeq_trainer(args.iterations, args.train, args.test, seed_list[i],
                                                ansatz, args.qn, args.qs, args.n_layers, args.n_single_qubit_params, 
                                                args.batch_size, args.lr, args.weight_decay, args.step_lr, args.gamma, version_original, args.pca)
        
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
    