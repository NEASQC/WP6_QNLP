import sys
import os
import argparse
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/quantum/pre-alpha-lambeq/")
from PreAlphaLambeq import *
from collections import Counter
import random
import pickle
import json
import time 
import torch 
import numpy as np
from statistics import mean
from save_json_output import JsonOutputer

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--seed", help = "Seed for the initial parameters", type = int)
    parser.add_argument(
        "-op", "--optimiser", help = "Optimiser to use", type = str)
    parser.add_argument(
        "-i", "--iterations", help = "Number of iterations of the optimiser", type = int)
    parser.add_argument(
        "-r", "--runs", help = "Number of runs", type = int)
    parser.add_argument(
        "-tr", "--train", help = "Directory of the train dataset", type = str)
    parser.add_argument(
        "-val", "--validation", help = "Directory of the validation dataset", type = str)
    parser.add_argument(
        "-te", "--test", help = "Directory of the test datset", type = str)
    parser.add_argument(
        "-o", "--output", help = "Output directory with the predictions", type = str)
    parser.add_argument(
        "-an", "--ansatz", help = "Ansatz to be used in quantum circuits", type = str)
    parser.add_argument(
        "-qn", "--qn", help = "Number of qubits per NOUN type", type = int)
    parser.add_argument(
        "-nl", "--n_layers", help = "Number of layers for the circuits", type = int)
    parser.add_argument(
        "-np", "--n_single_qubit_params", help = "Number of parameters per qubit", type = int)
    parser.add_argument(
        "-b", "--batch_size", help = "Batch size used for traininig the model", type = int)
    args = parser.parse_args()
    train_dataset_name = os.path.basename(args.train)
    train_dataset_name = os.path.splitext(train_dataset_name)[0]
    validation_dataset_name = os.path.basename(args.validation)
    validation_dataset_name = os.path.splitext(validation_dataset_name)[0]
    test_dataset_name = os.path.basename(args.test)
    test_dataset_name = os.path.splitext(test_dataset_name)[0]
    train_path = os.path.dirname(args.train)
    validation_path = os.path.dirname(args.validation)
    test_path = os.path.dirname(args.test)

    validation_accuracy_list = []
    best_val_accuracy = None
    best_val_run = None
    model_name = 'pre_alpha_lambeq'
    timestr = time.strftime("%Y%m%d-%H%M%S")
    json_outputer = JsonOutputer(model_name, timestr, args.output)
    # The number of qubits per sentence is pre-defined as we still need 
    # to improve our model
    qs = 1 
    name_file = args.output + f"pre_alpha_lambeq_{args.seed}_{args.optimiser}_"\
        f"{args.iterations}_{args.runs}_{args.ansatz}_{args.qn}_"\
        f"{qs}_{args.n_layers}_{args.n_single_qubit_params}_{args.batch_size}"
    # Name of the file to store the results. 

    random.seed(args.seed)
    seed_list = random.sample(range(1, 10000000000000), int(args.runs))
    # Set the random seed. 


    labels_train = PreAlphaLambeq.load_dataset(args.train)[1]
    labels_validation = PreAlphaLambeq.load_dataset(args.validation)[1]
    labels_test = PreAlphaLambeq.load_dataset(args.test)[1]

    # Load sentences and labels


    def loss(y_hat, y):
        return torch.nn.functional.binary_cross_entropy(
            y_hat, y
        )

    def acc(y_hat, y):
        return (torch.argmax(y_hat, dim=1) ==
            torch.argmax(y, dim=1)).sum().item()/len(y)

    acc_np = lambda y_hat, y: np.sum(np.round(y_hat.detach().numpy()) == y) / (2 * len(y)) 
    # Accuracy function using numpy. We divide by 2 to avoid double-counting

    if args.optimiser == 'Adadelta':
        opt = torch.optim.Adadelta
    elif args.optimiser == 'Adagrad':
        opt = torch.optim.Adagrad
    elif args.optimiser == 'Adam':
        opt = torch.optim.Adam
    elif args.optimiser == 'AdamW':
        opt = torch.optim.AdamW
    elif args.optimiser == 'Adamax':
        opt = torch.optim.Adamax
    elif args.optimiser == 'ASGD':
        opt = torch.optim.ASGD
    elif args.optimiser == 'NAdam':
        opt = torch.optim.NAdam
    elif args.optimiser == 'RAdam':
        opt = torch.optim.RAdam
    elif args.optimiser == 'RMSprop':
        opt = torch.optim.RMSprop
    elif args.optimiser == 'Rprop':
        opt = torch.optim.Rprop
    elif args.optimiser == 'SGD':
        opt = torch.optim.SGD
    # Optimisers available to be used
    
    
    for s in range(int(args.runs)):
        t1 = time.time()
        seed = seed_list[s]

        with open(train_path + '/diagrams_' + train_dataset_name + '.pickle' , 'rb') as file:
            diagrams_train = pickle.load(file)
        with open(validation_path + '/diagrams_' + validation_dataset_name + '.pickle' , 'rb') as file:
            diagrams_validation = pickle.load(file)
        with open(test_path + '/diagrams_' + test_dataset_name + '.pickle' , 'rb') as file:
            diagrams_test = pickle.load(file)

        circuits_train = PreAlphaLambeq.create_circuits(
            diagrams_train, args.ansatz, args.qn,
            qs, args.n_layers, args.n_single_qubit_params
        )
        circuits_validation = PreAlphaLambeq.create_circuits(
            diagrams_validation, args.ansatz, args.qn,
            qs, args.n_layers, args.n_single_qubit_params
        )
        circuits_test = PreAlphaLambeq.create_circuits(
            diagrams_test, args.ansatz, args.qn,
            qs, args.n_layers, args.n_single_qubit_params
        )
        
        dataset_train = PreAlphaLambeq.create_dataset(
            circuits_train, labels_train, args.batch_size)
        dataset_validation = PreAlphaLambeq.create_dataset(
            circuits_validation, labels_validation, args.batch_size)
        dataset_test = PreAlphaLambeq.create_dataset(
            circuits_test, labels_test, args.batch_size)
        all_circuits = circuits_train + circuits_validation + circuits_test
        
        model = PreAlphaLambeq.create_model(all_circuits)
        
        if torch.cuda.is_available():
            device = 0
        else:
            device = -1
        
        trainer = PreAlphaLambeq.create_trainer(
            model, loss, opt, args.iterations,
            {'acc': acc}, seed = seed, device = device
        )

        trainer.fit(dataset_train, dataset_validation)

        train_loss = trainer.train_epoch_costs
        val_loss = trainer.val_costs
        train_acc = trainer.train_eval_results['acc']
        val_acc = trainer.val_eval_results['acc']
        validation_accuracy_list.append(val_acc)
        weights = []
        for i in range(len(model.weights)):
            weights.append(model.weights.__getitem__(i).tolist())

        vectors_train = PreAlphaLambeq.post_selected_output(
            model, all_circuits)[:len(labels_train)].tolist()
        vectors_validation = PreAlphaLambeq.post_selected_output(
            model, all_circuits)[-len(labels_validation):].tolist()
        vectors_test = PreAlphaLambeq.post_selected_output(
            model, all_circuits)[-len(labels_test):].tolist()
        
        prediction_list = []
        for i,v in enumerate(vectors_test):
            if v[0]>0.5:
                prediction_list.append(0)
            else:
                prediction_list.append(1)

        
        test_acc = acc_np(model(circuits_test), labels_test)

        t2 = time.time()
        time_taken = t2 - t1

        for index, sublist in enumerate(validation_accuracy_list):
            current_max = max(sublist)
            if best_val_accuracy is None or current_max > best_val_accuracy:
                best_val_accuracy = current_max
                best_val_run = index
                iteration_best_val_accuracy = sublist.index(max(sublist))

        json_outputer.save_json_output_run_by_run(
            args, prediction_list, time_taken,
            best_val_acc = best_val_accuracy, best_run = best_val_run,
            iteration_best_val_accuracy = iteration_best_val_accuracy,
            seed_list = seed_list, train_loss = train_loss,
            train_acc = train_acc, val_loss = val_loss,
            val_acc = val_acc, test_acc = test_acc,
            weights = weights, vectors_train = vectors_train,
            vectors_validation = vectors_validation, vectors_test = vectors_test
        )



if __name__ == "__main__":
    main()


