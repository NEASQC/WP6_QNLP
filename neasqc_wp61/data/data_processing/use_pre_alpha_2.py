import sys
import os
import argparse
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/quantum/pre_alpha_2/")
from pre_alpha_2 import *
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
        "-s", "--seed", help = "Seed for the initial parameters", type = int, default = 1906)
    parser.add_argument(
        "-op", "--optimiser", help = "Optimiser to use", type = str, default = 'Adam')
    parser.add_argument(
        "-i", "--iterations", help = "Number of iterations of the optimiser", type = int, default = 100)
    parser.add_argument(
        "-r", "--runs", help = "Number of runs", type = int, default = 2)
    parser.add_argument(
        "-tr", "--train", help = "Directory of the train dataset", type = str, default = './../datasets/toy_datasets/binary_toy_train.tsv')
    parser.add_argument(
        "-val", "--validation", help = "Directory of the validation dataset", type = str, default = './../datasets/toy_datasets/binary_toy_validation.tsv')
    parser.add_argument(
        "-te", "--test", help = "Directory of the test datset", type = str, default = './../datasets/toy_datasets/binary_toy_test.tsv')
    parser.add_argument(
        "-o", "--output", help = "Output directory with the predictions", type = str, default = './../../benchmarking/results/raw/')
    parser.add_argument(
        "-an", "--ansatz", help = "Ansatz to be used in quantum circuits", type = str, default = 'IQP')
    parser.add_argument(
        "-qn", "--qn", help = "Number of qubits per NOUN type", type = int, default = 1)
    parser.add_argument(
        "-nl", "--n_layers", help = "Number of layers for the circuits", type = int, default = 1)
    parser.add_argument(
        "-np", "--n_single_qubit_params", help = "Number of parameters per qubit", type = int, default = 1)
    parser.add_argument(
        "-b", "--batch_size", help = "Batch size used for traininig the model", type = int, default = 10)
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
    model_name = 'pre_alpha_2'
    timestr = time.strftime("%Y%m%d-%H%M%S")
    json_outputer = JsonOutputer(model_name, timestr, args.output)
    # The number of qubits per sentence is pre-defined as we still need 
    # to improve our model
    qs = 1 
    name_file = args.output + f"pre_alpha_2_{args.seed}_{args.optimiser}_"\
        f"{args.iterations}_{args.runs}_{args.ansatz}_{args.qn}_"\
        f"{qs}_{args.n_layers}_{args.n_single_qubit_params}_{args.batch_size}"
    # Name of the file to store the results. 

    random.seed(args.seed)
    seed_list = random.sample(range(1, 10000000000000), int(args.runs))
    # Set the random seed. 


    labels_train = PreAlpha2.load_dataset(args.train)[1]
    labels_validation = PreAlpha2.load_dataset(args.validation)[1]
    labels_test = PreAlpha2.load_dataset(args.test)[1]
    labels_train_int = torch.argmax(
        torch.tensor(labels_train), dim=1).tolist()
    labels_val_int = torch.argmax(
        torch.tensor(labels_validation), dim=1).tolist()
    labels_test_int = torch.argmax(
        torch.tensor(labels_test), dim=1).tolist()


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
        print(f'RUN == {s}')
        print('##########################\n\n')
        t1 = time.time()
        seed = seed_list[s]

        with open(train_path + '/diagrams_' + train_dataset_name + '.pickle' , 'rb') as file:
            diagrams_train = pickle.load(file)
        with open(validation_path + '/diagrams_' + validation_dataset_name + '.pickle' , 'rb') as file:
            diagrams_validation = pickle.load(file)
        with open(test_path + '/diagrams_' + test_dataset_name + '.pickle' , 'rb') as file:
            diagrams_test = pickle.load(file)

        circuits_train = PreAlpha2.create_circuits(
            diagrams_train, args.ansatz, args.qn,
            qs, args.n_layers, args.n_single_qubit_params
        )
        circuits_validation = PreAlpha2.create_circuits(
            diagrams_validation, args.ansatz, args.qn,
            qs, args.n_layers, args.n_single_qubit_params
        )
        circuits_test = PreAlpha2.create_circuits(
            diagrams_test, args.ansatz, args.qn,
            qs, args.n_layers, args.n_single_qubit_params
        )
        
        dataset_train = PreAlpha2.create_dataset(
            circuits_train, labels_train, args.batch_size)
        dataset_validation = PreAlpha2.create_dataset(
            circuits_validation, labels_validation, args.batch_size)
        dataset_test = PreAlpha2.create_dataset(
            circuits_test, labels_test, args.batch_size)
        all_circuits = circuits_train + circuits_validation + circuits_test
        
        model = PreAlpha2.create_model(all_circuits)
        
        if torch.cuda.is_available():
            device = 0
        else:
            device = -1
        
        trainer = PreAlpha2.create_trainer(
            model, loss, opt, args.iterations,
            {'acc': acc}, seed = seed, device = device
        )

        trainer.fit(dataset_train, dataset_validation)

        train_loss = trainer.train_epoch_costs
        val_loss = trainer.val_costs
        train_acc = trainer.train_eval_results['acc']
        val_acc = trainer.val_eval_results['acc']
        validation_accuracy_list.append(val_acc)
        train_probabilities = trainer.train_probabilities
        val_probabilities = trainer.val_probabilities
        test_probabilities = PreAlpha2.post_selected_output(
            model, circuits_test)
        train_predictions = trainer.train_predictions
        val_predictions = trainer.val_predictions
        test_predictions = [
            int(torch.argmax(tensor)) for tensor in test_probabilities]
        test_probabilities = test_probabilities.tolist()
        weights = []
        for i in range(len(model.weights)):
            weights.append(model.weights.__getitem__(i).tolist())

        
        ####### Compute test accuracy 
        test_acc = acc_np(model(circuits_test), labels_test)

        ####### Compute time taken 
        t2 = time.time()
        time_taken = t2 - t1

        for index, sublist in enumerate(validation_accuracy_list):
            current_max = max(sublist)
            if best_val_accuracy is None or current_max > best_val_accuracy:
                best_val_accuracy = current_max
                best_val_run = index
                iteration_best_val_accuracy = sublist.index(max(sublist))

        
        json_outputer.save_json_output_run_by_run(
            args,  time_taken, labels_train_int,
            labels_val_int, labels_test_int,
            best_val_acc = best_val_accuracy, best_run = best_val_run,
            iteration_best_val_accuracy = iteration_best_val_accuracy,
            seed_list = seed_list, train_predictions = train_predictions,
            val_predictions = val_predictions,
            test_predictions = test_predictions,
            train_loss = train_loss, train_acc = train_acc,
            val_loss = val_loss,
            val_acc = val_acc, test_acc = test_acc,
            weights = weights, train_probabilities = train_probabilities,
            val_probabilities = val_probabilities,
            test_probabilities = test_probabilities
        )



if __name__ == "__main__":
    main()