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
from statistics import mean
from save_json_output import save_json_output

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
    args = parser.parse_args()
    train_dataset_name = os.path.basename(args.train)
    train_dataset_name = os.path.splitext(train_dataset_name)[0]
    test_dataset_name = os.path.basename(args.test)
    test_dataset_name = os.path.splitext(test_dataset_name)[0]
    train_path = os.path.dirname(args.train)
    test_path = os.path.dirname(args.test)
    # The number of qubits per sentence is pre-defined as we still need 
    # to improve our model
    qs = 1 
    name_file = args.output + f"pre_alpha_lambeq_{args.seed}_{args.optimiser}_"\
        f"{args.iterations}_{args.runs}_{args.ansatz}_{args.qn}_"\
        f"{qs}_{args.n_layers}_{args.n_single_qubit_params}"
    # Name of the file to store the results. 

    random.seed(args.seed)
    seed_list = random.sample(range(1, 10000000000000), int(args.runs))
    # Set the random seed. 

    sentences_train = PreAlphaLambeq.load_dataset(args.train)[0]
    labels_train = PreAlphaLambeq.load_dataset(args.train)[1]
    sentences_test = PreAlphaLambeq.load_dataset(args.test)[0]
    labels_test = PreAlphaLambeq.load_dataset(args.test)[1]
    # Load sentences and labels
    
    predictions = [[] for i in range(len(sentences_test))]
    vectors_train_list = []
    vectors_test_list = []
    cost_train = []
    cost_test = []
    weights = []
    accuracies_train = [] 
    accuracies_test = []
    time_list = []
    # Lists to store the different quantities of our output

    def loss(y_hat, y):
        return torch.nn.functional.binary_cross_entropy(
            y_hat, y
        )
    def acc(y_hat, y):
        return (torch.argmax(y_hat, dim=1) ==
            torch.argmax(y, dim=1)).sum().item()/len(y)
    # Default loss function to use

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
        with open(test_path + '/diagrams_' + test_dataset_name + '.pickle' , 'rb') as file:
            diagrams_test = pickle.load(file)

        circuits_train = PreAlphaLambeq.create_circuits(
            diagrams_train, args.ansatz, args.qn,
            qs, args.n_layers, args.n_single_qubit_params
        )
        circuits_test = PreAlphaLambeq.create_circuits(
            diagrams_test, args.ansatz, args.qn,
            qs, args.n_layers, args.n_single_qubit_params
        )
        
        dataset_train = PreAlphaLambeq.create_dataset(
            circuits_train, labels_train)
        dataset_test = PreAlphaLambeq.create_dataset(
            circuits_test, labels_test
        )
        all_circuits = circuits_train + circuits_test
        
        model = PreAlphaLambeq.create_model(all_circuits)
        
        if torch.cuda.is_available():
            device = 0
        else:
            device = -1
        
        trainer = PreAlphaLambeq.create_trainer(
            model, loss, opt, args.iterations,
            {'acc': acc}, seed = seed, device = device
        )
        
        trainer.fit(dataset_train, dataset_test)

        cost_train.append(trainer.train_epoch_costs)
        cost_test.append(trainer.val_costs)
        accuracies_train.append(trainer.train_results['acc'])
        accuracies_test.append(trainer.val_results['acc'])
        weights_run = []
        for i in range(len(model.weights)):
            weights_run.append(model.weights.__getitem__(i).tolist())
        weights.append(weights_run)

        vectors_train = []
        vectors_test = []
        for i,circuit in enumerate(circuits_test):
            output = PreAlphaLambeq.post_selected_output(
            circuit, model
            )
            vector = output.flatten().tolist()
            vectors_test.append(vector)
            predictions[i].append(
                PreAlphaLambeq.predicted_label(output)
            )
        for i,circuit in enumerate(circuits_train):
            output = PreAlphaLambeq.post_selected_output(
            circuit, model
            )
            vector = output.flatten().tolist()
            vectors_train.append(vector)
        # We compute the class predictions for the test dataset 
        # and the vectors for both training and test datatset

        with open (name_file + f'_predictions_run_{s}.pickle', 'wb') as file:
            pickle.dump(predictions, file)
        # We store temporary predictions and vectors

        vectors_train_list.append(vectors_train)
        vectors_test_list.append(vectors_test)
        t2 = time.time()
        time_list.append(t2 - t1)
        

    best_accuracy = None
    best_run = None

    for index, sublist in enumerate(accuracies_test):
        current_max = max(sublist)
        if best_accuracy is None or current_max > best_accuracy:
            best_accuracy = current_max
            best_run = index
    # We compute the best accuracy and the best run 

    
    predictions_majority_vote = []



    for i in range(len(sentences_test)):
        c = Counter(predictions[i])
        value, count = c.most_common()[0]
        predictions_majority_vote.append(value)

    
    with open(name_file + "_predictions.txt", "w") as output:
            for pred in predictions_majority_vote:
                output.write(f"{pred}\n")
    # We store the final results of our experiments. 
    # The predictions will be selected with majority vote.
    # For the probability vectors we will use the mean values

    for i in range(args.runs):
        os.remove(name_file + f'_predictions_run_{i}.pickle')

    # We remove the pickle temporary files when comptutations 
    # are finished .

    save_json_output(
    'pre_alpha_lambeq', args, predictions_majority_vote,
    time_list, args.output, best_val_acc = best_accuracy,
    best_run = best_run, seed_list = seed_list,
    val_acc = accuracies_test, val_loss = cost_test,
    train_acc = accuracies_train, train_loss = cost_train,
    weights = weights, vectors_train = vectors_train_list,
    vectors_test = vectors_test_list)
    # We save the json output 


if __name__ == "__main__":
    main()


