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
import git 
import time 
import numpy as np 
from statistics import mean

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

    qs = 1
    # The number of qubits per sentence is pre-defined as we still need 
    # to improve our model
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
    vectors_train = [[[], []] for i in range(len(sentences_train))]
    vectors_test = [[[], []] for i in range(len(sentences_test))]
    cost = []
    weights = []
    # Lists to store the predictions, probability vectors and costs

    def loss(y_hat, y):
        return torch.nn.functional.binary_cross_entropy(
            y_hat, y
        )
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
    
    t1 = time.time()
    for s in range(int(args.runs)):
        seed = seed_list[s]

        diagrams_train = PreAlphaLambeq.create_diagrams(sentences_train)
        diagrams_test = PreAlphaLambeq.create_diagrams(sentences_test)
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
        trainer = PreAlphaLambeq.create_trainer(
            model, loss, opt, args.iterations,
            seed = seed
        )
        trainer.fit(dataset_train, dataset_test)
        cost.append(trainer.train_epoch_costs)
        weights_run = []
        print('length model weights : ', len(model.weights))
        for i in range(len(model.weights)):
            weights_run.append(model.weights.__getitem__(i).tolist())
        weights.append(weights_run)
    
        for i,circuit in enumerate(circuits_test):
            output = PreAlphaLambeq.post_selected_output(
            circuit, model
            )
            vectors_test[i][0].append(float(output[0]))
            vectors_test[i][1].append(float(output[1]))
            predictions[i].append(
                PreAlphaLambeq.predicted_label(output)
            )
        for i,circuit in enumerate(circuits_train):
            output = PreAlphaLambeq.post_selected_output(
            circuit, model
            )
            vectors_train[i][0].append(float(output[0]))
            vectors_train[i][1].append(float(output[1]))
        # We compute the class predictions for the test dataset 
        # and the vectors for both training and test datatset

        with open (name_file + f'_predictions_run_{s}.pickle', 'wb') as file:
            pickle.dump(predictions, file)
        with open (name_file + f'_vectors_test_run_{s}.pickle', 'wb') as file:
            pickle.dump(vectors_test, file)
        with open (name_file + f'_vectors_train_run_{s}.pickle', 'wb') as file:
            pickle.dump(vectors_train, file)
        # We store temporary predictions and vectors

        
    t2 = time.time()
    predictions_majority_vote = []
    vectors_test_mean = [[] for i in range(len(sentences_test))]
    vectors_train_mean = [[] for i in range(len(sentences_train))]


    for i in range(len(sentences_test)):
        c = Counter(predictions[i])
        value, count = c.most_common()[0]
        predictions_majority_vote.append(value)
        vectors_test_mean[i].append(mean(vectors_test[i][0]))
        vectors_test_mean[i].append(mean(vectors_test[i][1]))
    
    for i in range(len(sentences_train)):
        vectors_train_mean[i].append(mean(vectors_train[i][0]))
        vectors_train_mean[i].append(mean(vectors_train[i][1]))        
    
    with open(name_file + "_predictions.txt", "w") as output:
            for pred in predictions_majority_vote:
                output.write(f"{pred}\n")
    with open (name_file + '_vectors_test.json', 'w') as file:
            json.dump(vectors_test_mean, file)
    with open (name_file + '_vectors_train.json', 'w') as file:
            json.dump(vectors_train_mean, file)
    # We store the final results of our experiments. 
    # The predictions will be selected with majority vote.
    # For the probability vectors we will use the mean values

    for i in range(args.runs):
        os.remove(name_file + f'_predictions_run_{i}.pickle')
        os.remove(name_file + f'_vectors_test_run_{i}.pickle')
        os.remove(name_file + f'_vectors_train_run_{i}.pickle')
    # We remove the pickle temporary files when comptutations 
    # are finished .

    ############################################################
    ############## JSON output #################################
    ############################################################

    output = {}

    # 1. Commit HASH 

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    output['hash'] = sha

    # 2. Input arguments 

    input_args = {
        'seed' : args.seed, 'optimiser' : args.optimiser,
        'iterations' : args.iterations, 'runs' : args.runs,
        'train' : args.train, 'test' : args.test, 'output' : args.output,
        'ansatz' : args.ansatz, 'qn' : args.qn, 'nl' : args.n_layers, 
        'np'  :args.n_single_qubit_params
    }
    output['input_args'] = input_args

    # 3. Predictions 

    output['predictions'] = predictions_majority_vote

    # 4. Loss function 

    arrays = [np.array(x) for x in cost]
    mean_cost = [np.mean(k) for k in zip(*arrays)]
    output['cost'] = mean_cost

    # 5. Time taken 

    output['time'] = t2 -t1 

    # 6. Model parameters 

    arrays = [np.array(x) for x in weights]
    mean_weights = [np.mean(k) for k in zip(*arrays)]
    output['weights'] = mean_weights

    # Save the results 
    timestr = time.strftime("%Y%m%d-%H%M%S")
    with open (args.output + f'pre_alpha_lambeq_{timestr}.json', 'w') as file:
        json.dump(output, file) 

if __name__ == "__main__":
    main()


