import sys
import os
import argparse
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/quantum/pre-alpha-lambeq/")
from PreAlphaLambeq import *
from collections import Counter
import random
import pickle

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--seed", help = "Seed for the initial parameters", type = int)
    parser.add_argument(
        "-op", "--optimiser", help = "Optimiser to use", type = callable)
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
        "-qs", "--qs", help = "Number of qubits per sentence type", type = int)
    parser.add_argument(
        "-nl", "--n_layers", help = "Number of layers for the circuits", type = int)
    parser.add_argument(
        "-np", "--n_single_qubit_params", help = "Number of parameters per qubit", type = int)
    args = parser.parse_args()
    
    seed_list = random.sample(range(1, 10000000000000), int(args.runs))
    sentences_train = PreAlphaLambeq.load_dataset(args.train)[0]
    labels_train = PreAlphaLambeq.load_dataset(args.train)[1]
    sentences_test = PreAlphaLambeq.load_dataset(args.test)[0]
    labels_test = PreAlphaLambeq.load_dataset(args.test)[1]
    
    predictions = [[] for i in range(len(sentences_test))]
    def loss(y_hat, y):
        return torch.nn.functional.binary_cross_entropy(
            y_hat, y
        )
    for s in range(int(args.runs)):
        seed = seed_list[s]

        diagrams_train = PreAlphaLambeq.create_diagrams(sentences_train)
        diagrams_test = PreAlphaLambeq.create_diagrams(sentences_test)
        circuits_train = PreAlphaLambeq.create_circuits(
            diagrams_train, args.ansatz, args.qn,
            args.qs, args.n_layers, args.n_single_qubit_params
        )
        circuits_test = PreAlphaLambeq.create_circuits(
            diagrams_train, args.ansatz, args.qn,
            args.qs, args.n_layers, args.n_single_qubit_params
        )
        dataset_train = PreAlphaLambeq.create_dataset(circuits_train, labels_train)
        dataset_test = PreAlphaLambeq.create_dataset(
            circuits_test, labels_test
        )
        all_circuits = circuits_train + circuits_test
        
        model = PreAlphaLambeq.create_model(all_circuits)
        trainer = PreAlphaLambeq.create_trainer(
            model, loss, torch.optim.Adam, args.iterations,
            seed = seed
        )
        trainer.fit(dataset_train, dataset_test)
    
        for i,circuit in enumerate(circuits_test):
            output = PreAlphaLambeq.post_selected_output(
            circuit, model
            )
            predictions[i].append(
                PreAlphaLambeq.predicted_label(output)
            )
        with open (
            args.output +
            f'pre_alpha_lambeq_{args.seed}_{args.optimiser}_{args.iterations}_{args.runs}_run_{s}.pickle', 'wb') as file:
            pickle.dump(predictions, file)
    
    predictions_majority_vote = []
    for i in range(len(sentences_test)):
        c = Counter(predictions[i])
        value, count = c.most_common()[0]
        predictions_majority_vote.append(value)

    with open(args.output + f"pre_alpha_lambeq_{args.seed}_{args.optimiser}_{args.iterations}_{args.runs}.txt", "w") as output:
            for pred in predictions_majority_vote:
                output.write(f"{pred}\n")
    for i in range(args.runs):
        os.remove(args.output + f"pre_alpha_lambeq_{args.seed}_{args.optimiser}_{args.iterations}_{args.runs}_run_{i}.pickle")
    # We remove the pickle temporary files when comptutations 
    # are finished .

if __name__ == "__main__":
    main()


