import sys
import os

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/quantum/pre-alpha/")
import matplotlib.pyplot as plt
import random
import argparse
from collections import Counter
import optimizer
import loader
import dictionary
from sentence import createsentencelist
from optimizer import compute_predictions
import circuit
import pickle
import time 
from save_json_output import save_json_output

##########################
########################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################




def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s", "--seed", help = "Seed for the initial parameters", type = int)
    parser.add_argument(
        "-op", "--optimiser", help = "Optimiser to use", type = str)
    # The optimisers that can be used are the ones appearing in method section in 
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    # with the excepction of : Newton-CG, TNC, dogleg, trust-ncg, trust-exact, trust-kyrlov
    parser.add_argument(
        "-i", "--iterations", help = "Maximum iterations of the optimiser", type = int)
    parser.add_argument(
        "-r", "--runs", help = "Number of runs", type = int)
    parser.add_argument(
        "-tr", "--train", help = "Path of the train dataset", type = str)
    parser.add_argument(
        "-te", "--test", help = "Path of the test datset", type = str)
    parser.add_argument(
        "-o", "--output", help = "Output directory with the predictions", type = str)
    args = parser.parse_args()
    random.seed(int(args.seed))
    seed_list = random.sample(range(1, 10000000000000), int(args.runs))
    Dftrain, Dftest = loader.createdf(args.train, args.test)
    train_truth_value = Dftrain['truth_value'].tolist()
    test_truth_value = Dftest['truth_value'].tolist()
    predictions = [[] for i in range(Dftest.shape[0])]
    cost = []
    accuracies_train = [] 
    accuracies_test = [] 
    weights = []
    times_list = []

    test_loss_list = []
    test_acc_list = []
    train_acc_list = []

    for s in range(int(args.runs)):
        t1 = time.time()
        seed = seed_list[s]
        Myvocab = loader.getvocabdict(Dftrain, Dftest)
        MyDict = dictionary.QuantumDict(qn=1, qs=1)
        MyDict.addwords(myvocab=Myvocab)
        MyDict.setvocabparams(seed=seed)

        

        SentencesList = createsentencelist(Dftrain, MyDict)
        SentencesTest = createsentencelist(Dftest, MyDict)

        par, ix = MyDict.getindexmodelparams()
        myopt = optimizer.ClassicalOptimizer()
        result = myopt.optimizedataset(
            SentencesList, SentencesTest, par, MyDict,
            options={'maxiter': int(args.iterations), 'disp' : False},
            method=args.optimiser)
        cost.append(myopt.itercost)

        test_loss_list.append(myopt.test_loss_list)
        test_acc_list.append(myopt.test_acc_list)
        train_acc_list.append(myopt.train_acc_list)

        MyDict.updateparams(result.x)
        weights.append(result.x.tolist())

        SentencesTest = createsentencelist(Dftest, MyDict)
        SentencesTrain = createsentencelist(Dftrain, MyDict)
        predictions_iteration_test = compute_predictions(SentencesTest)
        predictions_iteration_train = compute_predictions(SentencesTrain)
        true_values_train = 0 
        true_values_test = 0
        for i,pred in enumerate(predictions_iteration_test):
            predictions[i].append(pred)
            if pred == test_truth_value[i]:
                true_values_test += 1
        for i,pred in enumerate(predictions_iteration_train):
            if pred == train_truth_value[i]:
                true_values_train += 1
        
        accuracies_test.append(true_values_test/len(predictions_iteration_test))
        accuracies_train.append(true_values_train/len(predictions_iteration_train))
        with open (
            args.output +
            f'pre_alpha_{args.seed}_{args.optimiser}_{args.iterations}_{args.runs}_run_{s}.pickle', 'wb') as file:
            pickle.dump(predictions, file)
        ## We use pickle to store the temporary results 
        t2 = time.time()
        times_list.append(t2 - t1)
    best_final_accuracy = max(accuracies_train)
    best_run = accuracies_train.index(best_final_accuracy)
    predictions_majority_vote = []
    for i in range(Dftest.shape[0]):
        c = Counter(predictions[i])
        value, count = c.most_common()[0]
        predictions_majority_vote.append(value)

    with open(args.output + f"pre_alpha_{args.seed}_{args.optimiser}_{args.iterations}_{args.runs}_predictions.txt", "w") as output:
        for pred in predictions_majority_vote:
            output.write(f"{pred}\n")
    # We store the results in Tilde's format 

    for i in range(args.runs):
        os.remove(args.output + f"pre_alpha_{args.seed}_{args.optimiser}_{args.iterations}_{args.runs}_run_{i}.pickle")
    # We remove the pickle temporary files when comptutations 
    # are finished .


    save_json_output(
        'pre_alpha', args, predictions_majority_vote,
        t2 - t1, args.output, best_final_val_acc = best_final_accuracy,
        best_run = best_run, seed_list = seed_list, 
        final_val_acc = accuracies_test, final_train_acc = accuracies_train,
        train_loss = cost, weights = weights,

        train_acc = train_acc_list, val_acc = test_acc_list,
        val_loss = test_loss_list
    )



if __name__ == "__main__":
    main()
