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
import circuit
import pickle
import time 
from save_json_output import JsonOutputer

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
        "-val", "--validation", help = "Path of the validation dataset", type = str)
    parser.add_argument(
        "-te", "--test", help = "Path of the test dataset", type = str)
    parser.add_argument(
        "-o", "--output", help = "Output directory with the predictions", type = str)
    args = parser.parse_args()
    random.seed(int(args.seed))
    seed_list = random.sample(range(1, int(2**32 -1)), int(args.runs))
    Dftrain, Dfval, Dftest = loader.createdf(args.train, args.validation, args.test)


    validation_accuracy_list = []
    best_val_accuracy = None
    best_val_run = None
    model_name = 'pre_alpha'
    timestr = time.strftime("%Y%m%d-%H%M%S")
    json_outputer = JsonOutputer(model_name, timestr, args.output)

    for s in range(int(args.runs)):
        t1 = time.time()
        seed = seed_list[s]
        Myvocab = loader.getvocabdict(Dftrain, Dfval, Dftest)
        MyDict = dictionary.QuantumDict(qn=1, qs=1)
        MyDict.addwords(myvocab=Myvocab)
        MyDict.setvocabparams(seed=seed)

    
        SentencesTrain = createsentencelist(Dftrain, MyDict)
        SentencesVal = createsentencelist(Dfval, MyDict)
        SentencesTest = createsentencelist(Dftest, MyDict)

        par, ix = MyDict.getindexmodelparams()
        myopt = optimizer.ClassicalOptimizer()
        result = myopt.optimizedataset(
            SentencesTrain, par, MyDict,
            options={'maxiter': int(args.iterations), 'disp' : False},
            method=args.optimiser)

        val_loss, val_accuracy = myopt.compute_loss_accuracy_iterations(MyDict, SentencesVal)[:2]
        validation_accuracy_list.append(val_accuracy)
        test_loss, test_accuracy, prediction_list = myopt.compute_loss_accuracy_iterations(MyDict, SentencesTest)
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
            seed_list = seed_list, 
            train_loss = myopt.cobyla_train_loss,
            train_acc = myopt.cobyla_train_accuracy, val_loss = val_loss,
            val_acc = val_accuracy,
            test_acc = test_accuracy[-1], weights = myopt.results_weights
        )





if __name__ == "__main__":
    main()
