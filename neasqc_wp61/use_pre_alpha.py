import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/models/quantum/pre-alpha/")
import matplotlib.pyplot as plt
import random
import argparse
from collections import Counter
import optimizer
import loader
import dictionary
import sentence
import circuit
import pickle
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################




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
    args = parser.parse_args()
    random.seed(int(args.seed))
    seed_list = random.sample(range(1, 10000000000000), int(args.runs))
    Dftrain, Dftest = loader.createdf(args.train, args.test)
    predictions = [[] for i in range(Dftest.shape[0])]
    for s in range(int(args.runs)):
        seed = seed_list[s]
        Myvocab = loader.getvocabdict(Dftrain, Dftest)
        MyDict = dictionary.QuantumDict(qn=1,
                                        qs=1)
        MyDict.addwords(myvocab=Myvocab)
        MyDict.setvocabparams(seed=seed)


    ##################################################################################################################################################################
    ##################################################################################################################################################################
    ##################################################################################################################################################################    


        def createsentencelist(dftrain, mydict):
            sentences_list = []
            for i, DataInstance in dftrain.iterrows():
                a_sentence = sentence.Sentence(DataInstance,
                                            dataset=True,
                                            dictionary=mydict)
                a_sentence.getqbitcontractions()
                a_sentence.setparamsfrommodel(mydict)
                sentences_list.append(a_sentence)

            return sentences_list

    ##################################################################################################################################################################
    ##################################################################################################################################################################
    ##################################################################################################################################################################

        
        SentencesList = createsentencelist(Dftrain, MyDict)
        par, ix = MyDict.getindexmodelparams()
        myopt = optimizer.ClassicalOptimizer()
        result = myopt.optimizedataset(SentencesList, par, MyDict,
                                    tol=1e-5,
                                    options={'maxiter': int(args.iterations), 'rhobeg': 1},
                                    method=args.optimiser)
        
        MyDict.updateparams(result.x)
        SentencesTest = createsentencelist(Dftest, MyDict)
        for i,a_sentence in enumerate(SentencesTest):
            mycirc = circuit.CircuitBuilder()
            mycirc.createcircuit(a_sentence, dataset=True)
            mycirc.executecircuit()
            probs = [0, 0]
            for sample in mycirc.result:
                state = sample.state.bitstring
                postselectedqubits = ''.join(state[x] for x in range(len(state)) if x != a_sentence.sentencequbit)
                if postselectedqubits == '0' * (mycirc.qlmprogram.qbit_count - 1):
                    if state[a_sentence.sentencequbit] == '0':
                        probs[0] = sample.probability
                    elif state[a_sentence.sentencequbit] == '1':
                        probs[1] = sample.probability
            prob0 = probs[0] / sum(probs)
            if prob0 >= 0.5:
                predictions[i].append(1)
            else:
                predictions[i].append(2)


        with open (
            args.output +
            f'pre_alpha_{args.seed}_{args.optimiser}_{args.iterations}_{args.runs}_run_{s}.pickle', 'wb') as file:
            pickle.dump(predictions, file)

    predictions_majority_vote = []
    for i in range(Dftest.shape[0]):
        c = Counter(predictions[i])
        value, count = c.most_common()[0]
        predictions_majority_vote.append(value)
    with open(args.output + f"pre_alpha_{args.seed}_{args.optimiser}_{args.iterations}_{args.runs}.txt", "w") as output:
        for pred in predictions_majority_vote:
            output.write(f"{pred}\n")
    for i in range(args.runs):
        os.remove(args.output + f"pre_alpha_{args.seed}_{args.optimiser}_{args.iterations}_{args.runs}_run_{i}.pickle")
    # We remove the pickle temporary files when comptutations 
    # are finished .

if __name__ == "__main__":
    main()