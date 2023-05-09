import sys
import os
sys.path.append("./models/quantum/pre-alpha/")
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import Counter
import optimizer
import loader
import dictionary
import sentence
import circuit
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", help = "Seed for the initial parameters")
    parser.add_argument("-op", "--optimiser", help = "Optimiser to use")
    parser.add_argument("-i", "--iterations", help = "Number of iterations of the optimiser")
    parser.add_argument("-r", "--runs", help = "Number of runs")
    parser.add_argument("-tr", "--train", help = "Directory of the train dataset")
    parser.add_argument("-te", "--test", help = "Directory of the test datset")
    parser.add_argument("-o", "--output", help = "Output file with the predictions")
    args = parser.parse_args()
    np.random.seed(args.seed)
    seed_list = np.random.randint(0, 1e10, (1, args.runs))
    Dftrain, Dftest = loader.createdf(args.train, args.test)
    predictions = [[] for i in range(Dftest.shape[0])]
    for i in range(args.runs):
        seed = seed_list[i]
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
                                    options={'maxiter': args.iterations, 'rhobeg': 1},
                                    method=args.optimiser)
        
        MyDict.updateparams(result.x)
        SentencesTest = createsentencelist(Dftest, MyDict)
        for i,a_sentence in SentencesTest:
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
                predictions.append(1)
            else:
                predictions.append(2)
    
            predictions[i].append(predictions)

    predictions_majority_vote = []
    for i in range(Dftest.shape[0]):
        c = Counter(predictions[i])
        value, count = c.most_common()[0]
        predictions_majority_vote.append(value)
    with open(args.output, "w") as output:
        for pred in predictions:
            output.write(f"{pred}\n")

    if __name__ == "__main__":
        main()
