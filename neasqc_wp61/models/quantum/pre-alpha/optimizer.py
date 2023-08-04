import circuit
import scipy.optimize
import math
import numpy as np
from sentence import createsentencelist

def compute_predictions(Sentences):
    predictions = []
    for i,a_sentence in enumerate(Sentences):
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
            predictions.append(0)
        else:
            predictions.append(1)
    return predictions



class ClassicalOptimizer:

    def __init__(self, optimizer='cobyla', tol=0.1, maxiter=200):
        self.optimizer = optimizer
        self.tol = tol
        self.maxiter = maxiter
        self.itercost=[]
        self.iteration=0

    def optimizesentence(self, mySentence, tol=1e-2, options={'maxiter': 50}, method="COBYLA"):
        params0 = mySentence.getparameters()
        flat_params0 = [item for sublist in params0 for item in sublist]
        result = scipy.optimize.minimize(self.cost, flat_params0,
                                                    args=(mySentence),
                                                    tol=tol,
                                                    options=options,
                                                    method=method)
        return result


    def reshapeparams(self, parameters, mySentence):
        originalparams = mySentence.getparameters()
        shapedparams = []
        iparam=0
        for word in originalparams:
            iwparam=0
            wordparams = []
            while iwparam < len(word):
                wordparams.append(parameters[iparam])
                iparam+=1
                iwparam+=1
            shapedparams.append(wordparams)
        return shapedparams



    def datasetcost(
            self, parameters, SentencesList, mydict):
        cost=0
        par, ix = mydict.getindexmodelparams()
        for mysentence in SentencesList:
            shapedparams = []
            for word, cat in zip(mysentence.sentence.lower().split(' '), mysentence.sentencestructure.split(',')):
                wordparams =[parameters[i] for i in ix[(word, cat)]]
                shapedparams.append(wordparams)


            mysentence.setsentenceparamsfromlist(shapedparams)
            mycirc = circuit.CircuitBuilder()
            mycirc.createcircuit(mysentence, dataset=True)
            mycirc.executecircuit()
            probs = [0,0]
            for sample in mycirc.result:
                state = sample.state.bitstring
                postselectedqubits = ''.join(state[x] for x in range(len(state)) if x != mysentence.sentencequbit)
                if postselectedqubits == '0' * (mycirc.qlmprogram.qbit_count - 1):
                    if state[mysentence.sentencequbit] == '0':
                        probs[0] = sample.probability
                        #print(
                        #    "State %s: probability %s, amplitude %s" % (sample.state, sample.probability, sample.amplitude))
                    elif state[mysentence.sentencequbit] == '1':
                        probs[1] = sample.probability
                        #print(
                        #    "State %s: probability %s, amplitude %s" % (
                         #   sample.state, sample.probability, sample.amplitude))
            prob0 = probs[0] / sum(probs)
            prob1 = probs[1] / sum(probs)
            if prob0 == 1:
                prob0 = 0.9999999999
            if prob1 == 1:
                prob1 == 0.9999999999
            if mysentence.label == 0:
                cost+= -math.log(prob0)/-(math.log(prob0) + math.log(1-prob0))
                #print(cost)
            elif mysentence.label == 1:
                cost+= -math.log(1 - prob0)/-(math.log(prob0) + math.log(1-prob0))
                #print(cost)
        self.itercost.append(cost/len(SentencesList))
        self.iteration+=1

        #    print('iteration {}'.format(self.iteration), '\n Cost: {}'.format((cost / len(SentencesList))))
        return cost/len(SentencesList)


    def optimizedataset(self, sentencelist, params0, mydict, options={'maxiter':3}, method="COBYLA"):
        params0 = np.array(params0)
        result = scipy.optimize.minimize(self.datasetcost, params0,
                                                    args=(sentencelist,mydict),
                                                    options=options,
                                                    method=method)
        return result