import circuit
import scipy.optimize
import sentence
import math
import numpy as np


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




    def cost(self, parameters, mySentence):
        shapedparams = self.reshapeparams(parameters, mySentence)
        mySentence.setsentenceparameters(randompar=False, params=shapedparams)
        myCircBuilder = circuit.CircuitBuilder()
        myCircBuilder.createcircuit(mySentence)
        myCircBuilder.executecircuit()
        label=mySentence.label
        probs = []
        for sample in myCircBuilder.result:
            state = sample.state.bitstring
            postselectedqubits = ''.join(state[x] for x in range(len(state)) if x != mySentence.sentencequbit)
            if postselectedqubits == '0' * (myCircBuilder.qlmprogram.qbit_count - 1):
                probs.append(sample.probability)
                #print("State %s: probability %s, amplitude %s" % (sample.state, sample.probability, sample.amplitude))
        prob0 = probs[0] / sum(probs)
        prob1 = probs[1] / sum(probs)
        if label==0:
            costval = -math.log(prob0)
        elif label==1:
            costval = -math.log(1-prob0)
        self.iteration += 1
        if self.iteration % 10 == 0:
            print('iteration {}'.format(self.iteration), '\n Cost: {}'.format(costval))
        self.itercost.append(costval)
        return costval




    def datasetcost(self, parameters, SentencesList, mydict):
        cost=0
        par, ix = mydict.getindexmodelparams()
        for mysentence in SentencesList:
            shapedparams = []
            for word, cat in zip(mysentence.sentence.split(' '), mysentence.sentencestructure.split('-')):
                wordparams =[parameters[i] for i in ix[(word, cat)]]
                shapedparams.append(wordparams)

            mysentence.setsentenceparamsfromlist(shapedparams)
            mycirc = circuit.CircuitBuilder()
            mycirc.createcircuit(mysentence, dataset=True)
            mycirc.executecircuit()
            probs = [0,0]
            #print('sentence: {}'.format(mysentence.sentence))
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

            if mysentence.label == 0:
                cost+= -math.log(prob0)
                #print(cost)
            elif mysentence.label == 1:
                cost+= -math.log(1 - prob0)
                #print(cost)

        self.itercost.append(cost/len(SentencesList))
        self.iteration+=1
        if self.iteration % 10 == 0:
            print('iteration {}'.format(self.iteration), '\n Cost: {}'.format((cost / len(SentencesList))))
        return cost/len(SentencesList)


    def optimizedataset(self, sentencelist, params0, mydict, tol=1e-5, options={'maxiter':300, 'rhobeg': 1.5}, method="COBYLA"):
        params0 = np.array(params0)
        result = scipy.optimize.minimize(self.datasetcost, params0,
                                                    args=(sentencelist,mydict),
                                                    tol=tol,
                                                    options=options,
                                                    method=method)
        return result

