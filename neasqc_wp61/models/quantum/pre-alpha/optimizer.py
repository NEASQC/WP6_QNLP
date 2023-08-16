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

        self.test_loss_list = []
        self.test_acc_list = []
        self.train_acc_list = []

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
            self, parameters, SentencesList, SentencesListTest, mydict):
        cost=0
        correct_predictions = 0

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
            # if prob0 == 1:
            #     prob0 = 0.9999999999
            # if prob1 == 1:
            #     prob1 == 0.9999999999
            # if mysentence.label == 0:
            #     cost+= -math.log(prob0)/-(math.log(prob0) + math.log(1-prob0))
            #     #print(cost)
            # elif mysentence.label == 1:
            #     cost+= -math.log(1 - prob0)/-(math.log(prob0) + math.log(1-prob0))
            #     #print(cost)

            # Compute the cost
            # prob0 + prob1 = 1 normally, to check if its the case
            epsilon = 1e-7 # to avoid log(0)

            # BCE Loss
            #cost += -(mysentence.label * math.log(prob1+epsilon) + (1 - mysentence.label) * math.log(prob0+epsilon))

            #Normalized BCE Loss
            # From https://stats.stackexchange.com/questions/499423/normalized-cross-entropy
            # Here p = 0.5 for a balanced dataset
            # So we can simplify the formula
            p = 0.5
            cost+= -(mysentence.label * math.log(prob1+epsilon) + (1 - mysentence.label) * math.log(prob0+epsilon))/-(math.log(p))

            # Compute the accuracy
            # prob0 + prob1 = 1 normally, to check if its the case
            if prob0 >= 0.5 and mysentence.label == 0 or prob0 < 0.5 and mysentence.label == 1:
                correct_predictions += 1
            
    
        self.itercost.append(cost/len(SentencesList))
        self.iteration+=1

        accuracy = correct_predictions / len(SentencesList)
        test_loss, test_acc = self.compute_test_logs(parameters, SentencesListTest, mydict)

        self.test_loss_list.append(test_loss)
        self.test_acc_list.append(test_acc)
        self.train_acc_list.append(accuracy)

        #    print('iteration {}'.format(self.iteration), '\n Cost: {}'.format((cost / len(SentencesList))))
        return cost/len(SentencesList)


    def optimizedataset(self, sentencelist, SentencesListTest, params0, mydict, options={'maxiter':3}, method="COBYLA"):
        params0 = np.array(params0)
        result = scipy.optimize.minimize(self.datasetcost, params0,
                                                    args=(sentencelist, SentencesListTest,mydict),
                                                    options=options,
                                                    method=method)
        return result
    
    
    def compute_test_logs(self, parameters, SentencesListTest, mydict):
        #Compute the test loss and accuracy
        par, ix = mydict.getindexmodelparams()

        correct_predictions = 0
        cost = 0

        for mysentence in SentencesListTest:
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

            # Compute the cost
            # prob0 + prob1 = 1 normally, to check if its the case
            epsilon = 1e-7 # to avoid log(0)

            # BCE Loss
            #cost += -(mysentence.label * math.log(prob1+epsilon) + (1 - mysentence.label) * math.log(prob0+epsilon))

            #Normalized BCE Loss
            # From https://stats.stackexchange.com/questions/499423/normalized-cross-entropy
            # Here p = 0.5 for a balanced dataset
            # So we can simplify the formula
            p = 0.5
            cost+= -(mysentence.label * math.log(prob1+epsilon) + (1 - mysentence.label) * math.log(prob0+epsilon))/-(math.log(p))

            # Compute the accuracy
            # prob0 + prob1 = 1 normally, to check if its the case
            if prob0 >= 0.5 and mysentence.label == 0 or prob0 < 0.5 and mysentence.label == 1:
                correct_predictions += 1
            
        accuracy = correct_predictions / len(SentencesListTest)
        loss = cost / len(SentencesListTest)

        return loss, accuracy
        
