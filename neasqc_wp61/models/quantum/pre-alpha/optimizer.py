import circuit
import scipy.optimize
import math
import numpy as np
from sentence import createsentencelist




class ClassicalOptimizer:

    def __init__(self, optimizer='cobyla', tol=0.1, maxiter=200):
        self.optimizer = optimizer
        self.tol = tol
        self.maxiter = maxiter

        self.cobyla_train_loss = []
        self.cobyla_train_accuracy = [] 



    def datasetcost(
            self, parameters, SentencesList, mydict):
        loss=0
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
            probs = [0, 0]
            for sample in mycirc.result:
                state = sample.state.bitstring
                postselectedqubits = ''.join(state[x] for x in range(len(state)) if x != mysentence.sentencequbit)
                if postselectedqubits == '0' * (mycirc.qlmprogram.qbit_count - 1):
                    if state[mysentence.sentencequbit] == '0':
                        probs[0] = sample.probability
                    elif state[mysentence.sentencequbit] == '1':
                        probs[1] = sample.probability
            prob0 = probs[0] / sum(probs)
            prob1 = probs[1] / sum(probs)
            epsilon = 1e-07
            loss +=  -(mysentence.label * math.log(prob1+epsilon) + (1 - mysentence.label) * math.log(prob0+epsilon))
            if prob0 >= 0.5:
                sentence_prediction = 0
            else:
                sentence_prediction = 1

            if sentence_prediction == mysentence.label:
                correct_predictions += 1
            
        self.cobyla_train_loss.append(loss/len(SentencesList))
        self.cobyla_train_accuracy.append(correct_predictions/len(SentencesList))
            
        return loss/len(SentencesList)


    def optimizedataset(self, sentencelist, params0, mydict, options={'maxiter':3}, method="COBYLA"):
        params0 = np.array(params0)
        self.results_weights = []
        def callback_function(x):
            self.results_weights.append(x.copy().tolist())
        result = scipy.optimize.minimize(self.datasetcost, params0,
                                                    args=(sentencelist,mydict),
                                                    options=options,
                                                    method=method,
                                                    callback = callback_function)
        return result
    
    def compute_bceloss_prediction_dataset(
            self, SentencesList, mydict):
        
        loss=0
        correct_predictions = 0
        predictions_list = []
        par, ix = mydict.getindexmodelparams()
        for mysentence in SentencesList:
            shapedparams = []
            for word, cat in zip(mysentence.sentence.lower().split(' '), mysentence.sentencestructure.split(',')):
                wordparams =[par[i] for i in ix[(word, cat)]]
                shapedparams.append(wordparams)
            mysentence.setsentenceparamsfromlist(shapedparams)
            mycirc = circuit.CircuitBuilder()
            mycirc.createcircuit(mysentence, dataset=True)
            mycirc.executecircuit()
            probs = [0, 0]
            for sample in mycirc.result:
                state = sample.state.bitstring
                postselectedqubits = ''.join(state[x] for x in range(len(state)) if x != mysentence.sentencequbit)
                if postselectedqubits == '0' * (mycirc.qlmprogram.qbit_count - 1):
                    if state[mysentence.sentencequbit] == '0':
                        probs[0] = sample.probability
                    elif state[mysentence.sentencequbit] == '1':
                        probs[1] = sample.probability
            prob0 = probs[0] / sum(probs)
            prob1 = probs[1] / sum(probs)
            epsilon = 1e-07
            loss +=  -(mysentence.label * math.log(prob1+epsilon) + (1 - mysentence.label) * math.log(prob0+epsilon))

            if prob0 >= 0.5:
                sentence_prediction = 0
            else:
                sentence_prediction = 1

            if sentence_prediction == mysentence.label:
                correct_predictions += 1

            predictions_list.append(sentence_prediction)
            
        return loss/len(SentencesList), correct_predictions/len(SentencesList), predictions_list

    def compute_loss_accuracy_iterations(self, mydict, SentencesList):
        loss_list = []
        accuracies_list = []
        for r in self.results_weights:
            parameters = r
            mydict.updateparams(parameters)
            loss, accuracy, predictions = self.compute_bceloss_prediction_dataset(SentencesList, mydict)       
            loss_list.append(loss)
            accuracies_list.append(accuracy)
        return loss_list, accuracies_list, predictions
    
    
