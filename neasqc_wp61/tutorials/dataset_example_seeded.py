import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/quantum/pre-alpha/")
import pandas as pd
import json
import matplotlib.pyplot as plt
import random
import optimizer
import loader
import dictionary
import sentence
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################


def pre_alpha_classifier_seeded(seed, n_iter = 500):
    random.seed(seed)
    filename = 'amazonreview_reduced_train_pre_alpha.json'
    Dftrain = loader.createdf(current_path +"/../../data/datasets/"+filename)
    Myvocab = loader.getvocabdict(Dftrain)
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
                                   options={'maxiter': n_iter, 'rhobeg': 1},
                                   method="COBYLA")
    


    return (myopt.itercost)

