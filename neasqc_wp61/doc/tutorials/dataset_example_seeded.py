import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../../models/quantum/pre-alpha/")
import pandas as pd
import json
import matplotlib.pyplot as plt
import random
import optimizer_cost_normalised
import loader
import circuit
import dictionary_seeded
import sentence_seeded
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################


def pre_alpha_classifier_seeded(seed):
    random.seed(seed)
    filename = 'dataset_vectorized_fasttext.json'
    Dftrain, Dftest = loader.createdf(current_path +"/../../data/datasets/"+filename)#Dataframe form Json file
    #Several datasets are found in the Dataset folder. 
    #In the complete dataset, three types of sentences are found:
    # 

    Myvocab = loader.getvocabdict(Dftrain, Dftest)#A dictionary with word and categories found in dataset
    MyDict = dictionary_seeded.QuantumDict(qn=1,#Number of qubits for noun category. 
                                    qs=1)#Number of qubits for sentence category
    MyDict.addwords(myvocab=Myvocab)
    MyDict.setvocabparams(seed=seed)#Randomly assign params to words following and ansatz structure.




##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################    


    def createsentencelist(dftrain, mydict):
        sentences_list = []
        for i, DataInstance in dftrain.iterrows():
            a_sentence = sentence_seeded.Sentence(DataInstance,
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
    SentencesList = [x for x in random.sample(SentencesList, 40)]#For some dataset that may be too big,
                                                                #we can select just a few sentences to try out
                                                                #the model parameters.
    par, ix = MyDict.getindexmodelparams()
    myopt = optimizer_cost_normalised.ClassicalOptimizer()
    result = myopt.optimizedataset(SentencesList, par, MyDict,
                                   tol=1e-5,
                                   options={'maxiter':500, 'rhobeg': 1},
                                   method="COBYLA")
    


    return (myopt.itercost)

