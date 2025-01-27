#!/bin/env python3

import sys
import argparse
import json
import time
import os
sys.path.append("./models/classical/")
from NNClassifier import (loadData, NNClassifier, prepareXYWords, prepareXYSentence, prepareClassValueDict, prepareXYWordsNoEmbedd, prepareTokenDict)
 
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-it", "--traindata", help = "Json data file for classifier training (with embeddings)")
    parser.add_argument("-id", "--devdata", help = "Json data file for classifier validation (with embeddings)")
    parser.add_argument("-ie", "--testdata", help = "Json data file for classifier evaluation (with embeddings)")
    parser.add_argument("-f", "--field", help = "Classify by field")
    parser.add_argument("-e", "--etype", help = "Embedding type: 'sentence', 'word' or '-'")
    parser.add_argument("-m", "--modeldir", help = "Directory where to save the trained model")
    parser.add_argument("-g", "--gpu", help = "Number of GPU to use (from '0' to available GPUs), '-1' if use CPU (default is '-1')")
    parser.add_argument("-r", "--runs", type=int, default=30, help = "Number of runs (useful for random restarts)")
    args = parser.parse_args()
    print(args)
    #try:
    traindata = loadData(args.traindata)
    devdata = loadData(args.devdata)
    testdata = loadData(args.testdata)
    idxdict = prepareClassValueDict(traindata, args.field)
    batch_size=32
    
    if len(traindata) > 10000:
        batch_size=1024#512
    if args.etype == "word":
        maxLen = 6
        trainX, trainY = prepareXYWords(traindata, maxLen, args.field, idxdict)
        vecsize = len(trainX[0][0])
        print(F"Vec size: {vecsize}")
        classifier = NNClassifier(model='CNN',vectorSpaceSize=vecsize, gpu=int(args.gpu), batch_size=batch_size)
        devX, devY = prepareXYWords(devdata, maxLen, args.field)
        testX, testY = prepareXYWords(testdata, maxLen, args.field)
    elif args.etype == "sentence":
        classifier = NNClassifier(gpu=int(args.gpu),batch_size=batch_size)
        trainX, trainY = prepareXYSentence(traindata, args.field, idxdict)
        devX, devY = prepareXYSentence(devdata, args.field)
        testX, testY = prepareXYSentence(testdata, args.field)
    elif args.etype == "-":
        maxLen = 6
        tokdict = prepareTokenDict(traindata)
        trainX, trainY = prepareXYWordsNoEmbedd(traindata, tokdict, maxLen, args.field, idxdict)
        vecsize = len(trainX[0])
        print(F"Vec size: {vecsize}")
        classifier = NNClassifier(model='LSTM',vectorSpaceSize=vecsize,gpu=int(args.gpu),batch_size=batch_size)
        devX, devY = prepareXYWordsNoEmbedd(devdata, tokdict, maxLen, args.field)
        testX, testY = prepareXYWordsNoEmbedd(testdata, tokdict, maxLen, args.field)
    else:
        print("Invalid embedding type. it must be 'word' or 'sentence'.")
        sys.exit(0)
    
    ds = {"input_args": {"runs": args.runs,"iterations": 100},
            "best_val_acc": 0,
            "best_run": 0,
            "time": [],
            "train_acc": [],
            "train_loss": [],
            "val_acc": [],
            "val_loss": [],
            "test_acc": [],
            "test_loss": []}
    
    classifier.shuffle(devX, devY, 28345)
    classifier.shuffle(trainX, trainY, 28345)
        
    for i in range(args.runs):
        print(f"Run number {i}")
        start_time= time.time()
        history = classifier.train(trainX, trainY, devX, devY)
        end_time= time.time()
                        
        nn_train_acc = max(history.history["accuracy"])
        
        test_acc, test_loss = classifier.test(testX, testY)
        
        print(f"Model train accuracy: {nn_train_acc}")
        print(f"Saving model to {args.modeldir}_{i}")
        
        classifier.save(f"{args.modeldir}_{i}")
        
        inv_map = {v: k for k, v in idxdict.items()}
        with open(f"{args.modeldir}_{i}/dict.json", 'w') as map_file:
            map_file.write(json.dumps(inv_map))

        if args.etype == "-":
            with open(f"{args.modeldir}_{i}/tokdict.json", 'w') as tok_file:
               tok_file.write(json.dumps(tokdict))
        
        ds["time"].append(end_time-start_time)
        ds["train_acc"].append(history.history["accuracy"])
        ds["train_loss"].append(history.history["loss"])
        ds["val_acc"].append(history.history["val_accuracy"])
        ds["val_loss"].append(history.history["val_loss"])
        ds["test_acc"].append(test_acc)
        ds["test_loss"].append(test_loss)
        
        if ds["best_val_acc"] < max(history.history["val_accuracy"]):
            ds["best_val_acc"] = max(history.history["val_accuracy"])
            ds["best_run"]=i
        if len(history.history["val_accuracy"]) < ds["input_args"]["iterations"]:
            ds["input_args"]["iterations"] = len(history.history["val_accuracy"])

        if not os.path.exists(args.modeldir):
            os.makedirs(args.modeldir)
        
        with open(f"{args.modeldir}/results.json", "w", encoding="utf-8") as f:
            json.dump(ds, f, ensure_ascii=False, indent=2)
    
if __name__ == "__main__":
    sys.exit(int(main() or 0))
