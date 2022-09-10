import sys
import os
import json
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Dense, Activation, Conv1D,
                          Dropout, MaxPooling1D, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers


def ts():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

class NNClassifier:
    """
    A class implementing neural network classifiers.
    
    Currently, a shallow feedforward neural network and a convolutional network are implemented.
    """
    def __init__(self, **kwargs):
        # default values
        self.params = {
            "model": "FFNN",
            "vectorSpaceSize": 768,
            "nClasses": 2,
            "learning_rate": 0.001,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "decay": 0,
            "epochs": 100,
            "epsilon": None,
            "amsgrad": False,
        }
        if "model" in kwargs and kwargs["model"] == "CNN": #defaults for CNN
            self.params["epochs"] = 30
            self.params["filterCounts"] = [300, 300]
            self.params["maxSentenceLen"] = 5
            self.params["dropout"] = 0.5

        self.params.update(kwargs)

    @staticmethod
    def createModel1(vectorSpaceSize, nClasses, **kwargs):
        model = Sequential()
        model.add(Dense(nClasses, input_dim=vectorSpaceSize, activation='softmax'))
        return model


    @staticmethod
    def createModelCNN(vectorSpaceSize, maxSentenceLen, nClasses, filterCounts, dropout, **kwargs):
        inp = Input(shape=(maxSentenceLen, vectorSpaceSize))
        filterLayer = []
        for ws, filters in enumerate(filterCounts, start=1):
            if filters > 0:
                conv = Conv1D(filters=filters,
                                kernel_size=ws,
                                activation='relu'
                                )(inp)
                conv = MaxPooling1D(pool_size=maxSentenceLen - ws + 1)(conv)
                filterLayer.append(conv)
        if len(filterLayer) > 1:
            merged = tf.keras.layers.concatenate(filterLayer)
        else:
            merged = filterLayer[0]
        merged = Flatten()(merged)
        if dropout>0:
            merged = Dropout(rate=dropout)(merged)
        out = Dense(units=nClasses, activation='softmax')(merged)
        model = Model(inp, out)
        return model


    @staticmethod
    def createAdamOptimizer(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                            decay=0, epsilon=None, amsgrad=False, **kwargs):
        opt = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2,
                    epsilon=epsilon, decay=decay, amsgrad=amsgrad)
        return opt


    def train(self, trainX, trainY):
        '''Trains the model.
        @param trainX: training data
        @param trainY: training targets (labels)
        @return: training history data as returned by Keras
        '''
        if self.params["model"] == "CNN":
            self.model = NNClassifier.createModelCNN(**self.params)
        elif self.params["model"] == "FFNN":
            self.model = NNClassifier.createModel1(**self.params)
        else:
            raise NotImplementedError(f"Unknown model type: {self.params['model']}")
        optimizer = NNClassifier.createAdamOptimizer(**self.params)
        self.model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        return self.model.fit(trainX, trainY, epochs=self.params["epochs"], verbose=2,)
                  #callbacks=callbacks)

    def predict(self, testX):
        res = self.model.predict(testX)
        topPrediction = np.argmax(res, axis=1)
        return topPrediction


def loadData(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def prepareTrainTestXYSentence(data):
    '''Prepares the data as numpy arrays suitable for training.
    @param data: the data to process
    @return: four numpy arrays corresponding to training and test input data and labels
    '''
    dX = []
    dY = []
    for i, dd in enumerate(["train_data", "test_data"]):
        arrX = []
        arrY = []
        for ex in data[dd]:
            arrX.append(ex["sentence_vectorized"][0])
            arrY.append(ex["truth_value"])
        X = np.array(arrX)
        Y = tf.keras.utils.to_categorical(arrY, 2)
        dX.append(X)
        dY.append(Y)
    return dX[0], dY[0], dX[1], dY[1]

def appendZeroVector(x, N, dim):
    if len(x) > N:
        return x[:N]
    return x + [[0] * dim] * (N - len(x))


def prepareTrainTestXYWords(data, maxLen):
    '''Prepares the data as numpy arrays suitable for training.
    @param data: the data to process
    @param maxLen: maximum sentence length. Longer sentences are truncated,
        shorter sentences are padded with all-zero vectors
    @return: four numpy arrays corresponding to training and test input data and labels
    '''
    dX = []
    dY = []
    for i, dd in enumerate(["train_data", "test_data"]):
        arrX = []
        arrY = []
        for ex in data[dd]:
            sentenceVectors = []
            for w in ex["sentence_vectorized"]:
                sentenceVectors.append(w["vector"])
                dim = len(w["vector"])
            arrX.append(sentenceVectors)
            arrY.append(ex["truth_value"])
        arrX = [appendZeroVector(sv, maxLen, dim) for sv in arrX]
        X = np.array(arrX)
        Y = tf.keras.utils.to_categorical(arrY, 2)
        dX.append(X)
        dY.append(Y)
    return dX[0], dY[0], dX[1], dY[1]

def evaluate(predictions, testY):
    """Evaluates the accuracy of the predictions."""
    return np.sum(predictions == np.argmax(testY, axis=1))/len(testY)

def main():


    classifier = NNClassifier()
    data = loadData("dataset_vectorized_bert_cased.json")
    #data = loadData("dataset_vectorized_fasttext.json")
    #trainX, trainY, testX, testY = prepareTrainTestXYSentence(data)

    maxLen = 5
    trainX, trainY, testX, testY = prepareTrainTestXYWords(data, maxLen)

    classifier.train(trainX, trainY)

    res = classifier.predict(testX)
    score = evaluate(res, testY)
    print(score)

    resFileName = datetime.datetime.now().strftime('results_%Y-%m-%d_%H-%M-%S.json')
    with open(resFileName , "w", encoding="utf-8") as f:
        json.dump({
            "res": res.tolist(),
            "score": score,
        }, f, indent=2)


if __name__ == "__main__":
    sys.exit(int(main() or 0))
