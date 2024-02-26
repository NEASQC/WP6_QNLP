
import math
import random



class PartOfSpeech:
    def __init__(self, parts=None,
                 cats=None):
        if cats is None:
            cats = [["n"],
                    ["nr", "s", "nl"],
                    ["nr", "s"],
                    ["n", "nl"],
                    ["nr", "s", "sl", "n"]]
        if parts is None:
            parts = ["n",
                     "nrsnl",
                     "nrs",
                     "nnl",
                     "nrssln"]
        self.cats = dict()
        for i, part in enumerate(parts):
            self.cats[part] = cats[i]


class QuantumDict:

    def __init__(self, qn=1, qs=1):

        self.dictionary = dict()
        self.partsOfSpeech = PartOfSpeech()
        self.qn = qn
        self.qs = qs

    def checkverbtype(self, token):
        indirect_object = False
        direct_object = False
        for item in token.children:
            if (item.dep_ == "iobj" or item.dep_ == "pobj"):
                indirect_object = True
            if (item.dep_ == "dobj" or item.dep_ == "dative"):
                direct_object = True
        if indirect_object and direct_object:
            return 'DTVERB'
        elif direct_object and not indirect_object:
            return 'TVERB'
        elif not direct_object and not indirect_object:
            return 'ITVERB'
        else:
            return 'VERB'

    def addwords(self, mysentence=None, myvocab=None):

        if (mysentence is not None) & (myvocab is not None):
            print("sentence and vocab cannot be provided at the same time")
            raise


        if myvocab is not None:
            for wordstring in myvocab.keys():
                wordtypes=myvocab[wordstring]
                if wordstring not in self.dictionary.keys():
                    self.dictionary[wordstring] = QuantumWord(wordstring=wordstring, wordtype=wordtypes)
                    self.dictionary[wordstring].setwordproperties(self, myvocab=myvocab)


    def setwordparams(self, myword, wordtype, randompar=True, parameterization='Simple', layers=1, seed=30031935):
        random.seed(seed)
        gates = []
        if randompar:
            if parameterization == 'Simple':  # Two rotations + C-NOT gate per layer
                for layer in range(layers):
                    for qubit in range(self.dictionary[myword].nqubits[wordtype]):
                        ry = 2 * math.pi * random.random()
                        rz = 2 * math.pi * random.random()
                        gates.append(dict({'Gate': 'RY', 'Angle': ry, 'Qubit': qubit}))
                        gates.append(dict({'Gate': 'RZ', 'Angle': rz, 'Qubit': qubit}))

                    for qubit in range(self.dictionary[myword].nqubits[wordtype])[:-1]:
                        gates.append(dict({'Gate': 'CX', 'Qubit': [qubit, qubit + 1]}))
        return gates

    def setvocabparams(self, randompar=True, parameterization='Simple', layers=1, seed=30031935):
        random.seed(seed)
        if randompar:
            if parameterization == 'Simple':  # Two rotations + C-NOT gate per layer
                for layer in range(layers): #This need to be refactored to use NumPy
                    for word in self.dictionary.keys():
                        self.dictionary[word].gateset = dict()
                        for wordtype in self.dictionary[word].wordtype:
                            gates = self.setwordparams(word,wordtype, seed=random.randint(0,int(1e10)))
                            self.dictionary[word].gateset[wordtype] = gates


    def getindexmodelparams(self):
        index = 0
        paramindex = dict()
        modelparams = []
        for word in self.dictionary.keys():
            for cat in self.dictionary[word].gateset.keys():
                paramindex[(word, cat)] = []
                for gate in self.dictionary[word].gateset[cat]:
                    if (gate['Gate'] == 'RY') or (gate['Gate'] == 'RZ'):
                        modelparams.append(gate['Angle'])
                        paramindex[(word, cat)].append(index)
                        index += 1
        return modelparams, paramindex

    def updateparams(self, params):
        index = 0
        for word in self.dictionary.keys():
            for cat in self.dictionary[word].gateset.keys():
                igate = 0
                for gate in self.dictionary[word].gateset[cat]:
                    if (gate['Gate'] == 'RY') or (gate['Gate'] == 'RZ'):
                        self.dictionary[word].gateset[cat][igate]['Angle'] = params[index]
                        index += 1
                        igate += 1






















class QuantumWord:

    def __init__(self, token=None, wordstring=None, wordtype=None):

        if (token is not None) & (wordstring is not None):
            print("Token and string cannot be provided at the same time")
            raise

        if token is not None:
            self.token = token
            self.word = token.text.strip()
            self.lemma = self.token.lemma_
        if wordstring is not None:
            if wordtype is None:
                print("A word type must be provided when using a word string")
                raise
            self.token = None
            self.word = wordstring
            self.lemma=None
            self.wordtype= wordtype

    def setwordproperties(self, mydict, mysentence=None, myvocab=None):

        if (mysentence is not None) & (myvocab is not None):
            print("sentence and vocab cannot be provided at the same time")
            raise

        if mysentence is not None:
            wordtype = mysentence.nlp(self.token.text)[0].pos_
            if (wordtype == "NOUN") or (wordtype == "PROPN"):
                mydict.dictionary[self.word].nqubits = mydict.qn
            elif wordtype == "ADJ":
                mydict.dictionary[self.word].nqubits = 2 * mydict.qn
            elif wordtype == "ADP":
                wordtype = "PREP"
                mydict.dictionary[self.word].nqubits = 3 * mydict.qn + 2 * mydict.qs
            elif wordtype == "VERB":
                wordtype = mydict.checkverbtype(self.token)
                if wordtype == "TVERB":
                    mydict.dictionary[self.word].nqubits = 2 * mydict.qn + mydict.qs
                elif wordtype == "ITVERB":
                    wordtype = "IVERB"
                    mydict.dictionary[self.word].nqubits = mydict.qn + mydict.qs

            mydict.dictionary[self.word].partofspeech = wordtype
            mydict.dictionary[self.word].category = mydict.partsOfSpeech.cats[wordtype]

        if myvocab is not None:
            mydict.dictionary[self.word].nqubits = dict()
            mydict.dictionary[self.word].category = dict()
            for wordtype in self.wordtype:
                if wordtype == "n":
                    mydict.dictionary[self.word].nqubits[wordtype] = mydict.qn
                elif wordtype == "nrsnl":
                    mydict.dictionary[self.word].nqubits[wordtype] = 2 * mydict.qn + mydict.qs
                elif wordtype == "nrs":
                    mydict.dictionary[self.word].nqubits[wordtype] = mydict.qn + mydict.qs
                elif wordtype == "nnl":
                    mydict.dictionary[self.word].nqubits[wordtype] = 2 * mydict.qn
                elif wordtype == "nrssln":
                    mydict.dictionary[self.word].nqubits[wordtype] = 2 * mydict.qn + 2 * mydict.qs
                mydict.dictionary[self.word].category[wordtype] = mydict.partsOfSpeech.cats[wordtype]
