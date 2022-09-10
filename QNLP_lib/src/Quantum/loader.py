import json
import pandas as pd


def createdf(file):
    with open(file) as f:
        data = json.load(f)
    dftrain = pd.DataFrame(data['train_data'])
    dftest = pd.DataFrame(data['test_data'])
    mapsentypes = {'NOUN-TVERB-NOUN': 0,
                   'NOUN-IVERB-PREP-NOUN': 1,
                   'ADJ-NNOUN-TVERB-ADJ-NOUN': 2,
                   'NOUN-IVERB': 3}

    dftrain['sentence_type_code'] = dftrain['sentence_type'].map(mapsentypes)
    dftest['sentence_type_code'] = dftest['sentence_type'].map(mapsentypes)
    return dftrain, dftest


def getvocabdict(dftrain, dftest):
    vocab = dict()
    words = []
    for i, row in dftrain.iterrows():
        for word, wtype in zip(row['sentence'].split(' '), row['sentence_type'].split('-')):
            if word not in words:
                words.append(word)
                vocab[word] = [wtype]
            else:
                wtypes = vocab[word]
                if wtype not in wtypes:
                    wtypes.append(wtype)
                    vocab[word] = [wtypes]
                    vocab[word] = vocab[word][0]

    for i, row in dftest.iterrows():
        for word, wtype in zip(row['sentence'].split(' '), row['sentence_type'].split('-')):
            if word not in words:
                words.append(word)
                vocab[word] = [wtype]
            else:
                wtypes = vocab[word]
                if wtype not in wtypes:
                    wtypes.append(wtype)
                    vocab[word] = [wtypes]
                    vocab[word] = vocab[word][0]
    return vocab
