import json
import pandas as pd


def createdf(file):
    with open(file) as f:
        data = json.load(f)
    dftrain = pd.DataFrame(data['train_data'])


    map_tilde_lambeq = {
        's[n[n] (s\\\\n)[((s\\\\n)/(s\\\\n))   (s\\\\n)]]' : 'n,nrssln,nrs',
        's[n[(n/n)   n] (s\\\\n)[((s\\\\n)/(s\\\\n))   (s\\\\n)]]' : 'nnl,n,nrssln,nrs',
        's[n[n[(n/n)   n]] (s\\\\n)]' : 'nnl,n,nrs',
        's[n   (s\\\\n)[((s\\\\n)/n)   n[(n/n)   n]]]' : 'n,nrsnl,nnl,n',
        's[n   (s\\\\n)[((s\\\\n)/n)   n[(n/n)   n[(n/n)   n]]]]' : 'n,nrsnl,nnl,nnl,n'}

    dftrain['sentence_type'] = dftrain['structure_tilde'].map(map_tilde_lambeq)

    
    mapsentypes = {'n,nrssln,nrs': 0,
                   'nnl,n,nrssln,nrs': 1,
                   'nnl,n,nrs': 2,
                   'n,nrsnl,nnl,n': 3, 
                   'n,nrsnl,nnl,nnl,n': 4}

    dftrain['sentence_type_code'] = dftrain['sentence_type'].map(mapsentypes)
    return dftrain


def getvocabdict(dftrain):
    vocab = dict()
    words = []
    for i, row in dftrain.iterrows():
        for word, wtype in zip(row['sentence'].split(' '), row['sentence_type'].split(',')):
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
