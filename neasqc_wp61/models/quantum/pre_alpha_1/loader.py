import pandas as pd


def createdf(train_csv, val_csv, test_csv):
    dftrain = pd.read_csv(
    train_csv, sep='\t+',
    header=None, names=['label', 'sentence', 'structure_tilde'],
    engine='python')

    dfval = pd.read_csv(
    val_csv, sep='\t+',
    header=None, names=['label', 'sentence', 'structure_tilde'],
    engine='python'
    )
   
    dftest = pd.read_csv(
    test_csv, sep='\t+',
    header=None, names=['label', 'sentence', 'structure_tilde'],
    engine='python')
    

    map_tilde_lambeq = {
        's[n[n] (s\\\\n)[((s\\\\n)/(s\\\\n))   (s\\\\n)]]' : 'n,nrssln,nrs',
        's[n[(n/n)   n] (s\\\\n)[((s\\\\n)/(s\\\\n))   (s\\\\n)]]' : 'nnl,n,nrssln,nrs',
        's[n[n[(n/n)   n]] (s\\\\n)]' : 'nnl,n,nrs',
        's[n   (s\\\\n)[((s\\\\n)/n)   n[(n/n)   n]]]' : 'n,nrsnl,nnl,n',
        's[n   (s\\\\n)[((s\\\\n)/n)   n[(n/n)   n[(n/n)   n]]]]' : 'n,nrsnl,nnl,nnl,n'}

    dftrain['sentence_type'] = dftrain['structure_tilde'].map(map_tilde_lambeq)
    dfval['sentence_type'] = dfval['structure_tilde'].map(map_tilde_lambeq)
    dftest['sentence_type'] = dftest['structure_tilde'].map(map_tilde_lambeq)
    
    mapsentypes = {'n,nrssln,nrs': 0,
                   'nnl,n,nrssln,nrs': 1,
                   'nnl,n,nrs': 2,
                   'n,nrsnl,nnl,n': 3, 
                   'n,nrsnl,nnl,nnl,n': 4}

    dftrain['sentence_type_code'] = dftrain['sentence_type'].map(mapsentypes)
    dfval['sentence_type_code'] = dfval['sentence_type'].map(mapsentypes)
    dftest['sentence_type_code'] = dftest['sentence_type'].map(mapsentypes)
    truth_value_train_list = []
    truth_value_val_list = []
    truth_value_test_list = []
    for i in range(dftrain.shape[0]):
        if dftrain['label'].iloc[i] == 1:
            truth_value_train_list.append(False)
        if dftrain['label'].iloc[i] == 2:
            truth_value_train_list.append(True)
    for i in range(dfval.shape[0]):
        if dftrain['label'].iloc[i] == 1:
            truth_value_val_list.append(False)
        if dftrain['label'].iloc[i] == 2:
            truth_value_val_list.append(True)
    for i in range(dftest.shape[0]):
        if dftest['label'].iloc[i] == 1:
            truth_value_test_list.append(False)
        if dftest['label'].iloc[i] == 2:
            truth_value_test_list.append(True)
    
    dftrain['truth_value'] = truth_value_train_list
    dfval['truth_value'] = truth_value_val_list
    dftest['truth_value'] = truth_value_test_list

        

    return dftrain, dfval, dftest


def getvocabdict(dftrain, dfval, dftest):
    vocab = dict()
    words = []
    for i, row in dftrain.iterrows():
        for word, wtype in zip(row['sentence'].lower().split(' '), row['sentence_type'].split(',')):
            if word not in words:
                words.append(word)
                vocab[word] = [wtype]
            else:
                wtypes = vocab[word]
                if wtype not in wtypes:
                    wtypes.append(wtype)
                    vocab[word] = [wtypes]
                    vocab[word] = vocab[word][0]
    
    for i, row in dfval.iterrows():
        for word, wtype in zip(row['sentence'].lower().split(' '), row['sentence_type'].split(',')):
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
        for word, wtype in zip(row['sentence'].lower().split(' '), row['sentence_type'].split(',')):
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
