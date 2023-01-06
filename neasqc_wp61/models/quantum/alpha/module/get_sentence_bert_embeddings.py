from module.bert_text_preparation import *
from module.get_bert_embeddings import *

def get_sentence_bert_embeddings(SenList: list)->list:
    """Returns word embedding for each sentence.

    Takes a list of sentences and find a Bert embedding for each.:

    Parameters
    ----------
    SenList: list
        list of strings that represent each sentence.

    Returns
    -------
    Sentences_Embeddings: list
        List consisting of word embeddings for each sentence.
        
    """
    Sentences_Embeddings = []
    if type(SenList) == str:
        SenList = [SenList]
    for sentence in SenList:
        tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(sentence, tokenizer)
        list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
        nwords = len(sentence.split(" "))

        word_embeddings = []
        for word in sentence.split(" "):
            word_index = tokenized_text.index(word.lower().replace(".",""))
            word_embeddings.append(list_token_embeddings[word_index])

        Sentences_Embeddings.append(word_embeddings)
    return Sentences_Embeddings