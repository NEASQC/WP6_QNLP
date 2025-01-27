from transformers import BertTokenizer, BertModel
import json
import torch
import numpy as np

model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True,
                                  )

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

infile = "../../data/Complete_dataset.json"
outfile = "../../data/dataset_vectorised_bert.json"


def bert_text_preparation(text, tokenizer):

    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    del_list = []
    for i,x in enumerate(tokenized_text):
        if x[0]=='#':
            tokenized_text[i] = tokenized_text[i-1] + tokenized_text[i][2:]
            del_list.append(i-1)
    tokenized_text = [tokenized_text[i] for i in range(len(tokenized_text)) if i not in del_list]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)


    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors


def get_bert_embeddings(tokens_tensor, segments_tensors, model):


    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)

        hidden_states = outputs[2][1:]


    token_embeddings = hidden_states[-1]

    token_embeddings = torch.squeeze(token_embeddings, dim=0)

    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

    return list_token_embeddings

def get_sentence_BERT_embeddings(SenList):
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

def generate_vectorised_data(infile,outfile):
    with open(infile) as f:
        data = json.load(f)
        for data_type in data:
            for element in data[data_type]:
                sentence_embedding = np.zeros(768)
                number_of_words = len(element['sentence'])
                for word_embedding in get_sentence_BERT_embeddings(element['sentence'])[0]:
                    word_embedding = np.array(word_embedding)
                    sentence_embedding += word_embedding
                sentence_embedding = sentence_embedding/number_of_words
                element['sentence_vectorized'] = [list(sentence_embedding)]
    with open(outfile, 'w') as g:
        g.write(json.dumps(data,indent=2))

generate_vectorised_data(infile, outfile)