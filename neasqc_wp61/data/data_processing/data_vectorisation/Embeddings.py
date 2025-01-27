import sys
import json
import os
from keras.utils import pad_sequences
import numpy as np
import fasttext
import fasttext.util
from sentence_transformers import SentenceTransformer
import torch
from transformers import BertTokenizer, BertModel
import csv
import argparse

class Embeddings:
    def __init__(self, path, embtype, etype='sentence', gpu=-1):
        self.embtype=embtype
        self.model=None
        self.tokenizer=None      
        self.etype=etype
        #self.device = ('cpu' if gpu < 0 else f'cuda:{gpu}')
        self.device = ('cpu' if gpu < 0 else f'cuda')
        print("Using device: ", self.device)  
        print("Embtype: ", embtype)  
        print("Embtype: ", etype)  
    
        if (embtype=='fasttext'):
            if not os.path.exists('cc.en.300.bin'):
                fasttext.util.download_model('en', if_exists='ignore')     
            self.model = fasttext.load_model('cc.en.300.bin')
        elif (embtype == 'bert'):
            self.model=BertModel.from_pretrained(path, output_hidden_states = True, ).to(self.device)
            self.tokenizer=BertTokenizer.from_pretrained(path)
        elif (embtype == 'transformer'):
            self.model=SentenceTransformer(path, self.device)
        elif (embtype == 'jiina_transformer'):
            self.task = "retrieval.query"
            self.model=SentenceTransformer(path, self.device, trust_remote_code=True)
        print(path + ' loaded!')

    def getBertEmbeddingVector(self, sentence):
        embvec=[]
        try:
            if self.model:
                input_tokens = []
                encodedsent = self.tokenizer.tokenize("[CLS] " + sentence + " [SEP]")
                input_tokens.append(self.tokenizer.convert_tokens_to_ids(encodedsent))
                input_ids = pad_sequences(input_tokens, maxlen=50, dtype="long", value=0, truncating="post", padding="post")
                input_masks = []
                input_masks.append([int(token_id > 0) for token_id in input_ids[0]])
                input_ids = torch.tensor(input_ids, device=self.device)
                attention_mask = torch.tensor(input_masks, device=self.device)
                with torch.no_grad():
                    last_hidden_states  = self.model(input_ids, attention_mask=attention_mask)

 #The BERT uses [CLS] token to represent sentence information - tensor position [0].
 #See https://github.com/VincentK1991/BERT_summarization_1/blob/master/notebook/Primer_to_BERT_extractive_summarization_March_25_2020.ipynb 
                embvec = [last_hidden_states[0][0,0,:].cpu().detach().numpy().tolist()]
        except Exception as err:
            print(f"Unexpected {err=}")
        return embvec
        
    def getEmbeddingVector(self, sentence):
        embvec=[]
        if self.model:
            if (self.embtype=='fasttext'):
                embvec = [self.model.get_sentence_vector(sentence).tolist()]
            elif (self.embtype=='bert'):
                if self.etype=='word':
                    words = sentence.split()
                    embvec = [{'word': w, 'vector': self.getBertEmbeddingVector(w)[0]} for w in words]
                else:
                    embvec = self.getBertEmbeddingVector(sentence)
            elif (self.embtype=='transformer'):
                if self.device != 'cpu':
                    self.model = self.model.cuda()
                res = self.model.encode([sentence])
                embvec = [res.tolist()[0]]
            elif (self.embtype=='jiina_transformer'):
                res = self.model.encode([sentence], task=task, prompt_name=task)
                embvec = [res.tolist()[0]]
        return embvec
        
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", help = "file with text examples")
    parser.add_argument("-o", "--outfile", help = "json file with embeddings")
    parser.add_argument("-c", "--column", help = "'2' - 2-column file containing class and text columns, '0' - if the whole line is text example")
    parser.add_argument("-t", "--mtype", help = "Embedding model type: 'fasttext', 'bert' or 'transformer'")
    parser.add_argument("-e", "--etype", help = "Embedding type: sentence or word")
    parser.add_argument("-m", "--model", help = "Pre-trained embedding model. Some examples: 'cc.en.300.bin' 'bert-base-uncased' 'all-distilroberta-v1'")
    parser.add_argument("-g", "--gpu", help = "Number of GPU to use (from '0' to available GPUs), '-1' if use CPU (default is '-1')")
    args = parser.parse_args()

  #  os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu    
    dataset = []
    
    #try:
    rowlist = open(args.infile,encoding="utf-8").read().splitlines()
    vc = Embeddings(args.model,args.mtype, args.etype, gpu=int(args.gpu))
    
    for i, line in enumerate(rowlist):
        print(f"{i}/{len(rowlist)}")
        d = {}
        if str(args.column) == '0':
            d["sentence"]=line.rstrip()
        elif  str(args.column) == '2':
            linecols = line.split('\t')
            d["class"]=linecols[0].rstrip()
            d["sentence"]=linecols[1].rstrip()
        else:
            raise Exception("Wrong value of the column argument! Must be '0' to vectorize whole line or '3' if line contains class, text and parse tree columns.")
        d["sentence_vectorized"] = vc.getEmbeddingVector(d["sentence"])
        dataset.append(d)

    print(f"Saving vectors to {args.outfile}")
    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    #except Exception as err:
    #    print(f"Unexpected {err=}")


if __name__ == "__main__":
    sys.exit(int(main() or 0))