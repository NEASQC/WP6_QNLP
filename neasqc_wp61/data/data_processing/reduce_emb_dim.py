#!/bin/env python3

import sys
import argparse
import json
import pandas as pd
import numpy as np
from dim_reduction import (PCA, ICA, TSVD, UMAP, TSNE)
 
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-it", "--data", help = "Json data file with embeddings (train)")
    parser.add_argument("-ot", "--dataout", help = "Json data file with reduced embeddings (train)")
    parser.add_argument("-iv", "--data_val", help = "Json data file with embeddings (val)")
    parser.add_argument("-ov", "--dataout_val", help = "Json data file with reduced embeddings (val)")
    parser.add_argument("-ie", "--data_eval", help = "Json data file with embeddings (eval)")
    parser.add_argument("-oe", "--dataout_eval", help = "Json data file with reduced embeddings (eval)")
    parser.add_argument("-n", "--dimout", help = "Desired output dimension of the vectors")
    parser.add_argument("-a", "--algorithm", help = "Dimensionality reduction algorithm - 'PCA', 'ICA', 'TSVD', 'UMAP' or 'TSNE'")
    args = parser.parse_args()
    print(args)
       
    if args.algorithm not in ['PCA', 'ICA', 'TSVD', 'UMAP', 'TSNE']:
        print(f"{args.algorithm} not supported.")
        return 

    try:
        df = pd.read_json(args.data)
        df_val = pd.read_json(args.data_val)
        df_eval = pd.read_json(args.data_eval)
  
        vectors=df["sentence_vectorized"].tolist()
        flat_list=[x[0] for x in vectors]
        df["sentence_embedding"]=flat_list
        reduceddf=df[['class', 'sentence', 'sentence_embedding']]
        reduceddf['class'] = reduceddf['class'].astype(str)
        
        vectors_val=df_val["sentence_vectorized"].tolist()
        flat_list_val=[x[0] for x in vectors_val]
        df_val["sentence_embedding"]=flat_list_val
        reduceddf_val=df_val[['class', 'sentence', 'sentence_embedding']]
        reduceddf_val['class'] = reduceddf_val['class'].astype(str)
        
        vectors_eval=df_eval["sentence_vectorized"].tolist()
        flat_list_eval=[x[0] for x in vectors_eval]
        df_eval["sentence_embedding"]=flat_list_eval
        reduceddf_eval=df_eval[['class', 'sentence', 'sentence_embedding']]
        reduceddf_eval['class'] = reduceddf_eval['class'].astype(str)
        
        if args.algorithm=='PCA':
            reducer=PCA(reduceddf,int(args.dimout))
        elif args.algorithm=='ICA':
            reducer=ICA(reduceddf,int(args.dimout))
        elif args.algorithm=='TSVD':
            reducer=TSVD(reduceddf,int(args.dimout))
        elif args.algorithm=='UMAP':
            reducer=UMAP(reduceddf,int(args.dimout))
        elif args.algorithm=='TSNE':
            reducer=TSNE(reduceddf,int(args.dimout))

        reducer.reduce_dimension()
        dataset_val = reducer.apply_pretrained_reduction_model(reduceddf_val)
        dataset_eval = reducer.apply_pretrained_reduction_model(reduceddf_eval)

        reducer.dataset=reducer.dataset.rename(columns={"reduced_sentence_embedding": "sentence_vectorized"})
        dataset_val=dataset_val.rename(columns={"reduced_sentence_embedding": "sentence_vectorized"})
        dataset_eval=dataset_eval.rename(columns={"reduced_sentence_embedding": "sentence_vectorized"})


        vectors=reducer.dataset["sentence_vectorized"].tolist()
        list_of_list=[[x] for x in vectors]
        reducer.dataset["sentence_vectorized"]=list_of_list

        dflistofdict=reducer.dataset[['class', 'sentence', 'sentence_vectorized']].apply(lambda x: x.to_dict(), axis=1).to_list()

        with open(args.dataout,'w') as fout:
            json.dump(dflistofdict,fout, indent=2)
        
        vectors_val=dataset_val["sentence_vectorized"].tolist()
        list_of_list_val=[[x] for x in vectors_val]
        dataset_val["sentence_vectorized"]=list_of_list_val

        dflistofdict_val=dataset_val[['class', 'sentence', 'sentence_vectorized']].apply(lambda x: x.to_dict(), axis=1).to_list()

        with open(args.dataout_val,'w') as fout:
            json.dump(dflistofdict_val,fout, indent=2)
        
        vectors_eval=dataset_eval["sentence_vectorized"].tolist()
        list_of_list_eval=[[x] for x in vectors_eval]
        dataset_eval["sentence_vectorized"]=list_of_list_eval

        dflistofdict_eval=dataset_eval[['class', 'sentence', 'sentence_vectorized']].apply(lambda x: x.to_dict(), axis=1).to_list()

        with open(args.dataout_eval,'w') as fout:
            json.dump(dflistofdict_eval,fout, indent=2)
        
        print("Done!")
        
    except Exception as err:
        print(f"Unexpected {err=}")
        
if __name__ == "__main__":
    sys.exit(int(main() or 0))
