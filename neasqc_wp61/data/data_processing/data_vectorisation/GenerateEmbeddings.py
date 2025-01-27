import sys
import json
import fasttext
import fasttext.util
import argparse
from Embeddings import Embeddings
        
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", help = "Json data file with train/test split")
    parser.add_argument("-o", "--outfile", help = "Json data file with sentence embeddings")
    parser.add_argument("-t", "--mtype", help = "Embedding model type: 'fasttext', 'bert' or 'transformer'")
    parser.add_argument("-m", "--model", help = "Pre-trained embedding model. Some examples: 'cc.en.300.bin' 'bert-base-uncased' 'all-distilroberta-v1'")
    args = parser.parse_args()

    vc = Embeddings(args.model,args.mtype)   
    with open(args.infile, "r", encoding="utf-8") as f:
        jj = json.load(f)

    for dataname, dataset in jj.items():
        for i, d in enumerate(dataset):
            print(f"{i}/{len(dataset)}")
            d["sentence_vectorized"] = vc.getEmbeddingVector(d["sentence"])

    with open(args.outfile, "w", encoding="utf-8") as f:
        json.dump(jj, f, indent=2)


if __name__ == "__main__":
    sys.exit(int(main() or 0))