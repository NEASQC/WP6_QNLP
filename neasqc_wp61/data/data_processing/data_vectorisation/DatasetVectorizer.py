import sys
import json
from VectorizerConnector import VectorizerConnector



def main():

    inFile = "dataset.json"
    outFile = "dataset_vectorized.json"

    with open(inFile, "r", encoding="utf-8") as f:
        jj = json.load(f)

    vc = VectorizerConnector("http://192.168.99.100", "12345")

    for dataname, dataset in jj.items():
        for i, d in enumerate(dataset):
            print(f"{i}/{len(dataset)}")
            d["sentence_vectorized"] = vc.vectorize_sentence(d["sentence"])

    with open(outFile, "w", encoding="utf-8") as f:
        json.dump(jj, f, indent=2)


if __name__ == "__main__":
    sys.exit(int(main() or 0))