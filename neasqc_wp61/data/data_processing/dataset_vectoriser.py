import json
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import argparse
import os


def vectorise(dataset, embedding):
    """
    This function takes the path to a dataset and a choice of BERT embedding
    (sentence or word) as input and returns nothing. It then reads this dataset
    and from it, builds a corresponding dataset with the chosen embedding for
    each sentence.
    """
    dftrain = pd.read_csv(dataset, sep='\t', header=None, usecols=[0, 1], names=['class','sentence'], dtype=str)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    for param in bert_model.parameters():
        param.requires_grad = False
    bert_model.eval()

    dataset_absolute_path = os.path.abspath(dataset)
    directory, file_name = os.path.split(dataset_absolute_path)

    if embedding == "sentence":
        print("Generating sentence embeddings. Please wait...")
        sentence_embedding_list = []
        for sentence in dftrain.sentence.values:
            inputs = tokenizer.encode_plus(sentence).input_ids
            inputs = torch.LongTensor(inputs)
            # inputs = inputs.to(mps_device)

            with torch.no_grad():
                sentence_embedding = (
                    bert_model(inputs.unsqueeze(0))[1]
                    .squeeze(0)
                    .cpu()
                    .detach()
                    .numpy()
                )
            sentence_embedding_list.append(
                [tensor.item() for tensor in sentence_embedding]
            )

        dftrain["sentence_embedding"] = sentence_embedding_list
        dftrain.to_csv( #NOTE: change desired output file name?
            directory + "/" + file_name.split(".")[0] + "_sentence_bert.tsv", sep='\t',
            index=False,
        )
        print("Done!")

    elif embedding == "word":
        print("Generating word embeddings. Please wait...")
        sentence_embedding_list = []
        for sentence in dftrain.sentence.values:
            sentence_embedding = []
            for word in sentence.split():
                inputs = tokenizer.encode_plus(word).input_ids
                inputs = torch.LongTensor(inputs)
                # inputs = inputs.to(mps_device)

                with torch.no_grad():
                    word_embedding = (
                        bert_model(inputs.unsqueeze(0))[1]
                        .squeeze(0)
                        .cpu()
                        .detach()
                        .numpy()
                    )

                sentence_embedding.append(word_embedding.tolist())
            sentence_embedding_list.append(sentence_embedding)

        dftrain["sentence_vectorized"] = sentence_embedding_list
        dftrain.to_csv(
            os.path.join(
                directory, file_name.split(".")[0] + "_word_bert.tsv"
            ), sep='\t',
            index=False,
        )
        print("Done!")

    else:
        print(
            "Unsupported embedding type specified. Please choose 'sentence' or 'word'."
        )


def main():
    """
    This is the main function in this script. It takes no arguments and
    returns nothing. It reads command line input from the user and calls
    the vectorise function with this input..
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="the filename of your dataset")
    parser.add_argument(
        "-e",
        "--embedding",
        required=True,
        help="choice of embedding (sentence or word)",
        type=str,
    )

    args = parser.parse_args()

    vectorise(args.dataset, args.embedding)


if __name__ == "__main__":
    main()
