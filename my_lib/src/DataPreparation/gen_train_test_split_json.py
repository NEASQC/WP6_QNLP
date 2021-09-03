#!/bin/env python3

import sys
import argparse
import random
import json
from sklearn.model_selection import train_test_split
from gen_animal_dataset import generate_dataset
from typing import Union, List, Tuple, Dict

def unpack_data(data: List[Tuple[str, str, bool]]) -> List[Dict[str, Union[str, bool]]]:
    return [{
        "sentence": sentence,
        "sentence_type": sentence_type,
        "truth_value": truth_value,
    } for sentence, sentence_type, truth_value in data]

def main():

    parser = argparse.ArgumentParser(
        description="""
            Generate and save a simple dataset with facts about animals."""
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        required=False,
        default=1,
        help="""
            A seed to initialize the random number generator.
            1, if not provided.""",
    )
    parser.add_argument(
        "-t",
        "--trainsize",
        type=float,
        required=False,
        default=0.8,
        help="""
            the proportion of the dataset to be used as training set""",
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        required=False,
        default="dataset.json",
        help="""
            the name of the output file""",
    )
    args = parser.parse_args()

    seed = args.seed
    train_size = args.trainsize
    filename = args.filename

    data = generate_dataset(seed)
    strat_data = [(truth_value, sentence_type) for sentence, sentence_type, truth_value in data]
    train_data, test_data = train_test_split(data, train_size=train_size, random_state=seed, stratify=strat_data)

    train_data = unpack_data(train_data)

    test_data = unpack_data(test_data)

    ds = {
        "train_data": train_data,
        "test_data": test_data,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(ds, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    sys.exit(int(main() or 0))
