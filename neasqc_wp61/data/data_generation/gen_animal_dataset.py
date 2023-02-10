#!/bin/env python3

import argparse
import random
from typing import Optional, Dict, List, Tuple

# Much of this is taken from
# https://github.com/oxford-quantum-group/discopy/blob/main/docs/notebooks/functorial_language_model.ipynb

# transitive sentences
# fmt: off
trans_corpus: Dict[str, Dict[str, List[str]]] = {
    "dog": {
        "chases": ["cat", "fox"],
        "bites": ["cat", "fox", "bone"],
        "eats": ["bone"],
    },
    "cat": {
        "chases": ["mouse"],
        "flees": ["dog"],
        "bites": ["mouse", "fish"],
        "eats": ["fish"],
    },
    "mouse": {
        "flees": ["cat"],
        "bites": ["cheese"],
        "eats": ["cheese"],
    },
    "fox": {
        "chases": ["chicken"],
        "flees": ["dog"],
        "bites": ["chicken"],
        "eats": ["chicken"],
    },
    "chicken": {
        "flees": ["fox"],
        "eats": ["grain"],
    },
    "whale": {
        "eats": ["krill"],
    },
    "seal": {
        "eats": ["fish"],
    },
}
# fmt: on

# fmt: off
trans_corpus_false: Dict[str, Dict[str, List[str]]] = {
    "dog": {
        "flees": ["cat", "fox"],
        "bites": ["grain", "krill"],
        "eats": ["grain", "krill"],
    },
    "cat": {
        "chases": ["fox"],
        "flees": ["mouse", "chicken", "krill"],
        "eats": ["grain"],
    },
    "mouse": {
        "chases": ["cat", "dog", "fox"],
        "flees": ["krill"],
        "bites": ["cheese"],
        "eats": ["fox", "dog"],
    },
    "fox": {
        "chases": ["dog"],
        "flees": ["mouse"],
        "bites": ["krill"],
        "eats": ["grain"],
    },
    "chicken": {
        "flees": ["mouse"],
        "eats": ["fox", "cat"],
    },
    "whale": {
        "eats": ["grain", "chicken"],
    },
    "seal": {
        "eats": ["grain"],
    },
}
# fmt: on

# intransitive sentences
# fmt: off
itrans_corpus: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "dog": {
        "runs": {
            "on": ["land"],
            "after": ["cat", "fox"],
        },
        "chases": {
            "after": ["cat", "fox"],
        },
        "barks": {
            "at": ["cat", "fox"],
        },
    },
    "cat": {
        "runs": {
            "on": ["land"],
            "after": ["mouse"],
        },
        "chases": {
            "after": ["mouse"],
        },
        "meows": {},
    },
    "mouse": {
        "runs": {
            "on": ["land"],
        },
        "squeaks": {},
    },
    "fox": {
        "runs": {
            "on": ["land"],
            "after": ["chicken"],
        },
        "chases": {
            "after": ["chicken"],
        },
    },
    "chicken": {
        "runs": {
            "on": ["land"],
        },
        "clucks": {},
    },
    "fish": {
        "swims": {
            "in": ["water"],
        },
    },
    "whale": {
        "swims": {
            "in": ["water"],
        },
    },
    "seal": {
        "swims": {
            "in": ["water"],
        },
        "runs": {
            "on": ["land"],
        },
    },
}
# fmt: on

# fmt: off
itrans_corpus_false: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "dog": {
        "runs": {
            "in": ["water"],
            "after": ["krill", "grain"],
        },
        "chases": {
            "after": ["krill", "grain"],
        },
        "barks": {
            "at": ["krill", "grain"],
        },
        "meows": {},
        "squeaks": {},
      },
    "cat": {
        "runs": {
            "in": ["water"],
            "after": ["dog", "fox"],
        },
        "chases": {
            "after": ["fox"],
        },
        "squeaks": {},
    },
    "mouse": {
        "runs": {
            "in": ["water"],
        },
        "meows": {},
    },
    "fox": {
        "runs": {
            "in": ["water"],
            "after": ["dog"],
        },
        "chases": {
            "after": ["dog"],
        },
    },
    "chicken": {
        "runs": {
            "in": ["water"],
        },
    },
    "fish": {
        "swims": {
            "on": ["land"],
        },
        "meows": {},
        "clucks": {},
    },
    "whale": {
        "swims": {
            "on": ["land"],
        },
    },
    "seal": {
        "swims": {
            "on": ["land"],
        },
        "runs": {
            "in": ["water"],
        },
        "clucks": {},
    },
}
# fmt: on


def generate_dataset(seed: Optional[int] = None) -> List[Tuple[str, str, bool]]:
    '''Generates a list of sentences, sentence types and their truth values.
    Sentences consist of facts about animals.

    @param seed: if a seed is provided, the returned list is shuffled; the seed
        is used to initialize the random number generator
    @return: the dataset -- a list of
        sentence, sentence type, sentence truth value tuples
    '''
    dataset: List[Tuple[str, str, bool]] = []

    for corpus, truth_value in [(trans_corpus, True), (trans_corpus_false, False)]:
        for subj, verb_dict in corpus.items():
            for verb, dobjs in verb_dict.items():
                for dobj in dobjs:
                    dataset.append((f"{subj} {verb} {dobj}", "NOUN-TVERB-NOUN", truth_value))

    for corpus, truth_value in [(itrans_corpus, True), (itrans_corpus_false, False)]:
        for subj, iverb_dict in corpus.items():  # can't reuse the verb_dict
                                                 # variable here because mypy
                                                 # complains about its type
                                                 # changing
            for verb, prep_dict in iverb_dict.items():
                # Given that "seal swims in water 1" we can infer that "seal swims 1",
                # but from "dog in water 0" we cannot infer that "dog runs 0", hence
                # we add the short "subj verb" sentences only for true facts or when
                # the prep_dict is explicitly left empty
                if truth_value or len(prep_dict) == 0:
                    dataset.append((f"{subj} {verb}", "NOUN-IVERB", truth_value))
                for prep, idobjs in prep_dict.items():
                    for idobj in idobjs:
                        dataset.append((f"{subj} {verb} {prep} {idobj}", "NOUN-IVERB-PREP-NOUN", truth_value))

    if seed is not None:
        rand_gen = random.Random(seed)
        rand_gen.shuffle(dataset)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
            Generate a simple dataset with facts about animals.
            The output format is "<sentence>\\t<sentence_type>\\t<truth_value>"
            where <truth_value> is 1 if the fact is true and 0 otherwise."""
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        required=False,
        help="""
            A seed to initialize the random number generator.
            Sentence order isn't randomized, if not provided.""",
    )
    args = parser.parse_args()
    for sentence, sentence_type, truth_value in generate_dataset(args.seed):
        print(sentence, sentence_type, 1 if truth_value else 0, sep="\t")


