# Notebooks
A set of notebooks to show the library workflow. Two notebooks are provided regarding the quantum training part. 
* In 'Single_sentence_example.ipynb' a single sentence with structure 'NOUN-IVERB-PREP-NOUN' is trained. 
* In 'Dataset_example.ipynb' a .json file is used to generate a dataset of transitive sentences ('NOUN-TVERB-NOUN'). 

The reason of not using all the sentence types available is that convergence for datasets is still not good, and better ways have to be found to represent the same words appearing as different parts of the speech. Right now, they are treated as separated instances, and thus, given different parameters. This is not completely realistic and can a point of improvement.
Also, the quality of the dataset is crucial, and a sweet spot for the number of sentences, different words and its relations, and sentences structures need to be found.

## Classical NLP notebooks
The notebooks for the classical NLP module are located in the
[classical](./classical/) subdirectory. The directory contains a separate
`requirements.txt` file that should be used to set up a separate python
environment for running the classical NLP notebooks.
