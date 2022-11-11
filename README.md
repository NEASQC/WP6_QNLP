# WP6 

# Installation

## Cloning the GitHub repository

To download the code as a local copy to run the code in your own machine, it is recommended to clone the repository using git. 

* Click on the 'Switch branches/tags' button to select v0.3 branch. 
* Click on the green code button and copy the HTTPS link. 
* Open a terminal in your computer and run the following command in the directory you want to clone the repository:
```console
  $ git clone <copied_url>
```
* Go to the cloned repository folder and switch to v0.3 branch by running the following command in the terminal:
```console
  $ git checkout v0.3
```

## Python version

The python version required to run the scripts and notebooks of this repo is Python3.9.

## Creating a new environment and installing required packages

Once you have cloned the repo, it is highly recommended to create a virtul environment which will act as a "virtual" isolated Python installation. To create a virtual environment, go to the directory where you want to create it and run the following command in the terminal:
```console
  $ python3.9 -m venv <environment name>
```
Once this has been done, the environment can be activated by running the following command:
```console
  $ source <environment_name>/bin/activate
```
If the environment has been activated correctly its name should appear in parentheses on the left of the user name in the terminal. To install the required packages, you will have to go to the root directory of the repository and run the command:
```console
  $ pip install -r requirements.txt
```
Instructions on how to install pip can be found in https://pip.pypa.io/en/stable/installation/ if it is not installed on the user's computer. 

## Download spacy model

Some of the tools used in the module require a language model to be donwloaded by the user. This can be done running the following command:
```console
  $ python3.9 -m spacy download en_core_web_lg
```
The language model will be stored in the created virtual environment. 


# Running the notebooks

We can use jupyterlab to run the jupyter notebooks that appear on the repo. To do so, we can run the following command in our terminal:
```console
  $ python -m ipykernel install --user --name  <environment_name> --display-name "<kernel name>"
  $ python3.9 -m jupyterlab <path of the notebook we want to run>
```
The first command will define a kernel, named <kernel name>, which you must change to after opening jupyterlab. The second command will open a jupyterlab terminal on our explorer, where we can run the selected notebook.
We will give now instructions for running each one of the notebooks, depending on the datasets that we want to use in our models. 


* [Classical_classifiers.ipynb](//repo//WP6_QNLP/neasqc_wp61/doc/tutorials/Classical_classifiers.ipynb)
    - In cell[2], the argument of ```loadData()``` must be one of the following:
        - ```"../../data/dataset_vectorized_bert_uncased.json"```
        - ```"../../data/dataset_vectorized_bert_cased.json"```
    - In cell[8], the argument of ```loadData()``` must be:
        - ```"../../data/dataset_vectorized_fasttext.json"```


* [Dataset_example.ipynb](//repo//WP6_QNLP/neasqc_wp61/doc/tutorials/Dataset_example.ipynb)
    - In cell[3], the value of the variable ```filename``` must be one of the following:
        - ```"Complete_dataset.json"```
        - ```"dataset_vectorized_bert_cased.json"```
        - ```"dataset_vectorized_bert_uncased.json"```
        - ```"dataset_vectorized_fasttext.json"```
        - ```"Expanded_Transitive_dataset.json"```
        - ```"Reduced_words_complete_dataset.json"```
  

* [Single_sentence_example.ipynb](//repo//WP6_QNLP/neasqc_wp61/doc/tutorials/Single_sentence_example.ipynb)
    - No dataset is input on this notebook. No restrictions when running the notebook. 


* [Dressed_QNLP_demo.ipynb](//repo/WP6_QNLP/neasqc_wp61/models/quantum/alpha/Dressed_QNLP_demo.ipynb)
    - In cell[24], the value of the variable ```filename``` can be any of the datasets:
        - ```"../../../data/Complete_dataset.json"```
        - ```'../../../data/dataset_vectorized_bert_cased.json"```
        - ```'../../../data/dataset_vectorized_bert_uncased.json"```
        - ```'../../../data/dataset_vectorized_fasttext.json"```
        - ```'../../../data/Expanded_Transitive_dataset.json"```
        - ```'../../../data/Reduced_words_complete_dataset.json"```
        - ```'../../../data/Reduced_words_transitive_dataset.json"```


# Pre-alpha Functionalities

The main scope of the pre-alpha model is to build a variational quantum algorithm that makes sentence classification in categories True or False. The structure of the analyzed sentences will be: 

* NOUN-TRANSITIVE VERB-NOUN
* NOUN-INTRANSIIVE VERB
* NOUN-INTRANSITIVE VERB-PREPOSITION-NOUN

## Classical classifiers

Some classical classifiers are implemented in order to have a reference against which to compare our quantum solution.

* K-nearest neighbors classifier from sklearn package applied to BERT embeddings.

* A feedforward neural network classifier from tensorflow package applied to BERT embeddings. 

* A convolutional neural network classifier from tensorflow package applied to fasttext embeddings. 

After the models are trained we compute and compare their accuracies on the training and test dataset. 

## Variational quantum circuit classfier

A variational quantum circuit is built with initial random parameters. In the circuit, only one qubit is measured (post-selection), and the variational parameters will be optimized using the labeled sentences in dataset so that the state obtained when measuring the qubit coincides with the value of the sentence (FALSE=0, TRUE=1). 

After the variational circuit is trained, the accuracy over the training and test dataset will be measured and compared. 

As an additional feature, a function that guesses a missing word in a sentence is also included. The function is applied to a new dataset generated by removing a noun from each of the sentences in the initial dataset. For a given sentence, the function will be looking for the missing word between all the words that have been removed, and will select the one whose circuit gives the greater probability of obtaining the state corresponding to the meaning of the sentence. 


# Pre-alpha Documentation

Let's briefly describe how the functionalities explained above are implemented. 

The classical classifiers are implemented in the notebook [Classical classifiers.ipynb](//repo//WP6_QNLP/neasqc_wp61/doc/tutorials/Classical_classifiers.ipynb). The file [NNClassifier.py](//repo/WP6_QNLP/neasqc_wp61/models/classical/NNClassifier.py) contains the class and functions used to prepare the data, build and train the convolutional and feedforward networks. 

Regarding the variational quantum circuit, in the notebook [Single_sentence_example.ipynb](//repo//WP6_QNLP/neasqc_wp61/doc/tutorials/Single_sentence_example.ipynb) we can find an example where the circuit parameters are optimized based on only one sentence. In the notebook [Dataset_example.ipynb](//repo//WP6_QNLP/neasqc_wp61/doc/tutorials/Dataset_example.ipynb), we can find an example on the variational algorithm trained with a whole dataset of sentences. We also can see there the implementation of a function to guess a missing word in a sentence. 

The functions and classes used to implement the variational quantum circuit are taken from different files. In [dictionary.py](//repo//WP6_QNLP/neasqc_wp61/models/quantum/pre-alpha/dictionary.py) we define classes that allow us to store the words than can appear on a sentence in dictionaries. Functions are defined that allow us to get and update the variational parameters associated to each word in the quantum circuit. Some of these functions are used in [sentence.py](//repo//WP6_QNLP/neasqc_wp61/models/quantum/pre-alpha/sentence.py), which provides the required tools to build the structure of the circuit that represents the sentence depending on its type and in some user-defined parameters. [ciruit.py](//repo//WP6_QNLP/neasqc_wp61/models/quantum/pre-alpha/circuit.py) contains functions that build, simplify (by qubit contractions) and execute the variational circuit. A class to optimize the variational parameters of the circuit with respect to a sentence or dataset of sentences can be found on [optimizer.py](//repo//WP6_QNLP/neasqc_wp61/models/quantum/pre-alpha/optimizer.py). Finally, in [loader.py](//repo//WP6_QNLP/neasqc_wp61/models/quantum/pre-alpha/loader.py) we can find functions that help in the processing of the datasets. 

# Alpha Functionalities

## Sentence Dataset and Model Purpose

The datatset used is Complete_dataset.json, which contains sentences of varying grammatical structure that are labelled as true or false. The idea is to train a model, using this dataset, that can classify unknown sentences as true or false.
 
## BERT Sentence Embeddings

Using the BERT model, the word embeddings of each word in each sentence in an inputted dataset are calculated. These word embeddings serve as the initial parameters, after dimensionality reduction, of the parametrised quantum circuits that represent each sentence.

## DisCoCat Circuits

Using the lambeq package, a tket parametrised quantum circuit is generated for each sentence in the dataset.

## Dressed Quantum Circuit for Sentence Classification

The dressed quantum circuit forms our trainable PyTorch neural network. Initially the sentence embedding dimensionality is reduced to be compatible with the dimension of their respective parameters within the parametrised quantum circuits. This is achieved using sequential, cascasding linear transformations. Next the quantum circuit is run, after which the measurement outcomes are fed into a post-processing neural network which ultimately classifies the sentence as true or false.

## Torch OPtimisation

The dressed quantum circuit model is trained using Torch.
  
# Alpha Documentation

The alpha is currently implemented in one single jupyter notebook, Dressed_QNLP_demo.ipynb. The beginning of the notebook contains a demonstration of how BERT sentence embeddings and parametrised quantum circuits can be generated using a small number of sample sentences. After this demonstration, and below the "Dressed Quantum Circuits" heading, two classes are defined which can be used in a pipeline that takes in our sentence dataset and generates a trainable model. After this generation, the model is trained using PyTorch. 

WARNING: You must run the initial demonstration as many of the functions and imports are used in the definition of the actual model, later in the notebook.

  




