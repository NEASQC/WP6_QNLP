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


* **Classical_classifiers.ipynb**: 
    - In cell[2], the argument of ```loadData()``` must be one of the following:
        - ```"../../data/dataset_vectorized_bert_uncased.json"```
        - ```"../../data/dataset_vectorized_bert_cased.json"```
    - In cell[8], the argument of ```loadData()``` must be:
        - ```"../../data/dataset_vectorized_fasttext.json"```


* **Dataset_example.ipynb**
    - In cell[3], the value of the variable ```filename``` must be one of the following:
        - ```"Complete_dataset.json"```
        - ```"dataset_vectorized_bert_cased.json"```
        - ```"dataset_vectorized_bert_uncased.json"```
        - ```"dataset_vectorized_fasttext.json"```
        - ```"Expanded_Transitive_dataset.json"```
        - ```"Reduced_words_complete_dataset.json"```
  

* **Single_sentence_example.ipynb**
    - No dataset is input on this notebook. No restrictions when running the notebook. 


* **Dressed_QNLP_demo.ipynb**
    - In cell[24], the value of the variable ```filename``` can be any of the datasets:
        - ```"../../../data/Complete_dataset.json"```
        - ```'../../../data/dataset_vectorized_bert_cased.json"```
        - ```'../../../data/dataset_vectorized_bert_uncased.json"```
        - ```'../../../data/dataset_vectorized_fasttext.json"```
        - ```'../../../data/Expanded_Transitive_dataset.json"```
        - ```'../../../data/Reduced_words_complete_dataset.json"```
        - ```'../../../data/Reduced_words_transitive_dataset.json"```

# Alpha Functionalities

## BERT Sentence Embeddings

Using the BERT model, the word embeddings of each word in each sentence in an inputted dataset are calculated. These word embeddings serve as the initial parameters, after dimensionality reduction, of the parametrised quantum circuits that represent each sentence.

## DisCoCat Circuits

Using the lambeq package, a tket parametrised quantum circuit is generated for each sentence in the dataset.

## Dressed Quantum Circuit for Sentence Classification

The dressed quantum circuit forms our trainable PyTorch neural network. Initially the sentence embedding dimensionality is reduced to be compatible with the dimension of their respective parameters within the parametrised quantum circuits. This is achieved using sequential, cascasding linear transformations. Next the quantum circuit is run, after which the measurement outcomes are fed into a post-processing neural network which ultimately classifies the sentence as true or false.

## Torch OPtimisation

The dressed quantum circuit model is trained using Torch.
  




