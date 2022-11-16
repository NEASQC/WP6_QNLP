# Classical NLP

A module implementing classical processing of the dataset.

# Data preparation

An explanation of this module

# Vectorizer

Services used for vectorizing using pretrained word embeddings.

The aim is to have vectorizing service detached from the rest of the library so that different vectorizing methods can easily be tested using the same interface.

Currently, vectorizing with `BERT` and `fastText` models are implemented.

### Setup

The vectorizing services are built as a `Docker` container and instructions for building the images are contained in the respective `Dockerfile`s. Therefore, a simple `docker build -t <service_name> .` command should suffice to build the image.

# Neural Networks in Dressed Quantum Circuits for NLP


## Dressed Quantum Circuits

The idea behind this Jupyter Notebook is to extend the work found in https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html. In transfer learning, the first layers of a pretrained neural network are used to solve a different problem to that used to train the network, adding new layers to the model that specialise in that specific task. 

We are using a BERT model to retrieve the context dependant embeddings for the words present in a sentence. Which layer is the best to retrieve the embeddings from is unclear, and it will need to be investigated. Once we have those vectors, they opropagate through a feedforward network that first will reduce the dimensionaility to an intermediate representation (in the notebook it is set to 20), and then the following layers will continue reducing the dimensionality of the vectors until reaching the number of parameters needed by the tensor representation of the quantum circuit offered by the Lambeq library for that word in a specific sentences. 

Some benefits of this approach are:

* Any sentence structure and word contained in the BERT model used can be processed by the full pipeline. No need to store the values of parameters for a dictionary
    
* It is possible to generalize to different NLP tasks
    
* If the dimensionaility of the category space is changed, the NN can be re-scaled to reuse the model for new circuit dimensionaility.
    
    

Issues
    
* Pytket loses the tensor nature of parameters, giving an output consisting of a list of floats or simply counts -> Differentiable circuits in Pennylane could be a solution.
    
* It is not clear if we gain any quantum advantage with this methods, as a classical NN has to be trained. 

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

# Quantum

An implementation of DisCoCat model for asserting the thruth value of sentences.

* sentence.py: **Sentence** class contains all the information for a sentence to be written as a quantum circuit: the number of qubits per word, the categories assigned to each part of speech, the parameters for the rotations and the qubits that need to be contracted. Also, some methods are provided to manipulate the parameters of sentences.

* dictionary.py:  Provides the information and parameters of the DisCoCat model. The different parameterizations, Ansatz shapes, and grammar-related information is contained in **QuantumDict** and **QuantumWord** classes.

* circuit.py: **CirctuiBuilder** class can use sentences to build the quantum circuit as QLM programs, submit jobs and apply the needed post-processing to sample output states. 

* optimizer.py: **ClassicalOptimizer** contains the cost functions for quantum circuits, and ue SciPy optimizers to train quantum circuits.

* loader.py: Used to read the .json files and create DataFrames used in the training process.

Currently, supported sentence structures are:

    - Noun+TransitiveVerb+Noun 
    - Adjective+Noun+TransitiveVerb+Adjective+Noun 
    - Noun+IntransitiveVerb+Preposition+Noun 
    - Noun+IntransitiveVerb

Contractions for these structures are done following the information contained in dictionaries within the code when the sentence type is given. As sentences grow in size and diversity, pregroup rules must be programmed to perform the needed contractions automatically.

# Quantum Natural Language Processing

This repository is associated to the use case of Quantum Natural Language Processing of the European Quantum Flagship project NEASQC.

## License

The [LICENSE](./LICENSE) file contains the default license statement as specified in the proposal and partner agreement.

## Jupyter notebooks

The `misc/notebooks` directory contains examples demonstrating how the quantum
and classical NLP modules are used to solve NLP-related problems (currently – a
sentence classification task).

### Setup
To run the notebooks you must install some software dependencies and download a
`spacy` language model. Due to some library incompatibilities, the quantum and
classical NLP modules currently cannot share a common set of software
dependencies and, as a result, need to be run in separete Python environments.
The instructions below describe the steps to set up an environment for the
**quantum NLP module** using the [requirements.txt](./requirements.txt) file in
the project's root. The steps for setting up an environement for the **classical
NLP module** are identical except for the fact that a different requirements
file –
[./misc/notebooks/classical/requirements.txt](./misc/notebooks/classical/requirements.txt)
– should be used.

#### Dependencies
We recommend using `virtualenv` or `anaconda` to create an isolated environment
for the project's dependencies. Here are instructions for `virtualenv`.

```sh
# Install virtualenv using your distribution's package manager or pip, e.g.,
apt install virtualenv  # if using Ubuntu

# Create a new virtual environment called, e.g., venv.
# This will create a directory called venv in the project root which will
# contain all of the packages installed in the following steps. It can be safelly
# delted to get rid of the these packages.
virtualenv venv

# Activate the newly created environment
. ./venv/bin/activate

# Install the dependencies into the virtual environment
pip install -r requirements.txt
```

##### A note on the Python version
We've tested that the code works using Python 3.9. Earlier versions might not
work on Linux due to incompatibility with `myqlm`.

#### Spacy language model
Some of the tools used in the quantum NLP module require a language model to be downloaded to
the users system. To do it automatically, execute the following command into the
virtual environment after the dependencies have been installed (see the
Dependencies section above).

```sh
# Make sure the virtual environement is activated
. ./venv/bin/activate

# The language model will be stored in the venv directory
python -m spacy download en_core_web_lg
```

### Running
To run the the notebooks, start jupyter lab and navigate to the notebooks in
[misc/notebooks/](./misc/notebooks/)

```sh
# Make sure the virtual environement is activated
. ./venv/bin/activate

# Start jupyter lab and follow the instructions in the console
# output to open jupyter lab in the browser
jupyter-lab
```

### More information
Some more information about the notebooks is provided in
[misc/notebooks/README.md](./misc/notebooks/README.md)

## Generating the animal dataset
Manual dataset generation isn't necessary for running the Jupyter notebooks.
However, if needed for some different purpose, the dataset can be generated
using the following commands.

Run
```sh
# Choose a seed number, e.g, 1337, to deterministically randomize the sentence order
./QNLP_lib/src/DataPreparation/gen_animal_dataset.py --seed 1337 > outfile
```
to generate a tab-separated file containing lines of the form
`<sentence>\t<sentence_type>\t<truth_value>` where `<truth_value>` is 1 if the sentence states a
fact that is true and 0 otherwise, and `<sentence_type>` denotes the sentence type, e.g., `NOUN-TVERB-NOUN`.

## Acknowledgements

This work is supported by the [NEASQC](www.neasqc.eu/) project, funded by the European Union's Horizon 2020 programme, Grant Agreement No. 951821.
