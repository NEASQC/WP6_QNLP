# WP6

## Download the code
To download the code as a local copy to run the code in your own machine, it is recommended to clone the repository using **git**.

 * To see the different releases codes, make sure you are viewing the **releases** branch. Click on the drop down list with the branch symbol at the top left and select this branch. 
 * To navigate between different versions click on the drop down list again and select the **tags** tab. There you can access the available releases (for now pre-alpha prototype is the only one).
 * Then click on the green **code** button and copy the HTTPS link.
 * To download an specific release, open a terminal in your computer and run the following command: 
```console
your@user:~$ git clone --depth 1 --releases <tag_name> <copied_url>
```
 * The code will be downloaded to the terminal directory

## Repository structure: Pre alpha and Alpha gaps

The repository contains code for the pre alpha and the alpha prototype. For the former, a task consisting of true/false sentence classification was implemented, and
it can be found under the ´pre_alpha´ folder, along with example jupyter notebooks showing the training pipeline.
For the latter, some research was made about hybrid quantum-classical algorithms and how quantum could fit in more complex NLP algorithm architectures. This lead to
creating what XanaduAI calls Dressed Quantm Circuits. For this purpose, pretrained classical models are used as preprocessing layers in a transfer learning fashion. 
We wanted to implement Neural Networks able to transform pretrained word vectors into usable lower dimensional vectors acting as the parameters for the rotations of
variational quantum circuits. A first iteration (WIP) of this approach can be found under the folder ´alpha´. A Juyter Notebook can be found where context dependant BERT vectors 
are retrieved for the words in a sentence, and a neural network reduce the dimensionality of the parameters to encode the sentence as a DisCoCat circuit.


## Jupyter notebooks

An example of the pre-alpha implementation is found in  `my_lib/alpha/DressedQuantumCircuit` directory, and more examples for the pre-alpha are located in ``my_lib/pre_alpha/notebooks` containing examples demonstrating how the quantum
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
[./my_lib/pre_alpha/notebooks/classical/requirements.txt](./my_lib/pre_alpha/notebooks/classical/requirements.txt)
– should be used.

An additional [./my_lib/alpha/DressedQuantumCircuit/requirements.txt](./requirements.txt) needs to be installed to run the Notebook for the Alpha protoype demonstration.

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
./my_lib/src/DataPreparation/gen_animal_dataset.py --seed 1337 > outfile
```
to generate a tab-separated file containing lines of the form
`<sentence>\t<sentence_type>\t<truth_value>` where `<truth_value>` is 1 if the sentence states a
fact that is true and 0 otherwise, and `<sentence_type>` denotes the sentence type, e.g., `NOUN-TVERB-NOUN`.
