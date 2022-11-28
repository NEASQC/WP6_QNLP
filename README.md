# Quantum Natural Language Processing : NEASQC WP6.1


## Installing locally

### Obtaining a local copy of the code repository
In order to run the code locally, you will need to obtain a copy of the repository. To this end, you can either fork the repository or clone it. 

#### Cloning
We here detail the procedure to be followed for cloning.

<ol>
  <li>Open the code repository in your browser.</li>
  <li>Open the drop-down menu on the leftClick on the 'Switch branches/tags' button to select v0.3 branch.</li>
  <li>Click on the green code button and choose the cloning method you want to use, GitHub provides detailes steps for each method (HTTPS, SSH, etc).</li>
  <li>Open a terminal on your computer and navigate to the directory you wish to clone the repository into. </li>
  <li>Run the following command in your terminal:
      <pre><code>$ git clone &ltcopied_url&gt</pre></code></li>
  <li>Navigate into the cloned repository by using 
     <pre><code>$ cd WP6_QNLP</pre></code> </li>
  <li>Run the following command in your terminal: 
      <pre><code>$ git checkout v0.3</pre></code></li>
</ol>


### Creating a new environment and installing required packages

#### Python version
The Python version required to run the scripts and notebooks of this repository is Python 3.10. Due to the presence of myQLM , only [python.org](https://www.python.org/downloads/macos/) and brew python distributions are supported and `pyenv` won't work.


<ol>
<li>If Python3.10 hasn't been installed (<em><strong>using brew</strong></em>) yet, or Python3.10 has been installed using any other method:
  <ol>
    <li>We run the following command on the terminal to install it on your local device.
      <pre><code>$ brew install python@3.10</pre></code></li>
    <li>By running the following command on the terminal, we make sure that we will link the recently installed Python3.10 to the environmental variable <em><strong>python3.10</em></strong>.
      <pre><code>$ brew link --overwrite python@3.10</pre></code>
    We may get an error if there was any other environmental variable named <em><strong>python3.10</em></strong>. In that case we must remove the variable from the PATH with the command: 
      <pre><code>$ unset python3.10</pre></code>
    and then use brew link command again.</li>
  </ol>
</li>

<li>If Python3.10 has been already installed (<em><strong>using brew</em></strong>):
  <ol>
    <li>We make sure that we have it linked to the the environmental variable <em><strong>python3.10</em></strong> using the command shown on section 1.2. A warning message will appear if we have it already linked (we can ignore it).</li>
    <li>We make sure that there are no packages installed on the global Python by running the command: 
      <pre><code>$ python3.10 -m pip list</pre></code>
    In the case where there were packages installed on the global Python we should uninstall them with the command: 
      <pre><code>$ python3.10 -m pip uninstall &ltundesired package&gt</pre></code></li>
  </ol>
</li>
</ol>





#### Virtual environment
Once you have cloned the repository, we recommend creating a virtual environment to run the code and notebooks. There are several tools to create a virtual environment, here we describe one of them and encourage users to resort to whichever approach is most familiar.
<ul>
  <li>To create a virtual environment, go to the directory where you want to create it and run the following command in the terminal:
    <pre><code>$ python3.10 -m venv &ltenvironment_name&gt</pre></code></li>
  <li> Once this has been done, the environment can be activated by running the following command:
    <pre><code>$ source &ltenvironment_name&gt/bin/activate</pre></code>
  If the environment has been activated correctly its name should appear in parentheses on the left of the user name in the terminal.</li>
  <li>Ensure pip is installed. If if not, follow instructions found <a href="https://pip.pypa.io/en/stable/installation/">here</a> to install it.</li>
  <li> To install the required packages, run the command:
    <pre><code>$ python3.10 -m pip install -r requirements.txt</pre></code></li>
</ul>

#### Spacy model
Some of the tools used in the module require a language model to be donwloaded by the user. This can be done running the following command:
  <pre><code>$ python3.10 -m spacy download en_core_web_lg</pre></code>
The language model will be stored in the created virtual environment. 


## Running the notebooks
We can use jupyter notebook to run the jupyter notebooks that appear on the repository. To do so, we can run the following command:
 <pre>
  <code>
    $ python3.10 -m ipykernel install --user --name  &ltenvironment_name&gt --display-name "&ltkernel_name&gt"
    $ python3.10 -m jupyter notebook &ltpath of the notebook we want to run&gt
  </code>
</pre>


The first command will define a kernel, named <kernel_name>, which you must change to after opening jupyter notebook. The second command will open a jupyter notebook terminal on our explorer, where we can run the selected notebook.
We will give now instructions for running each one of the notebooks, depending on the datasets that we want to use in our models. 

### [Classical_classifiers.ipynb](//repo//WP6_QNLP/neasqc_wp61/doc/tutorials/Classical_classifiers.ipynb)
  <ul>
    <li>In cell[2], the argument of <code>loadData()</code> must be one of the following:
      <ul>
        <li><code>"../../data/dataset_vectorized_bert_uncased.json"</code></li>
        <li><code>"../../data/dataset_vectorized_bert_cased.json"</code></li>
      </ul>
    </li>
    <li>In cell[8], the argument of <code>loadData()</code> must be:
      <ul>
        <li><code>"../../data/dataset_vectorized_fasttext.json"</code></li>
      </ul>
    </li>
  </ul>


### [Dataset_example.ipynb](//repo//WP6_QNLP/neasqc_wp61/doc/tutorials/Dataset_example.ipynb)
  <ul>
    <li>In cell[3], the value of the variable <code>filename</code> must be one of the following:
      <ul>
        <li><code>"Complete_dataset.json"</code></li>
        <li><code>"dataset_vectorized_bert_cased.json"</code></li>
        <li><code>"dataset_vectorized_bert_uncased.json"</code></li>
        <li><code>"dataset_vectorized_fasttext.json"</code></li>
        <li><code>"Expanded_Transitive_dataset.json"</code></li>
        <li><code>"Reduced_words_complete_dataset.json"</code></li>
      </ul>
    </li>
  </ul>
  

### [Single_sentence_example.ipynb](//repo//WP6_QNLP/neasqc_wp61/doc/tutorials/Single_sentence_example.ipynb)
  <ul>
    <li>No dataset is input on this notebook. No restrictions when running the notebook.</li>
  </ul>


### [Dressed_QNLP_demo.ipynb](//repo/WP6_QNLP/neasqc_wp61/models/quantum/alpha/Dressed_QNLP_demo.ipynb)
  <ul>
    <li>In cell[24], the value of the variable <code>filename</code> can be any of the datasets:
      <ul>
        <li><code>"../../../data/Complete_dataset.json"</code></li>
        <li><code>'../../../data/dataset_vectorized_bert_cased.json"</code></li>
        <li><code>'../../../data/dataset_vectorized_bert_uncased.json"</code></li>
        <li><code>'../../../data/dataset_vectorized_fasttext.json"</code></li>
        <li><code>'../../../data/Expanded_Transitive_dataset.json"</code></li>
        <li><code>'../../../data/Reduced_words_complete_dataset.json"</code></li>
        <li><code>'../../../data/Reduced_words_transitive_dataset.json"</code></li>
      </ul>
    </li>
  </ul>


# Pre-alpha Functionalities
The main scope of the pre-alpha model is to build a variational quantum algorithm that makes sentence classification in categories True or False. The structure of the analyzed sentences will be: 
<ul>
  <li>NOUN-TRANSITIVE VERB-NOUN</li>
  <li>NOUN-INTRANSITIVE VERB</li>
  <li>NOUN-INTRANSITIVE VERB-PREPOSITION-NOUN</li>
</ul>

## Classical classifiers

Some classical classifiers are implemented in order to have a reference against which to compare our quantum solution.
<ul>
  <li>K-nearest neighbors classifier from sklearn package applied to BERT embeddings.</li>
  <li>A feedforward neural network classifier from tensorflow package applied to BERT embeddings.</li>
  <li>A convolutional neural network classifier from tensorflow package applied to fasttext embeddings.</li>
</ul>

After the models are trained we compute and compare their accuracies on the training and test dataset. 

## Variational quantum circuit classfier

A variational quantum circuit is built with initial random parameters. In the circuit, only one qubit is measured (post-selection), and the variational parameters will be optimized using the labeled sentences in dataset so that the state obtained when measuring the qubit coincides with the value of the sentence (FALSE=0, TRUE=1). 

After the variational circuit is trained, the accuracy over the training and test dataset will be measured and compared. 

As an additional feature, a function that guesses a missing word in a sentence is also included. The function is applied to a new dataset generated by removing a noun from each of the sentences in the initial dataset. For a given sentence, the function will be looking for the missing word between all the words that have been removed, and will select the one whose circuit gives the greater probability of obtaining the state corresponding to the meaning of the sentence. 


# Pre-alpha Documentation

Let's briefly describe how the functionalities explained above are implemented. 

The classical classifiers are implemented in the notebook [Classical classifiers.ipynb](https://github.com/NEASQC/WP6_QNLP/blob/v0.3/neasqc_wp61/doc/tutorials/Classical_classifiers.ipynb). The file [NNClassifier.py](https://github.com/NEASQC/WP6_QNLP/blob/v0.3/neasqc_wp61/models/classical/NNClassifier.py) contains the class and functions used to prepare the data, build and train the convolutional and feedforward networks. 

Regarding the variational quantum circuit, in the notebook [Single_sentence_example.ipynb](https://github.com/NEASQC/WP6_QNLP/blob/v0.3/neasqc_wp61/doc/tutorials/Single_sentence_example.ipynb) we can find an example where the circuit parameters are optimized based on only one sentence. In the notebook [Dataset_example.ipynb](https://github.com/NEASQC/WP6_QNLP/blob/v0.3/neasqc_wp61/doc/tutorials/Dataset_example.ipynb), we can find an example on the variational algorithm trained with a whole dataset of sentences. We also can see there the implementation of a function to guess a missing word in a sentence. 

The functions and classes used to implement the variational quantum circuit are taken from different files. In [dictionary.py](https://github.com/NEASQC/WP6_QNLP/blob/v0.3/neasqc_wp61/models/quantum/pre-alpha/dictionary.py) we define classes that allow us to store the words than can appear on a sentence in dictionaries. Functions are defined that allow us to get and update the variational parameters associated to each word in the quantum circuit. Some of these functions are used in [sentence.py](https://github.com/NEASQC/WP6_QNLP/blob/v0.3/neasqc_wp61/models/quantum/pre-alpha/sentence.py), which provides the required tools to build the structure of the circuit that represents the sentence depending on its type and in some user-defined parameters. [ciruit.py](https://github.com/NEASQC/WP6_QNLP/blob/v0.3/neasqc_wp61/models/quantum/pre-alpha/circuit.py) contains functions that build, simplify (by qubit contractions) and execute the variational circuit. A class to optimize the variational parameters of the circuit with respect to a sentence or dataset of sentences can be found on [optimizer.py](https://github.com/NEASQC/WP6_QNLP/blob/v0.3/neasqc_wp61/models/quantum/pre-alpha/optimizer.py). Finally, in [loader.py](https://github.com/NEASQC/WP6_QNLP/blob/v0.3/neasqc_wp61/models/quantum/pre-alpha/loader.py) we can find functions that help in the processing of the datasets. 

# Alpha Functionalities

## Sentence Dataset and Model Purpose

The datatset used is [Complete_dataset.json](https://github.com/NEASQC/WP6_QNLP/blob/v0.3/neasqc_wp61/data/Complete_dataset.json), which contains sentences of varying grammatical structure that are labelled as true or false. The idea is to train a model, using this dataset, that can classify unknown sentences as true or false.
 
## BERT Sentence Embeddings

Using the BERT model, the word embeddings of each word in each sentence in an inputted dataset are calculated. These word embeddings serve as the initial parameters, after dimensionality reduction, of the parametrised quantum circuits that represent each sentence.

## DisCoCat Circuits

Using the lambeq package, a tket parametrised quantum circuit is generated for each sentence in the dataset.

## Dressed Quantum Circuit for Sentence Classification

The dressed quantum circuit forms our trainable PyTorch neural network. Initially the sentence embedding dimensionality is reduced to be compatible with the dimension of their respective parameters within the parametrised quantum circuits. This is achieved using sequential, cascasding linear transformations. Next the quantum circuit is run, after which the measurement outcomes are fed into a post-processing neural network which ultimately classifies the sentence as true or false.

## Torch OPtimisation

The dressed quantum circuit model is trained using Torch.
  
# Alpha Documentation

The alpha is currently implemented in one single jupyter notebook, [Dressed_QNLP_demo.ipynb](https://github.com/NEASQC/WP6_QNLP/blob/v0.3/neasqc_wp61/models/quantum/alpha/Dressed_QNLP_demo.ipynb). The beginning of the notebook contains a demonstration of how BERT sentence embeddings and parametrised quantum circuits can be generated using a small number of sample sentences. After this demonstration, and below the "Dressed Quantum Circuits" heading, two classes are defined which can be used in a pipeline that takes in our sentence dataset and generates a trainable model. After this generation, the model is trained using PyTorch. 

WARNING: You must run the initial demonstration as many of the functions and imports are used in the definition of the actual model, later in the notebook.

## Alpha Discussion

The idea behind this Jupyter Notebook is to extend the work found in https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html. In transfer learning, the first layers of a pretrained neural network are used to solve a different problem to that used to train the network, adding new layers to the model that specialise in that specific task.

We are using a BERT model to retrieve the context dependant embeddings for the words present in a sentence. Which layer is the best to retrieve the embeddings from is unclear, and it will need to be investigated. Once we have those vectors, they propagate through a feedforward network that first will reduce the dimensionality to an intermediate representation (in the notebook it is set to 20), and then the following layers will continue reducing the dimensionality of the vectors until reaching the number of parameters needed by the tensor representation of the quantum circuit offered by the Lambeq library for that word in a specific sentences.

Some benefits and issues of this approach are:

### Benefits
<ul>
  <li>Any sentence structure and word contained in the BERT model used can be processed by the full pipeline. No need to store the values of parameters for a dictionary</li>
  <li>It is possible to generalize to different NLP tasks</li>
  <li>If the dimensionaility of the category space is changed, the NN can be re-scaled to reuse the model for new circuit dimensionaility.</li>
</ul>  

### Issues
<ul>
  <li>Pytket loses the tensor nature of parameters, giving an output consisting of a list of floats or simply counts -> Differentiable circuits in Pennylane could be a solution.</li>
  <li>It is not clear if we gain any quantum advantage with this methods, as a classical NN has to be trained.</li>
</ul>

# Classical NLP

A module implementing classical processing of the dataset.

The NNClassifier.py for the classical NLP module is located in the
[classical](./neasqc_wp61/models/classical/) subdirectory.          

## Classical NLP Notebooks
The notebooks for the classical NLP module are located in the
[classical notebooks](./neasqc_wp61/doc/tutorials/) subdirectory.

# Benchmarking

## Vectorizer

Services are found [here](./neasqc_wp61/benchmarking/data_processing/) for vectorizing using pretrained word embeddings.

The aim is to have vectorizing service detached from the rest of the library so that different vectorizing methods can easily be tested using the same interface.

Currently, vectorizing with `BERT` and `fastText` models are implemented.

## Dataset Generation Example

### Generating the animal dataset
Manual dataset generation isn't necessary for running the Jupyter notebooks.
However, if needed for some different purpose, the dataset can be generated
using the following commands.

Run
<pre>
  <code>
# Choose a seed number, e.g, 1337, to deterministically randomize the sentence order
./neasqc_wp61/benchmarking/data_processing/gen_animal_dataset.py --seed 1337 > outfile
  </code>
</pre>
to generate a tab-separated file containing lines of the form
`<sentence>\t<sentence_type>\t<truth_value>` where `<truth_value>` is 1 if the sentence states a
fact that is true and 0 otherwise, and `<sentence_type>` denotes the sentence type, e.g., `NOUN-TVERB-NOUN`.

# Data

This subdirectory contains the relevant datasets used in the models. For example, consider the [complete_dataset.json](./neasqc_wp61/data/complete_dataset.json). In this dataset, a list of true or false labelled sentences of varying grammatical structure is contained. We can use this dataset in sentence classification training as in the alpha model.

# Doc
In this subdirectory,relevant documentation and tutorials are contained. Tutorials come in the form of jupyter notebooks and cover the classical and pre-alpha models.

# Models

Here we can find the Alpha and Pre-Alpha models.

# Acknowledgements

This work is supported by the [NEASQC](www.neasqc.eu/) project, funded by the European Union's Horizon 2020 programme, Grant Agreement No. 951821.

## License

The [LICENSE](./LICENSE) file contains the default license statement as specified in the proposal and partner agreement.  




