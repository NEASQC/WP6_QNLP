# Quantum Natural Language Processing : NEASQC WP 6.1


## Installing locally

### Obtaining a local copy of the code repository
In order to run the code locally, you will need to obtain a copy of the repository. To this end, you can either fork the repository or clone it. 

#### Cloning
We here detail the procedure to be followed for cloning.

  1. Open the code repository in your browser.
  2. Open the drop-down menu on the left. Click on the 'Switch branches/tags' button to select v0.3 branch.
  3. Click on the green code button and choose the cloning method you want to use, GitHub provides detailes steps for each method (HTTPS, SSH, etc).
  4. Open a terminal on your computer and navigate to the directory you wish to clone the repository into. 
  5. Run the following command in your terminal:
      <pre><code>$ git clone &ltcopied_url&gt</pre></code></li>
  6. Navigate into the cloned repository by using 
     <pre><code>$ cd WP6_QNLP</pre></code> </li>
  7. Run the following command in your terminal: 
      <pre><code>$ git checkout dev</pre></code></li>


### Creating a new environment and installing required packages

#### Python version
The Python version required to run the scripts and notebooks of this repository is Python 3.10. Due to the use of myQLM in one of our models, only [python.org](https://www.python.org/downloads/macos/) and `brew` python distributions are supported. `conda` and `pyenv` won't work.


1. If Python3.10 hasn't been installed (<em><strong>using brew</strong></em>) yet, or Python3.10 has been installed using any other method:
    * Install brew following the instructions detailed [here](https://brew.sh/). 
    * Run the following command on the terminal to install it on your local device.
      <pre><code>$ brew install python@3.10</pre></code>
    * By running the following command on the terminal, we make sure that we will link the recently installed Python3.10 to the environmental variable <em><strong>python3.10</em></strong>.
      <pre><code>$ brew link --overwrite python@3.10</pre></code>
      We may get an error if there was any other environmental variable named <em><strong>python3.10</em></strong>. In that case we must remove the variable from the PATH with the command: 
      <pre><code>$ unset python3.10</pre></code>
      and then use brew link command again.
2. If Python3.10 has been already installed (<em><strong>using brew</em></strong>):
    * We make sure that we have it linked to the the environmental variable <em><strong>python3.10</em></strong> using the command shown on section 1.2. A warning message will appear if we have it already linked (we can ignore it).
    * We make sure that there are no packages installed on the global Python by running the command: 
      <pre><code>$ python3.10 -m pip list</pre></code>
      In the case where there were packages installed on the global Python we should uninstall them with the command: 
      <pre><code>$ python3.10 -m pip uninstall &ltundesired package&gt</pre></code>



#### Dependencies

##### Poetry
Note: we here assume that you are happy to use `poetry`'s lightweight virtual environenment set-up. If for some reason you prefer to use an external virtual environemnt, simply activate it before using `poetry`, it will respect its precedence.
<ol>
  <li> Make sure you have <code>poetry</code> installed locally. This can be done by running  <pre><code>$ poetry --version</pre></code> in your shell and checking the output. If installed, proceed, if not, follow instructions on their official website <a href="https://python-poetry.org/docs/#installation">here</a>. </li>
  <li> <code>cd</code> to the root of the repository where the files <code>pyproject.toml</code> and <code>poetry.lock</code> are located. </li>
  <li> Run the following command in your shell: <pre><code>$ poetry install</pre></code>
  If you also want to install the dependancies used to build sphinx documentation, run the following command insted:
  <pre><code>poetry install --with docs</pre></code></li>
</ol>


##### Virtualenv
Note: As mentioned previously, one of our models uses myQLM, which will not work on a virtual env. However, all other models should work without issues.
<ol>
  <li>To create a virtual environment, go to the directory where you want to create it and run the following command in the terminal:
    <pre><code>$ python3.10 -m venv &ltenvironment_name&gt</pre></code></li>
  <li> Activate the environment (see instructions <a href="#venv_activation">here</a>). If the environment has been activated correctly its name should appear in parentheses on the left of the user name in the terminal.</li>
  <li>Ensure pip is installed. If if not, follow instructions found <a href="https://pip.pypa.io/en/stable/installation/">here</a> to install it.</li>
  <li> To install the required packages, run the command:
    <pre><code>$ python3.10 -m pip install -r requirements.txt</pre></code></li>
</ol>




#### Activating the virtual environments

##### Poetry
To activate <code>poetry</code>'s default virtual environment, simply run:
<pre><code>poetry shell</code></pre>
inside your terminal. More details can be found <a href="https://python-poetry.org/docs/basic-usage/#using-your-virtual-environment">here</a>.

##### Virtualenv
To activate your virtualenv, simply type the following in your terminal:
<pre><code>$ source &ltenvironment_name&gt/bin/activate</pre></code>
Note that contrary to <code>poetry</code>, this virtual environment needs to be activated before you install the requirements.

## Models

### pre-Alpha Models

We have two pre-Alpha models on this repository, pre-Alpha 1 (<code>pre_alpha</code>) and pre-Alpha 2 (<code>pre_alpha_lambeq</code>). 

Pre-Alpha 1 classifies sentences by creating variational quantum circuits using <code>myQLM</code>, where the measurement of one or more qubits (sentence-type qubits) will dictate the prediction. Pre-Alpha 2 does the same but using the QNLP Python library <code>lambeq</code>, which allows for a more concise implementation of the pre-Alpha pipeline, as well as for the tweaking of additional parameters.

The functions and classes used to implement the variational quantum circuits for pre-Alpha 1 are taken from different files. In [dictionary.py](https://github.com/NEASQC/WP6_QNLP/blob/dev/neasqc_wp61/models/quantum/pre-alpha/dictionary.py) we define classes that allow us to store the words than can appear on a sentence in dictionaries. Functions are defined that allow us to get and update the variational parameters associated to each word in the quantum circuit. Some of these functions are used in [sentence.py](https://github.com/NEASQC/WP6_QNLP/blob/dev/neasqc_wp61/models/quantum/pre-alpha/sentence.py), which provides the required tools to build the structure of the circuit that represents the sentence depending on its type and on some user-defined parameters. [ciruit.py](https://github.com/NEASQC/WP6_QNLP/blob/dev/neasqc_wp61/models/quantum/pre-alpha/circuit.py) contains functions that build, simplify (by qubit contractions) and execute the variational circuit. A class to optimize the variational parameters of the circuit with respect to a sentence or dataset of sentences can be found on [optimizer.py](https://github.com/NEASQC/WP6_QNLP/blob/dev/neasqc_wp61/models/quantum/pre-alpha/optimizer.py). Finally, in [loader.py](https://github.com/NEASQC/WP6_QNLP/blob/dev/neasqc_wp61/models/quantum/pre-alpha/loader.py) we can find functions that help in the processing of the datasets. The pipeline for this model is then implemented in [use_pre_alpha.py](https://github.com/NEASQC/WP6_QNLP/blob/dev/neasqc_wp61/data/data_processing/use_pre_alpha.py).

For pre-Alpha 2, all functions and classes are located in [PreAlphaLambeq.py](https://github.com/NEASQC/WP6_QNLP/blob/dev/neasqc_wp61/models/quantum/pre-alpha-lambeq/PreAlphaLambeq.py), and the pipeline is implemented in [use_pre_alpha_lambeq.py](https://github.com/NEASQC/WP6_QNLP/blob/dev/neasqc_wp61/data/data_processing/use_pre_alpha_lambeq.py). 


### Alpha Models

We have 3 Alpha models: Alpha 1 (<code>alpha_pennylane_lambeq_original</code>), Alpha 2 (<code>alpha_pennylane_lambeq</code>) and Alpha 3 (<code>alpha_pennylane</code>).

Alpha 1 extends the pre-Alpha models by utilizing BERT-based word and sentence embeddings. It maps these embeddings to quantum parameters in parameterized quantum circuits, reducing dimensionality through PCA and linear layers.

Alpha 2, similar to Alpha 1, merges classical network and pre-Alpha 2 architecture. It diverges in its input and preprocessing, using sentences to create parameterized quantum circuits and sentence BERT embeddings. These embeddings are dimensionally reduced through linear layers and used as parameters in the quantum circuits.

For both Alpha 1 and 2, the functions and classes of the model are located in [alpha_pennylane_lambeq_model.py](https://github.com/NEASQC/WP6_QNLP/blob/dev/neasqc_wp61/models/quantum/alpha/module/alpha_pennylane_lambeq_model.py), and those of the trainer are located in [alpha_pennylane_lambeq_trainer.py](https://github.com/NEASQC/WP6_QNLP/blob/dev/neasqc_wp61/models/quantum/alpha/module/alpha_pennylane_lambeq_trainer.py). The pipeline for both models is implemented in [use_alpha_lambeq.py](https://github.com/NEASQC/WP6_QNLP/blob/dev/neasqc_wp61/data/data_processing/use_alpha_lambeq.py).

Alpha 3 follows a "dressed quantum architecture", meaning that it combines a classical network architecture with a quantum circuit. It takes sentence BERT embeddings as input and employs a shared quantum circuit, known for its expressibility advantages. Unlike Alpha 1 and Alpha 2, it does not generate a different quantum circuit for each sentence, making it substantially faster to run.

The functions and classes of the model are located in [alpha_pennylane_model.py](https://github.com/NEASQC/WP6_QNLP/blob/dev/neasqc_wp61/models/quantum/alpha/module/alpha_pennylane_lambeq_model.py) and those of the trainer are located in [alpha_pennylane_trainer.py](https://github.com/NEASQC/WP6_QNLP/tree/dev/neasqc_wp61/models/quantum/alpha/module). The pipeline is implemented in [use_alpha_pennylane.py](https://github.com/NEASQC/WP6_QNLP/blob/dev/neasqc_wp61/data/data_processing/use_alpha_pennylane.py).

All Alpha models use <code>PyTorch</code>'s Adam optimiser. We plan to add a parameter that allows users to use their <code>PyTorch</code> optimiser of choice in future releases.


### Beta Model

Beta classifies sentences using a quantum version of the K-Nearest Neighbours (KNN) algorithm. The vectors we input are BERT embeddings reduced using PCA. The module [QuantumDistance.py](https://github.com/NEASQC/WP6_QNLP/blob/dev/neasqc_wp61/models/quantum/beta/QuantumDistance.py) computes the quantum distance and the module [QuantumKNearestNeighbours_bert.py](https://github.com/NEASQC/WP6_QNLP/blob/dev/neasqc_wp61/models/quantum/beta/QuantumKNearestNeighbours_bert.py) uses that distance to implement the KNN algorithm. The pipeline for the model is implemented in [use_beta_neighbours_with_bert.py](https://github.com/NEASQC/WP6_QNLP/blob/dev/neasqc_wp61/data/data_processing/use_beta_neighbours_with_bert.py).

### Tutorials

We have a number of tutorials tutorials located in the [doc/tutorials](https://github.com/NEASQC/WP6_QNLP/tree/dev/neasqc_wp61/doc/tutorials) and the [models/quantum/alpha/jupyter_tutorial](https://github.com/NEASQC/WP6_QNLP/tree/dev/neasqc_wp61/models/quantum/alpha/jupyter_tutorial) directories in the form of Jupyter Notebooks.

[pre_alpha_lambeq_example.ipynb](https://github.com/NEASQC/WP6_QNLP/blob/dev/neasqc_wp61/doc/tutorials/pre_alpha_lambeq_example.ipynb) is a tutorial on the pre-Alpha 2 model.

For the Alpha 1, 2 and 3 models, the corresponding tutorials are [trainer_quantum_pennylane_lambeq_original.ipynb](https://github.com/NEASQC/WP6_QNLP/blob/dev/neasqc_wp61/models/quantum/alpha/jupyter_tutorial/trainer_quantum_pennylane_lambeq_original.ipynb), [trainer_quantum_pennylane_lambeq.ipynb](https://github.com/NEASQC/WP6_QNLP/blob/dev/neasqc_wp61/models/quantum/alpha/jupyter_tutorial/trainer_quantum_pennylane_lambeq.ipynb) and [trainer_quantum_pennylane.ipynb](https://github.com/NEASQC/WP6_QNLP/blob/dev/neasqc_wp61/models/quantum/alpha/jupyter_tutorial/trainer_quantum_pennylane.ipyn) respectively. Each of these has a corresponding 'multiclass' version located in the [Multiclass_classification](https://github.com/NEASQC/WP6_QNLP/tree/dev/neasqc_wp61/models/quantum/alpha/jupyter_tutorial/Multiclass_classification) directory, in which the Alpha models are used for a multiclass sentence classification problem.

[beta_neighbours_example.ipynb](https://github.com/NEASQC/WP6_QNLP/blob/dev/neasqc_wp61/doc/tutorials/beta_neighbours_example.ipynb) is a tutorial on the Beta model.

To run these, one must execute the following:
 <pre>
  <code>
    $ python3.10 -m ipykernel install --user --name  &ltenvironment_name&gt --display-name "&ltkernel_name&gt"
    $ python3.10 -m jupyter notebook &ltpath of the notebook we want to run&gt
  </code>
</pre>

The first command will define a kernel, named <kernel_name>, which you must change to after opening <code>Jupyter Notebook</code>. The second command will open a <code>Jupyter Notebook</code> terminal on our explorer, where we can run the selected notebook.



### Running the Models

#### Parameter Descriptions

##### Pre Alpha Models

1 = Pre Alpha Original, 2 = Pre Alpha Lambeq

| Parameter | Command Line Tag | Model Version |
|-----------|------------------|---------------|
| Train dataset  | t | 1, 2 |
| Test dataset  | v | 1, 2 |
| Validation dataset  | j | 1, 2 |
| Seed  | s | 1, 2 |
| Model  | m | 1, 2 |
| Runs  | r | 1, 2 |
| Optimiser   | p | 1, 2 |
| Iterations  | i | 1, 2 |
| Outfile  | o | 1, 2 |
| Ansatz   | a | 2 |
| Number of qubits per noun   | q | 2 |
| Number of circuit layers   | n | 2 |
| Number of single qubit parameters  | x | 2 |
| Batch size | b | 2 |

##### Alpha Models 

1 = Alpha Original, 2 = Alpha Lambeq 
, 3 = Alpha Pennylane

| Parameter | Command Line Tag | Model Version | 
|-----------|------------------|---------------|
| Train dataset  | t | 1, 2, 3 |
| Test dataset | v | 1, 2, 3 |
| Validation dataset  | j | 1, 2, 3 |
| Seed  | s | 1, 2, 3 |
| Model  | m | 1, 2, 3 |
| Runs  | r | 1, 2, 3 |
| Iterations  | i | 1, 2, 3 |
| Outfile  | o | 1, 2, 3 |
| Ansatz   | a | 1, 2, 3 |
| Number of qubits per NOUN type   | q | 1, 2, 3 |
| Number of circuit layers   | n | 1, 2, 3 |
| Number of single qubit parameters  | x | 1, 2, 3 |
| Number of qubits in our circuit  | u | 3 |
| Initial spread of the parameters   | d | 3 |
| Batch size  | b | 1, 2, 3 |
| Learning rate   | l | 1, 2, 3 |
| Weight decay  | w | 1, 2, 3 |
| Step size for the learning rate scheduler  | z | 1, 2, 3 |
| Gamma for the learning rate scheduler  | g | 1, 2, 3 |
| Version choice of alpha_pennylane_lambeq or alpha_pennylane_lambeq_original   | y | 1, 2 |
| Reduced dimension for the word embeddings | c | 1, 2 |
| Number of qubits per SENTENCE type  | e | 1, 2 |

##### Beta Model


| Parameter | Command Line Tag | 
|-----------|------------------|
| Train dataset   | t |
| Test dataset   | v |
| Number(s) of clusters   | k | 
| Outfile  | o |

#### Command Line Examples

##### Pre-Alpha
```
bash 6_Classify_With_Quantum_Model.sh -m pre_alpha
-t ./data/datasets/reduced_amazonreview_train.tsv
-v ./data/datasets/reduced_amazonreview_test.tsv
-j ./data/datasets/reduced_amazonreview_val.tsv
-s 13 -r 10 -i 150 -p COBYLA
-o ./benchmarking/results/raw/
```
##### Pre-Alpha-Lambeq
```
bash 6_Classify_With_Quantum_Model.sh -m pre_alpha_lambeq
-t ./data/datasets/reduced_amazonreview_train.tsv
-v ./data/datasets/reduced_amazonreview_test.tsv
-j ./data/datasets/reduced_amazonreview_val.tsv
-s 13 -p Adam -i 150 -r 10 -a IQP -q 1 -n 1 -x 3 -b 512
-o ./benchmarking/results/raw/
```
##### Alpha-Original
```
bash 6_Classify_With_Quantum_Model.sh -m alpha_lambeq
-t ./data/datasets/reduced_amazonreview_train_word.csv
-j ./data/datasets/reduced_amazonreview_dev_word.csv 
-v ./data/datasets/reduced_amazonreview_test_word.csv
-s 13 -r 10 -i 150 -b 512 -l 2e-1 -w 0 -z 150 -g 1
-y alpha_pennylane_lambeq_original -c 22 -e 1 -a IQP
-q 1 -n 1 -x 3 -o ./benchmarking/results/raw/
```

##### Alpha-Lambeq
```
bash 6_Classify_With_Quantum_Model.sh -m alpha_lambeq
-t ./data/datasets/reduced_amazonreview_train_sentence.csv
-j ./data/datasets/reduced_amazonreview_dev_sentence.csv
-v ./data/datasets/reduced_amazonreview_test_sentence.csv
-s 13 -r 2 -i 150 -b 512 -l 2e-1 -w 0 -z 150 -g 1
-y alpha_pennylane_lambeq -c 22 -e 1 -a IQP -q 1 -n 1 -x 3
-o ./benchmarking/results/raw/
```

##### Alpha-Pennylane
```
bash 6_Classify_With_Quantum_Model.sh -m alpha_pennylane
-t ./data/datasets/reduced_amazonreview_train_sentence.csv
-j ./data/datasets/reduced_amazonreview_dev_sentence.csv
-v ./data/datasets/reduced_amazonreview_test_sentence.csv
-s 13 -r 10 -i 150 -u 3 -d 0.01 -b 512 -l 2e-1 -w 0 -z 150
-g 1 -o ./benchmarking/results/raw/
```

##### Beta QKNN
```
bash 6_Beta_Neighbours.sh
-t ./data/datasets/reduced_amazonreview_train_sentence.csv
-v ./data/datasets/reduced_amazonreview_test_sentence.csv
-k "1 3 5 7 9" -o ./benchmarking/results/raw/
```

#### Plotting Results

Navigate to <code>neasqc_wp61/benchmarking/results_processing/</code>. Then run the following command for a JSON file containing the results for an experiment for any model, except for the Beta model. Assume a file with name <code>output.json</code> located in neasqc_wp61/benchmarking/results/raw/. Then use:

```
python3.10 plot_single_experiments_results.py ../results/raw/output.json
```

In the case of a file <code>beta_output.json</code> containing the results for a Beta model experiment, we instead run the following to plot the results:
```
python3.10 /benchmarking/results_processing/plot_beta_experiment.py /output/path/results.json
```

## Running Models on HPC Systems
Here we give a brief description of how to run the different models included in this repo on an HPC cluster which uses SLURM as its workload manager. 

To clone the repo on the cluster the instructions are the same as those above for cloning into a local machine. The setting up of the environment may be different however. Installing brew or Python from source require sudo permissions, which a user does not have when using a cluster. Here we outline an approach that we have followed on Kay, ICHEC's supercomputer, which should also work on similar HPC systems.

### Setting Up the Environment
We will assume that the user has a working directory with path `/path/work_dir`. 

<ol>
  <li> Assuming Conda is available as a module, load the module using 
  <pre>
    <code> $ module load conda </code>
  </pre></li>
  <li> Create a directory <code> /path/work_dir/.conda</code>. This directory will store the environment.</li>
  <li> Create a Conda environment (example name: neasqc_env), using this path as a prefix, i.e.
  <pre>
    <code> $ conda create --prefix /path/work_dir/.conda/neasqc_env python=3.10 </code>
  </pre></li>
  <li> Once the environment has been created, activate it using 
  <pre>
    <code> $ conda activate /path/work_dir/.conda/neasqc_env </code>
  </pre>
  or, for older versions of Conda,
  <pre>
    <code> $ source activate /path/work_dir/.conda/neasqc_env </code>
  </pre></li>
  <li> Once the environment is activated, install poetry using
  <pre>
    <code> $ conda install -c conda-forge poetry </code>
  </pre></li>
  <li> Now that conda is installed, you can follow the same instructions that we gave before for installing in your local machine. <code> cd</code> into the root of the repo where the files <code>pyproject.toml</code> and <code>poetry.lock</code> are located and run the command
  <pre>
    <code> $ poetry install </code>
  </pre>
  You can add the <code>--with docs</code> flag at the end of the command as before if you wish to install the dependancies used to build sphinx documentation.</li>
</ol>

To activate the environment we do not have to use `poetry shell` because Poetry has installed the dependencies within the Conda environment, rather than within a Poetry shell. Thus, every time you want to activate the environment you must use the 
appropriate command given above in Step 4.


### Running HPC Jobs

We give instructions here for submitting jobs on an HPC cluster which uses SLURM for the management and scheduling of jobs. If the system you are working on uses an alternative workload manager, please refer to its documentation for instructions on submitting jobs. 

Say you want to run 3 different jobs, one with the `pre_alpha` model, one with the `pre_alpha_lambeq` model and one with the `beta_neighbours` model. Assume that you want to request a maximum walltime of 1 hour on 1 HPC node for each of these (this is currently the default option). Now say you have chosen to run the jobs with the following parameters:

<ul>
  <li> <code> pre_alpha</code> with seed=200, optimiser=COBYLA, iterations of the optimiser=1000, runs of the model=100.</li>
  <li> <code> pre_alpha_lambeq</code> with seed=200, optimiser=AdamW, iterations of the optimiser=1000, runs of the model=100, ansatz=IQP, qubits per noun=1, number of circuit layers=1, number of single qubit parameters=3 and batch size=512. Note that currently we have restrcited qubits per sentence=1 as default.</li>
  <li><code>alpha_lambeq</code> (which is split in two models, Alpha 1 and Alpha 2), with seed=200, runs of the model=100, iterations of the optimiser=1000, version=Alpha 1 (<code>alpha_pennylane_lambeq_original</code>), dimensions after PCA=10, qubits per noun=1, qubits per sentence=1 number of circuit layers=1, number of single qubit parameters=3, batch size=512, learning rate=0.01, weight decay=0, step learning rate=150 and gamma for the larning rate scheduler=1. Note that the optimiser is fixed by default to Adam. </li>
  <li><code>alpha_lambeq</code> with version=Alpha 2 (<code>alpha_pennylane_lambeq</code>) and all other parameters equal to the above. As above, the optimiser is fixed to Adam.</li>
  <li>Alpha 3 (<code>alpha_pennylane</code>) with seed=200, runs of the model=100, iterations of the (Adam) optimiser=1000, number of qubits in the circuit=3, initial spread of the parameters=0.01, batch size=512, learning rate=0.001, weight decay=0, step learning rate=150, gamma for the learning rate scheduler=1.</li>
  <li> <code> beta_neighbours</code> with number of neighbours for the KNN algorithm=2. </li>
</ul>

Then:

<ol>
  <li> Create a file with the name <code>benchmark_params.txt</code> in the <code>.../benchmarking/hpc</code> directory in your local copy of the repo. If the file exists skip this step and delete its contents before proceeding.</li>
  <li> This file is where we state our choice of models and corresponding parameters that we want to submit to the cluster. Assuming the above parameters, open the file and write the following:
  <pre>
    pre_alpha 200 COBYLA 1000 100
    pre_alpha_lambeq 200 AdamW 1000 100 IQP 1 1 3 512
    alpha_lambeq 200 100 1000 alpha_pennylane_lambeq_original 10 1 1 1 3 512 0.01 0 150 1
    alpha_lambeq 200 100 1000 alpha_pennylane_lambeq 10 1 1 1 3 512 0.01 0 150 1
    alpha_pennylane 200 100 1000 3 0.01 512 0.0001 0 150 1
    beta_neighbours 2
  </pre>
  Note that when inputting your own parameters to the file, they must be written in the same order as above. If less/more parameters are introduced for a given model choice, this will lead to warning messages later on and the line will be skipped.</li>
  <li> Save and close the file.</li>
  <li> Within the same directory, and with the environment activated, run the <code>generate_slurm_scripts.py</code> script by entering the command
  <pre>
    <code> $ python generate_slurm_scripts.py </code>
  </pre></li>
  <li> This script will create 6 SLURM scripts, one for each job. The names of the generated scripts will be in the format <code>[model]_[param_1]_..._[param_n].sh</code>, with the parameters in the same order as they were stated in the <code>benchmark_params.txt</code> file. These scripts will be located in the <code>slurm_scripts</code> subdirectory.</li>
  <li> To submit the jobs associated with a SLURM script <code>script_name.sh</code> to the HPC compute nodes, simply <code>cd</code> into the <code>slurm_scripts</code> directory where it is located and run
  <pre>
    <code>$ sbatch script_name.sh</code>
  </pre>
  Alternatively, if you want to submit multiple jobs to the queue simultaenously, you can run the <code>submit_jobs.sh</code> script in <code>.../benchmarking/hpc</code> by doing
  <pre>
  $ ./submit_jobs.sh
  </pre>
  This script submits every script contained within the <code>slurm_scripts</code> directory, and moves them to the subdirectory <code>slurm_scripts/scripts_archive</code> to keep the parent directory clean and only containing scripts that are yet to be submitted. <br />
  NOTE: if you get a permission error when trying to run the script, you can fix the permissions by running
  <pre>
  $ chmod u+x submit_jobs.sh
  </pre>
  within the directory in which the script is contained.
  </li>
</ol>
Please note that the scripts are generated with a default of 1 node per job, 1 hour maximum walltime and ProdQ as the queue of choice (this is a queue in ICHEC's supercomputer, Kay). Additionally, by default the scripts will be generated to run using the reduced amazonreview datasets. If you wish to change the HPC specifications, ot to use the full datasets, or any other dataset, you must edit the SLURM script templates located in the <code>.../hpc/slurm_templates</code> directory. These tools that we have described above are mainly intended for internal use, so they may appear somewhat confusing and restrictive to the external user. We hope to make these tools more user-friendly and to offer more automated ways to alter the parameters for different jobs in future updates to the repository.