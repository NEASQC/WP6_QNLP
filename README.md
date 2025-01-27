

# Quantum Natural Language Processing : NEASQC WP 6.1
This readme gives a brief introduction to the release candidate of the [NEASQC](https://www.neasqc.eu/) work-package 6.1 on quantum natural language processing and provides guidance on how to run and compare different models (both classical natural language processing (NLP) models and quantum NLP (QNLP) models). The release candidate is a benchmarking solution that allows to:

* download and pre-process data for quantum and classical NLP model training,
* Train quantum and classical NLP models for text classification (we have set up scripts such that each model is trained 30 times for statistical analysis).
* Evaluate the trained models on held-out test sets.

We give a brief overview of the following models:
* Quantum models Beta2/3 (for Alpha3, refer to the [previous deliverable](https://www.neasqc.eu/wp-content/uploads/2023/11/NEASQC_D6_10_QNLP.pdf) and the experimental implementation [here](https://github.com/NEASQC/WP6_QNLP/tree/qnlp_models_d6_13)).
* Classical NLP models. Although, we have experimented with neural network models trained on top of pre-trained word-level (using convolutional neural networks) and sentence-level (using a feed-forward neural network) embeddings as well as a scenario without pre-trained embeddings (using a long short-term memory neural network) (see experimental results [here](https://github.com/tilde-nlp/neasqc_wp6_qnlp/tree/v2-classical-nlp-models/neasqc_wp61/doc) and [here](https://github.com/tilde-nlp/neasqc_wp6_qnlp/tree/classical-nlp-models/neasqc_wp61/doc)), in the release candidate, we focused only on sentence-level pretrained embeddings that showed to achieve better results.

## Setup

### Pre-requisites
Prior to following any steps, you should ensure that you have on your local machine and readily available:
- A copy of the repository.
- `python 3.10`, our models *might* turn out to be compatible with later versions of python but they were designed with and intended for 3.10.
- `poetry`, you can follow the instructions if needed on <a href="https://python-poetry.org/docs/#installation">the official website</a>.

### Getting started
1. Position yourself in the `final_main` branch.
2. Position yourself in the root of the repository where the files `pyproject.toml` and `poetry.lock` are located.
3. Run <pre><code>poetry install</pre></code>
4. Activate `poetry` using <pre><code>poetry shell</code></pre> More details can be found <a href="https://python-poetry.org/docs/basic-usage/#using-your-virtual-environment">here</a>.
5. Download the `en_core_web_sm` model for `spacy` (which is used for data preparation) using the command <pre><code>python -m spacy download en_core_web_sm</pre></code>

## Workflow

The Workflow has 8 steps. Data preparation steps (Step 0 to Step 5) are common to the classical and the quantum algorithms. Model training and benchmarking is performed in Step 6 and combination of results from all experiments is performed in Step 7.

To run each step, use the corresponding bash script that is located in the directory `neasqc_wp61`. 

### Step 0 - data download

We use the following datasets from the kaggle.com. 

- `Reviews.csv` from <https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/versions/1> (we will name this resource after download and processing as `amazon-fine-food-reviews.csv`)
- `labelled_newscatcher_dataset.csv` from <https://www.kaggle.com/datasets/kotartemiy/topic-labeled-news-dataset> (we will name this resource after download and processing as `labelled_newscatcher_dataset.csv`)
- `train.csv` from <https://www.kaggle.com/datasets/kk0105/ag-news> (we will name this resource after download and processing as `ag_news.csv`)
- `RAW_interactions.csv` from <https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions> (we will name this resource after download and processing as `food-com-recipes-user-interactions.csv`)
- `train.csv` from <https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews> (we will name this resource as `amazon-reviews.csv`)
- `News_Category_Dataset_v3.json` from <https://www.kaggle.com/datasets/rmisra/news-category-dataset> (we will name this resource after download and processing as `huffpost-news.tsv`)

To retrive the datasets from the kaggle.com you first have to

- install the kaggle API using the command `pip install --user kaggle`
- create a Kaggle account, create an API token, make a directory `.kaggle` in the home `~` directory, and place `kaggle.json` in that directory.  
	
See instructions in <https://www.endtoend.ai/tutorial/how-to-download-kaggle-datasets-on-ubuntu/>

Then, step into the `neasqc_wp61` folder and run the script `0_FetchDatasets.sh` to download the datasets and convert them into a unified format (CSV). All further scripts will be executed from this folder.

### Step 1 - Tokenising, filtering, and normalising datasets

After downloading datasets, we tokenise the data using the `spacy` tokeniser, extract class labels and text (as all other columns are not needed), and perform normalisation (e.g., decoding of HTML entitities and fixing broken Unicode symbols in some datasets). To perform this step, run the script `1_Filter6Parse.sh` passing the following parameters:

- `-i <dataset>`           Comma-separated (CSV) data file
- `-d <delimiter>`          Field delimiter symbol
- `-c <class fiels>`        Name of the class field (only if the first line in the file contains field names)
- `-t <text field>`         Name of the text field (only if the first line in the file contains field names)
- `-o <output file>`            Output file in a tab-separated format (TSV)

Examples:

```bash
bash 1_Filter6Parse.sh -i ./data/datasets/amazon-fine-food-reviews.csv -o ./data/datasets/amazon-fine-food-reviews_tok.tsv -d ',' -c 'Score' -t 'Summary'

bash 1_Filter6Parse.sh -i ./data/datasets/labelled_newscatcher_dataset.csv -o ./data/datasets/labelled_newscatcher_dataset_tok.tsv -d ';' -c 'topic' -t 'title'

bash 1_Filter6Parse.sh -i ./data/datasets/ag_news.csv -o ./data/datasets/ag_news_tok.tsv -d ','

bash 1_Filter6Parse.sh -i ./data/datasets/food-com-recipes-user-interactions.csv -o ./data/datasets/food-com-recipes-user-interactions_tok.tsv -d ',' -c 'rating' -t 'review'

bash 1_Filter6Parse.sh -i ./data/datasets/amazon-reviews.csv -o ./data/datasets/amazon-reviews_tok.tsv -d ','

bash 1_Filter6Parse.sh -i ./data/datasets/huffpost-news.csv -o ./data/datasets/huffpost-news_tok.tsv -d ','
```

### Step 2 - Re-balancing the dataset

We train classical NLP and QNLP models using balanced datasets, such that each class would be equally represented in the dataset. To perform this step, run the script `2_BalanceClasses.sh` passing the following parameters:

- `-i \<input file\>`               Tab-separated (TSV) 2-column file to rebalance.
- `-c \<classes to ignore\>`		  Optionally classes to ignore in dataset separated by `,`.

Examples:

```bash
bash 2_BalanceClasses.sh -i ./data/datasets/amazon-fine-food-reviews_tok.tsv

bash 2_BalanceClasses.sh -i ./data/datasets/labelled_newscatcher_dataset_tok.tsv -c "SCIENCE"

bash 2_BalanceClasses.sh -i ./data/datasets/ag_news_tok.tsv

bash 2_BalanceClasses.sh -i ./data/datasets/food-com-recipes-user-interactions_tok.tsv

bash 2_BalanceClasses.sh -i ./data/datasets/amazon-reviews_tok.tsv

#We drop quite a few classes from the dataset as they have rather few examples
bash 2_BalanceClasses.sh -i ./data/datasets/huffpost-news_tok.tsv -c "ARTS,ARTS & CULTURE,COLLEGE,CULTURE & ARTS,EDUCATION,ENVIRONMENT,FIFTY,GOOD NEWS,GREEN,LATINO VOICES,MEDIA,MONEY,RELIGION,SCIENCE,STYLE,TASTE,TECH,U.S. NEWS,WEIRD NEWS,WORLDPOST,category"
```
	
### Step 3 - Spliting data in train/dev/test parts

Next, we split the corpus into 3 separate (non-overlapping) parts for training, validation, and evaluation of models. To perform this step, run the script `3_SplitTrainTestDev.sh` passing the following parameters:

- `-i \<input file\>`         Tab-separated (TSV) 2-column filtered file

3 files fill be created containing suffices `_train`, `_test`, and `_dev` in their names.

Example:

```bash
bash 3_SplitTrainTestDev.sh -i ./data/datasets/amazon-fine-food-reviews_tok_balanced.tsv

bash 3_SplitTrainTestDev.sh -i ./data/datasets/labelled_newscatcher_dataset_tok_balanced.tsv

bash 3_SplitTrainTestDev.sh -i ./data/datasets/ag_news_tok_balanced.tsv

bash 3_SplitTrainTestDev.sh -i ./data/datasets/food-com-recipes-user-interactions_tok_balanced.tsv

bash 3_SplitTrainTestDev.sh -i ./data/datasets/amazon-reviews_tok_balanced.tsv

bash 3_SplitTrainTestDev.sh -i ./data/datasets/huffpost-news_tok_balanced.tsv
```

### Step 4 - Acquiring embedding vectors using chosen pre-trained embedding model

We have experimented with two differentpre-trained embedding models (featuring less than 1B parameters) that have shown to allow achieving various levels of down-stream task accuracies (based on the [Massive Text Embedding Benchmark](https://huggingface.co/spaces/mteb/leaderboard)) .

| Pre-trained model | Parameters (millions) | Embedding type | Vectors for unit | About model |
|---|---|---|---|---|
| *ember-v1* | 335 | transformer    | sentence         | 1024-dimentional sentence transformer model. <br><https://huggingface.co/llmrails/ember-v1> |
| *bert-base-uncased* | 110 | bert           | sentence         | Case insensitive model pretrained on the BookCorpus (consisting of 11,038 unpublished books) and English Wikipedia; 768-dimensional sentence transformer model.<br><https://huggingface.co/bert-base-uncased> |

BERT models are older and slower than sentence transformer models. We compare the Transformer-based models also with an older skip-gram model (for word embedding; sentence embeddings are obtained by combining individual word embeddings) using `fasttext`.

To perform this step, run the script `4_GetEmbeddings.sh` passing the following parameters:

- `-i <input file>`      input file with text examples
- `-o <output file>`     output json file with embeddings
- `-c <column>`          `2` - if 2-column input file containing class and text columns, `0` - if the whole line is a text example
- `-m <embedding name>`  Name of the embedding model
- `-t <embedding model type>`  Type of the embedding model - `fasttext`, `transformer`, or `bert`	
- `-e <embedding unit>`  Embedding unit: `sentence` or `word`	
- `-g <gpu use>`         Number of GPUs to use (from `0` to available GPUs), `-1` if CPU shall be used (default)

The `fasttext` model works only on CPU.

Embedding unit is `word` or `sentence` for the `bert` models; `sentence` for the `transformer` and `fasttext` models. 

Run this step for all 3 parts of the dataset - train, dev and test. We have set up a bash script to generate embeddings for all data sets used in this repository and the three sentence-level embedding models - `4_1_GetEmbeddingsForAllExampleSets.sh`. However, if you wish to run the script on individual data files, here is an example:

```bash
bash 4_GetEmbeddings.sh -i ./data/datasets/labelled_newscatcher_dataset_tok_balanced_test.tsv -o ./data/datasets/labelled_newscatcher_dataset_tok_balanced_test_ember.json -c '2' -m 'llmrails/ember-v1' -t 'transformer' -e 'sentence' -g '1'
```
The output data of the vectoriser is stored using the following JSON format (and accepted as input data by the classification model training step 5 below):

```json
[
  {
    "class": "2",
    "sentence": "Late rally keeps Sonics ' win streak alive",
    "sentence_vectorized": [
      [
        1.2702344502208496,
        -1.109272517088847,
        0.729442862056898,
        -2.0236876799667654,
        -2.1608433294597083
      ]
    ]
  },
  {
    "class": "2",
    "sentence": "Fifth - Ranked Florida State Holds Off Wake Forest",
    "sentence_vectorized": [
      [
        4.42427794195838,
        2.71155631975937,
        -0.2619018355289202,
        -2.7190369563754815,
        -2.021439642325012
      ]
    ]
  }
]
```

### Step 5 - Reducing dimensions of embeddings

Since quantum computing models cannot process vectors of large dimensionality, we apply feature selection to reduce dimensions of embeddings. Embedding dimensions can be reduced using the python script `reduce_emb_dim.py` in the folder `./data/data_processing`. It uses dimensionality reduction classes from the script `dim_reduction.py`. 

Run the script using the following parameters:

- `-it \<input train file\>` 	 JSON input training file with embeddings (acquired using the script `4_GetEmbeddings.sh`)
- `-ot \<output train file\>`     JSON output training file with reduced embeddings
- `-iv \<input dev file\>` 	 JSON input validation file with embeddings (acquired using the script `4_GetEmbeddings.sh`)
- `-ov \<output dev file\>`     JSON output validation file with reduced embeddings
- `-ie \<input test file\>` 	 JSON input evaluation file with embeddings (acquired using the script `4_GetEmbeddings.sh`)
- `-oe \<output test file\>`     JSON output evaluation file with reduced embeddings
- `-n \<number of dimensions\>` Desired number of dimensions in the output vectors
- `-a \<reduction algorithm\>` Dimensionality reduction algorithm (possible values are `PCA`, `ICA`, `TSVD`, `UMAP` or `TSNE`)

Example:

`python ./data/data_processing/reduce_emb_dim.py -i ./data/datasets/reviews_filtered_train_ember.json -o ./data/datasets/reviews_filtered_train_100ember.json -n 100 -a PCA`

To reduce all datasets with `PCA` to `3`, `5`, and `8` (maximum for the quantum models) dimensions, use the script `5_ReduceDimensionsForAllDatasets.sh`.

### Step 6 - Training models

Once data is pre-processed, we can start training models. The following sub-sections document how to train both classical NLP and quantum NLP models.

#### Classical NLP

The folder `./models/classical` contains the source code of a class implementing neural network classifiers. For classical NLP, we implemented three classifiers - a shallow feed-forward neural network (FFNN; for sentence-level embeddings), a convolutional neural network (CNN; for word-level embeddings), and a bidirectional long short-term memory (LSTM) neural network (when pre-trained embeddings are not used).

To perform this step, run the script *6_TrainClassicalNNModel.sh* passing the following parameters:

- `-t \<train data file\>` JSON data file for classifier training (with embeddings) or TSV file (if not using pre-trained embeddings, acquired using the script `3_SplitTrainTestDev.sh`)
- `-d \<dev data file\>`   JSON data file for classifier validation (with embeddings) or TSV file (if not using pre-trained embeddings, acquired using the script `3_SplitTrainTestDev.sh`)
- `-f \<field\>`           Field in the JSON object by which to classify text
- `-e \<embedding unit\>`  Embedding unit: `sentence`, `word`, or `-` (if not using pre-trained embeddings)
- `-m \<model directory\>` Directory where to save the trained model (the directory will be created if not existing)
- `-g \<gpu use\>`         Number of the GPU to use (from `0` to available GPUs), `-1` if CPU should be used (default is `-1`)
	
Each model is trained 30 times (runs) for statistical analysis.

Example:

`bash ./6_TrainClassicalNNModel.sh -t ./data/datasets/labelled_newscatcher_dataset_tok_balanced_train_llmrails_ember-v1.json -d ./data/datasets/labelled_newscatcher_dataset_tok_balanced_dev_llmrails_ember-v1.json -e ./data/datasets/labelled_newscatcher_dataset_tok_balanced_test_llmrails_ember-v1.json -f 'class' -e 'sentence' -m ./benchmarking/results/raw/classical/labelled_newscatcher_dataset_tok_balanced_train_llmrails_ember-v1 -g '0'`

To train classical NLP models for all datasets, all 3 sentence-level embedding models, and all reduced dimensions, run the script `6_1_TrainClassicalNNModelsForAllDatasets.sh`. This script will train 30 classifiers for each training dataset using early stopping based on the validation set. When training each classifier, the best-performing checkpoint (i.e., checkpoint achieving the highest accuracy on the validation set) will be evaluated using a test set. Results for each training dataset will be stored in separate `results.json` files. 

The format of the `results.json` files is as follows:
```json
{
  "input_args": {
    "runs": 0,
    "iterations": 0
  },
  "best_val_acc": 0,
  "best_run": 0,
  "time": [..., ...],
  "train_acc": [[..., ...], ... [..., ...]],
  "train_loss": [[..., ...], ... [..., ...]],
  "val_acc": [[..., ...], ... [..., ...]],
  "val_loss": [[..., ...], ... [..., ...]],
  "test_acc": [..., ...],
  "test_loss": [..., ...]
}
```

#### Quantum NLP

Beta 2 and 3 models follow what we call a *semi-dressed quantum circuit* (SDQC) architecture. Here, the first layer of a DQC is stripped. The classical input is handed directly to the PQC once it has been brought to the correct dimension. The input to the circuit is a dimensionality-reduced sentence embedding, i.e. a vector of size N where N is the number of qubits in the quantum circuit (the code supports up to 8 dimensions). The advantage of this model is that it relies more heavily on quantum elements as compared with a DQC. Refer to the [D6.10: WP6 QNLP Report](https://www.neasqc.eu/wp-content/uploads/2023/11/NEASQC_D6_10_QNLP.pdf) for a more detailed description of the models.

The Beta 2 and 3 model architecture is defined in `neasqc_wp61/models/quantum/beta_2_3/beta_2_3_model.py`.

A model can be trained using the script `6_TrainQuantumModel.sh`; the following parameters must be specified as command line arguments:

* `-t` : the path to the training dataset.
* `-j` : the path to the validation dataset.
* `-v` : the path to the test dataset.
* `-o` : output directory path (for model(s) and results).
* `N` : the number of qubits of the fully-connected quantum circuit.
* `-s` : the initial spread of the quantum parameters (we recommend setting this to 0.01 initially).
* `-i` : the number of iterations (epochs) for the training of the model.
* `-b` : the batch size.
* `-w` : the weight decay (this can be set to 0).
* `-x` : an integer seed for result replication.
* `-p` : the `pytorch` optimiser of choice.
* `-l` : the learning rate for the optimiser.
* `-z` : the step size for the learning rate scheduler.
* `-g` : the gamma for the learning rate scheduler.
* `-r` : the number of runs of the model (each run will be initialised with a different seed determined by the -x parameter).

Example:
  ```bash
  bash 6_TrainQuantumModel.sh -t ./data/datasets/labelled_newscatcher_dataset_tok_balanced_train_llmrails_ember-v1_3.json -j ./data/datasets/labelled_newscatcher_dataset_tok_balanced_dev_llmrails_ember-v1_3.json -v ./data/datasets/labelled_newscatcher_dataset_tok_balanced_test_llmrails_ember-v1_3.json -p Adam -x 42 -r 1 -i 10 -N 8 -s 0.01 -b 320 -l 0.002 -w 0 -z 150 -g 1 -o ./benchmarking/results/raw/quantum/labelled_newscatcher_dataset_tok_balanced_train_llmrails_ember-v1_3_beta_2_3
  ```
Note that the trainer file is `neasqc_wp61/models/quantum/beta_2_3/beta_2_3_trainer_tests.py` and the pipeline file is `neasqc_wp61/data/data_processing/use_beta_2_3_tests.py`.

The script `6_2_TrainQuantumModelsForAllDatasets.sh` can be used to train and evaluate QNLP models for all pre-processed datasets. Note that this may require ~5-48 hours per dataset.

### Step 7 - Combining results
Results from individual `results.json` files can be combined using the `combine-results.py` script using the following parameters:
* `-base_dir <DIR>` base directory where `results.json` files must be searched (all sub-directories will be analysed).
* `-res_dir` output directory where TSV files with the combined results will be stored.

Example:
```bash
python data/data_processing/combine-results.py ./benchmarking/results/raw/ ./benchmarking/results/analysed
```

The script will create the following files:
* `time.tsv` - average training times of the 30 runs with a 95% confidence interval for each dataset.
* `train_loss.tsv` - average training loss scores of the 30 runs with a 95% confidence interval for each dataset and each best-performing model according to the validation set.
* `train_acc.tsv` - average training accuracy scores of the 30 runs with a 95% confidence interval for each dataset and each best-performing model according to the validation set.
* `val_loss.tsv` - average validation loss scores of the 30 runs with a 95% confidence interval for each dataset and each best-performing model according to the validation set.
* `val_acc.tsv` - average validation accuracy scores of the 30 runs with a 95% confidence interval for each dataset and each best-performing model according to the validation set.
* `test_loss.tsv` - average test loss scores of the 30 runs with a 95% confidence interval for each dataset and each best-performing model according to the validation set.
* `test_acc.tsv` - average test accuracy scores of the 30 runs with a 95% confidence interval for each dataset and each best-performing model according to the validation set.

Each TSV file consists of 5 columns:
* Path of the `results.json` file
* Mean score
* Lower bound (mean - 95% confidence interval)
* Upper bound (mean + 95% confidence interval)
* Confidence interval


## Results

### labelled_newscatcher_dataset

The test set consisted of 8736 examples featuring 7 classes.

| Model          | Embedding model   | Dimensionality | Test mean ± CI | Dev mean ± CI | Train mean ± CI | Seconds per model ± CI |
| -------------- | ----------------- | -------------- | -------------- | ------------- | --------------- | ---------------------- |
| Classical FFNN | ember-v1          | full           | **80.89±0.06** | 80.13±0.38    | 81.16±1.69      | 13.22±1.52             |
| Classical FFNN | fasttext          | full           | 76.26±0.07     | 74.43±0.68    | 74.64±0.98      | 25.27±0.34             |
| Classical FFNN | bert-base-uncased | full           | 75.49±0.05     | 75.44±0.18    | 76.41±0.43      | 28.88±1.23             |
| Classical FFNN | ember-v1          | 8              | 69.14±0.09     | 69.05±0.63    | 68.73±0.89      | 21.75±0.19             |
| QNLP Beta 2/3  | ember-v1          | 8              | 65.38±0.03     | 64.32±1.59    | 62.42±5.26      | 923.14±12.73           |
| Classical FFNN | bert-base-uncased | 8              | 56.47±0.08     | 55.84±0.69    | 55.25±0.9       | 21.55±0.2              |
| Classical FFNN | fasttext          | 8              | 55.27±0.16     | 54.21±0.93    | 54.53±1.05      | 21.87±0.21             |
| QNLP Beta 2/3  | bert-base-uncased | 8              | 52.41±0.04     | 50.3±1.97     | 48.04±4.02      | 899.48±1.45            |
| QNLP Beta 2/3  | fasttext          | 8              | 46.05±0.27     | 41.85±3.22    | 40.8±5.15       | 894.89±1               |
| Classical FFNN | ember-v1          | 5              | 66.03±0.08     | 65.24±0.97    | 65±1.22         | 21.62±0.25             |
| QNLP Beta 2/3  | ember-v1          | 5              | 63.63±0.03     | 61.63±1.74    | 59.84±5.03      | 910.57±11.6            |
| Classical FFNN | bert-base-uncased | 5              | 44.75±0.06     | 44.2±0.5      | 44.01±0.65      | 20.93±0.21             |
| QNLP Beta 2/3  | bert-base-uncased | 5              | 41.31±0.05     | 39.5±1.35     | 38.75±2.73      | 891.34±1.42            |
| Classical FFNN | fasttext          | 5              | 38.34±0.09     | 37.36±0.43    | 38.43±0.49      | 21.45±0.19             |
| QNLP Beta 2/3  | fasttext          | 5              | 33.86±0.24     | 29.65±1.35    | 29.19±3.07      | 890.01±1.04            |
| Classical FFNN | ember-v1          | 3              | 55.2±0.08      | 55.03±0.58    | 54.7±0.85       | 21.58±0.19             |
| QNLP Beta 2/3  | ember-v1          | 3              | 51.98±0.13     | 50.09±1.4     | 48.92±2.9       | 982.85±15.14           |
| Classical FFNN | fasttext          | 3              | 36.57±0.05     | 35.46±0.26    | 36.02±0.35      | 21.61±0.19             |
| QNLP Beta 2/3  | fasttext          | 3              | 32.27±0.36     | 28.24±0.99    | 27.99±2.58      | 960.96±0.78            |
| Classical FFNN | bert-base-uncased | 3              | 32.23±0.1      | 31.99±0.2     | 32.28±0.29      | 21.53±0.2              |
| QNLP Beta 2/3  | bert-base-uncased | 3              | 31.17±0.08     | 29.74±0.54    | 29.59±1.73      | 963.02±2.09            |

### food-com-recipes-user-interactions

The test set consisted of 6385 examples featuring 5 classes.

| Model          | Embedding model   | Dimensionality | Test mean ± CI | Dev mean ± CI | Train mean ± CI | Seconds per model ± CI |
| -------------- | ----------------- | -------------- | -------------- | ------------- | --------------- | ---------------------- |
| Classical FFNN | ember-v1          | full | **63.36±0.08** | 63.15±0.21 | 63.9±0.82  | 8.39±1.21    | 0.633563038 |
| Classical FFNN | fasttext          | full | 51.57±0.11 | 51.01±0.46 | 50.34±0.56 | 19.68±0.18   | 0.515672141 |
| Classical FFNN | bert-base-uncased | full | 50.94±0.11 | 49.74±0.27 | 51.63±0.58 | 19.12±1.72   | 0.509449232 |
| Classical FFNN | ember-v1          | 8    | 61.53±0.09 | 60.95±0.65 | 59.9±0.75  | 16.26±0.17   | 0.615343245 |
| QNLP Beta 2/3  | ember-v1          | 8    | 57.36±0.06 | 52.98±1.74 | 51.17±4.24 | 684.42±10.05 | 0.573552597 |
| Classical FFNN | fasttext          | 8    | 40.81±0.18 | 40.74±0.45 | 39.74±0.49 | 16.15±0.17   | 0.408071    |
| Classical FFNN | bert-base-uncased | 8    | 36.05±0.12 | 34.39±0.32 | 35.03±0.38 | 16.43±0.19   | 0.360480292 |
| QNLP Beta 2/3  | bert-base-uncased | 8    | 35.04±0.06 | 33.09±0.42 | 33.42±1.57 | 653.77±0.87  | 0.350394153 |
| QNLP Beta 2/3  | fasttext          | 8    | 33.83±0.31 | 31.48±1.86 | 29.1±2.44  | 653.3±0.98   | 0.338339859 |
| Classical FFNN | ember-v1          | 5    | 61.26±0.09 | 60.66±0.48 | 59.6±0.64  | 16.13±0.14   | 0.612623332 |
| QNLP Beta 2/3  | ember-v1          | 5    | 57.24±0.07 | 54.22±1.24 | 52.1±3.99  | 673.69±10.36 | 0.572440616 |
| Classical FFNN | fasttext          | 5    | 35.36±0.12 | 35.03±0.38 | 33.89±0.36 | 16.11±0.15   | 0.353636128 |
| Classical FFNN | bert-base-uncased | 5    | 32.63±0.1  | 32.35±0.21 | 32.78±0.29 | 15.98±0.12   | 0.326275122 |
| QNLP Beta 2/3  | bert-base-uncased | 5    | 32.27±0.05 | 31.4±0.21  | 31.63±1.19 | 645.21±0.76  | 0.322704255 |
| QNLP Beta 2/3  | fasttext          | 5    | 31.78±0.21 | 30.66±1.24 | 28.7±2.06  | 651.48±1.14  | 0.317781258 |
| Classical FFNN | ember-v1          | 3    | 57.28±0.13 | 56.09±0.55 | 55.05±0.71 | 16.54±0.15   | 0.572832161 |
| QNLP Beta 2/3  | ember-v1          | 3    | 53.72±0.16 | 50.7±1.19  | 48.99±3.1  | 729.66±13.7  | 0.53724354  |
| Classical FFNN | fasttext          | 3    | 34.11±0.08 | 33.99±0.15 | 33.02±0.22 | 16.29±0.11   | 0.341143302 |
| Classical FFNN | bert-base-uncased | 3    | 31.26±0.11 | 31.17±0.25 | 30.64±0.27 | 16.12±0.18   | 0.312649436 |
| QNLP Beta 2/3  | fasttext          | 3    | 31.15±0.2  | 29.46±1.22 | 27.7±1.78  | 703.36±1.18  | 0.311537458 |
| QNLP Beta 2/3  | bert-base-uncased | 3    | 30.69±0.08 | 30.34±0.26 | 30.27±0.86 | 700.55±0.76  | 0.306927695 |

### huffpost-news

The test set consisted of 7238 examples featuring 22 classes.

| Model          | Embedding model   | Dimensionality | Test mean ± CI | Dev mean ± CI | Train mean ± CI | Seconds per model ± CI |
| -------------- | ----------------- | -------------- | -------------- | ------------- | --------------- | ---------------------- |
| Classical FFNN | ember-v1          | full | **64.05±0.15** | 64.33±0.6  | 65.72±1.33 | 22.71±1.95  |
| Classical FFNN | bert-base-uncased | full | 53.6±0.21  | 53.41±0.75 | 54.67±0.99 | 26.27±0.71  |
| Classical FFNN | fasttext          | full | 46.23±0.37 | 44.89±1.08 | 45.09±1.19 | 21.79±0.26  |
| Classical FFNN | ember-v1          | 8    | 42.74±0.12 | 42.49±0.86 | 42.5±1.01  | 18.74±0.19  |
| QNLP Beta 2/3  | ember-v1          | 8    | 35.68±0.08 | 32.66±2.03 | 32.2±4.01  | 745.73±1.43 |
| Classical FFNN | bert-base-uncased | 8    | 21.45±0.12 | 20.6±0.42  | 21.08±0.5  | 18.37±0.18  |
| Classical FFNN | fasttext          | 8    | 20.47±0.15 | 19.74±0.52 | 19.47±0.54 | 18.15±0.16  |
| QNLP Beta 2/3  | bert-base-uncased | 8    | 19.06±0.06 | 16.74±1.19 | 16.84±1.7  | 740.58±0.77 |
| QNLP Beta 2/3  | fasttext          | 8    | 14.04±0.22 | 10.32±1.59 | 10.01±1.73 | 745.79±1.81 |
| Classical FFNN | ember-v1          | 5    | 35.2±0.11  | 34.99±0.58 | 35.24±0.69 | 18.07±0.17  |
| QNLP Beta 2/3  | ember-v1          | 5    | 26.3±0.29  | 24.43±2.19 | 23.48±3    | 743.77±1.27 |
| Classical FFNN | fasttext          | 5    | 17.57±0.1  | 17.59±0.31 | 17.71±0.35 | 18.26±0.18  |
| Classical FFNN | bert-base-uncased | 5    | 16.39±0.07 | 16.81±0.24 | 16.52±0.26 | 18.14±0.2   |
| QNLP Beta 2/3  | bert-base-uncased | 5    | 14.21±0.08 | 13.24±0.69 | 13.31±1.24 | 737.83±0.88 |
| QNLP Beta 2/3  | fasttext          | 5    | 13.54±0.19 | 10.68±1.2  | 10.3±1.53  | 739.18±1.66 |
| Classical FFNN | ember-v1          | 3    | 25.81±0.09 | 25.99±0.43 | 25.52±0.5  | 18.21±0.19  |
| QNLP Beta 2/3  | ember-v1          | 3    | 19.43±0.26 | 16.93±1.15 | 16.65±1.25 | 801.47±1.61 |
| Classical FFNN | fasttext          | 3    | 13.91±0.05 | 13.51±0.2  | 13.52±0.23 | 18.37±0.18  |
| Classical FFNN | bert-base-uncased | 3    | 12.8±0.07  | 13.38±0.2  | 13.4±0.23  | 18.52±0.19  |
| QNLP Beta 2/3  | bert-base-uncased | 3    | 11.7±0.08  | 10.6±0.82  | 10.96±1.06 | 797.71±0.73 |
| QNLP Beta 2/3  | fasttext          | 3    | 10.31±0.1  | 8.96±0.71  | 8.99±0.97  | 795.31±1.67 |

### amazon-fine-food-reviews

The test set consisted of 8415 examples featuring 5 classes.

| Model          | Embedding model   | Dimensionality | Test mean ± CI | Dev mean ± CI | Train mean ± CI | Seconds per model ± CI |
| -------------- | ----------------- | -------------- | -------------- | ------------- | --------------- | ---------------------- |
| Classical FFNN | ember-v1          | full | **52.31±0.1**  | 52.25±0.24 | 52.92±0.92 | 6.45±0.35   |
| Classical FFNN | bert-base-uncased | full | 45.88±0.08 | 45.84±0.17 | 47.46±0.39 | 17.84±1.81  |
| Classical FFNN | fasttext          | full | 45.26±0.07 | 45.06±0.25 | 44.82±0.33 | 24.27±0.41  |
| Classical FFNN | ember-v1          | 8    | 51.1±0.06  | 50.81±0.33 | 50.15±0.38 | 20.96±0.15  |
| QNLP Beta 2/3  | ember-v1          | 8    | 48.04±0.02 | 46.69±0.74 | 45.84±2.74 | 868.87±1.91 |
| Classical FFNN | fasttext          | 8    | 36.83±0.09 | 37.32±0.29 | 36.59±0.32 | 20.87±0.24  |
| QNLP Beta 2/3  | fasttext          | 8    | 36.14±0.08 | 34.22±1.3  | 33.14±2    | 868.59±1.09 |
| Classical FFNN | bert-base-uncased | 8    | 34.39±0.09 | 34.61±0.29 | 34.39±0.33 | 21±0.19     |
| QNLP Beta 2/3  | bert-base-uncased | 8    | 33±0.05    | 32.65±0.26 | 32.59±0.84 | 860.32±1.18 |
| Classical FFNN | ember-v1          | 5    | 50.91±0.07 | 50.87±0.17 | 50.24±0.23 | 20.45±0.18  |
| QNLP Beta 2/3  | ember-v1          | 5    | 48.15±0.08 | 46.95±0.33 | 46.35±2.14 | 859.43±1.47 |
| Classical FFNN | fasttext          | 5    | 32.75±0.08 | 32.66±0.14 | 32.41±0.19 | 20.94±0.23  |
| QNLP Beta 2/3  | fasttext          | 5    | 32.2±0.09  | 29.94±0.95 | 29.78±1.44 | 860.69±0.83 |
| Classical FFNN | bert-base-uncased | 5    | 31.4±0.14  | 31.62±0.2  | 31.76±0.27 | 20.42±0.16  |
| QNLP Beta 2/3  | bert-base-uncased | 5    | 30.14±0.06 | 30.41±0.48 | 29.58±0.89 | 857.57±1.43 |
| Classical FFNN | ember-v1          | 3    | 50.69±0.07 | 50.66±0.15 | 49.74±0.3  | 20.65±0.45  |
| QNLP Beta 2/3  | ember-v1          | 3    | 48.46±0.06 | 47.24±0.44 | 46.17±2.16 | 929.39±2.06 |
| Classical FFNN | fasttext          | 3    | 29.51±0.05 | 28.12±0.1  | 28.27±0.1  | 20.9±0.16   |
| QNLP Beta 2/3  | fasttext          | 3    | 29.01±0.11 | 26.54±0.6  | 26.88±0.88 | 931.62±0.83 |
| Classical FFNN | bert-base-uncased | 3    | 28.43±0.11 | 28.72±0.19 | 28.9±0.24  | 19.66±0.93  |
| QNLP Beta 2/3  | bert-base-uncased | 3    | 28.05±0.08 | 27.64±0.27 | 27.73±0.68 | 933.43±2.57 |

### ag-news

The test set consisted of 10692 examples featuring 4 classes.

| Model          | Embedding model   | Dimensionality | Test mean ± CI | Dev mean ± CI | Train mean ± CI | Seconds per model ± CI |
| -------------- | ----------------- | -------------- | -------------- | ------------- | --------------- | ---------------------- |
| Classical FFNN | ember-v1          | full | 88.53±0.04 | 88.1±0.17  | 88.56±0.57 | 13.62±0.97   | 0.885253773 |
| Classical FFNN | fasttext          | full | 84.5±0.02  | 83.75±0.34 | 83.66±0.7  | 29.85±0.48   | 0.845021204 |
| Classical FFNN | bert-base-uncased | full | 81.97±0.04 | 81.58±0.14 | 82.1±0.39  | 28.84±2.38   | 0.819672028 |
| Classical FFNN | ember-v1          | 8    | 84.12±0.05 | 83.39±0.83 | 83.12±1.24 | 25.75±0.21   | 0.841177199 |
| QNLP Beta 2/3   | ember-v1          | 8    | 80.41±0.03 | 79.28±1.52 | 78.02±3.61 | 1122.52±8.26 | 0.804133932 |
| Classical FFNN | fasttext          | 8    | 75.02±0.02 | 73.55±0.71 | 73.58±0.99 | 25.85±0.24   | 0.750193284 |
| QNLP Beta 2/3   | fasttext          | 8    | 72.7±0.09  | 68.08±3.45 | 66.78±4.89 | 1101.27±5.77 | 0.727010849 |
| Classical FFNN | bert-base-uncased | 8    | 72.23±0.05 | 71.3±0.41  | 71.34±0.64 | 25.7±0.26    | 0.722325099 |
| QNLP Beta 2/3   | bert-base-uncased | 8    | 67.79±0.02 | 66.64±0.75 | 65.81±2.9  | 1093.24±16.7 | 0.67794301  |
| Classical FFNN | ember-v1          | 5    | 82.66±0.05 | 82.26±0.48 | 82.06±0.93 | 25.93±0.23   | 0.82660868  |
| QNLP Beta 2/3   | ember-v1          | 5    | 80.3±0.03  | 79.34±1    | 78.28±3.34 | 1102.83±8.08 | 0.802986657 |
| Classical FFNN | fasttext          | 5    | 64.4±0.06  | 63.62±0.59 | 63.42±0.74 | 25.47±0.22   | 0.643961219 |
| Classical FFNN | bert-base-uncased | 5    | 62.02±0.06 | 60.67±0.59 | 60.34±0.8  | 25.67±0.39   | 0.620233202 |
| QNLP Beta 2/3   | fasttext          | 5    | 58.97±0.27 | 56.6±3.19  | 55.37±4.19 | 1105.26±6.15 | 0.589733757 |
| QNLP Beta 2/3   | bert-base-uncased | 5    | 58.22±0.07 | 56.83±0.65 | 55.7±2.27  | 1095.56±4.69 | 0.582223469 |
| Classical FFNN | ember-v1          | 3    | 81.36±0.05 | 81.12±0.67 | 80.92±0.99 | 25.81±0.22   | 0.813611418 |
| QNLP Beta 2/3   | ember-v1          | 3    | 79.59±0.05 | 79.21±0.51 | 78.16±3.03 | 1208.52±7.8  | 0.795891009 |
| Classical FFNN | fasttext          | 3    | 53.9±0.1   | 53.43±0.29 | 53.33±0.42 | 26.11±0.24   | 0.538954357 |
| QNLP Beta 2/3   | fasttext          | 3    | 49.65±0.22 | 47.38±1.49 | 46.62±2.79 | 1182.23±3.1  | 0.496470882 |
| Classical FFNN | bert-base-uncased | 3    | 44.62±0.07 | 44.45±0.36 | 44.27±0.49 | 25.44±0.6    | 0.446237061 |
| QNLP Beta 2/3   | bert-base-uncased | 3    | 43.48±0.06 | 42.58±0.24 | 42.39±1.34 | 994.81±60.4  | 0.434829779 |

### amazon-reviews

The test set consisted of 39718 examples featuring 2 classes.

| Model          | Embedding model   | Dimensionality | Test mean ± CI | Dev mean ± CI | Train mean ± CI | Seconds per model ± CI |
| -------------- | ----------------- | -------------- | -------------- | ------------- | --------------- | ---------------------- |
| Classical FFNN | ember-v1          | full | 87.56±0.03 | 87.55±0.09 | 87.5±0.26  | 21.78±1.19     | 0.875551207 |
| Classical FFNN | bert-base-uncased | full | 81.99±0.05 | 82.27±0.09 | 82.71±0.21 | 41.82±5.51     | 0.819895542 |
| Classical FFNN | fasttext          | full | 80.6±0.04  | 80.33±0.09 | 80.36±0.15 | 84.23±2.1      | 0.805976301 |
| Classical FFNN | ember-v1          | 8    | 86.77±0.01 | 86.97±0.02 | 86.49±0.16 | 66.56±3.04     | 0.867651105 |
| QNLP Beta 2/3   | ember-v1          | 8    | 86.27±0.01 | 86.4±0.06  | 85.99±0.12 | 4106.68±20.83  | 0.862725548 |
| Classical FFNN | bert-base-uncased | 8    | 73.88±0.03 | 74.09±0.05 | 73.71±0.28 | 63.64±4.04     | 0.73875321  |
| QNLP Beta 2/3   | bert-base-uncased | 8    | 72.88±0.02 | 72.52±0.31 | 72.15±0.65 | 2357.72±24.04  | 0.728825721 |
| Classical FFNN | fasttext          | 8    | 71.35±0.03 | 71.28±0.09 | 71.02±0.18 | 67.96±0.18     | 0.713464932 |
| QNLP Beta 2/3   | fasttext          | 8    | 70.55±0.04 | 70.16±0.33 | 69.52±1.02 | 2557.94±154.83 | 0.705510516 |
| Classical FFNN | ember-v1          | 5    | 86.68±0.01 | 86.9±0.02  | 86.41±0.06 | 64.66±3.7      | 0.86678584  |
| QNLP Beta 2/3   | ember-v1          | 5    | 86.3±0.02  | 86.43±0.07 | 85.99±0.12 | 4063.31±19.98  | 0.863024321 |
| Classical FFNN | fasttext          | 5    | 68.28±0.02 | 68.13±0.06 | 67.87±0.13 | 67.23±1.97     | 0.68284741  |
| QNLP Beta 2/3   | fasttext          | 5    | 68.05±0.05 | 67.55±0.26 | 66.94±0.82 | 2567.51±174.82 | 0.680547862 |
| Classical FFNN | bert-base-uncased | 5    | 62.33±0.04 | 62.41±0.05 | 62.13±0.09 | 65.64±2.55     | 0.62332024  |
| QNLP Beta 2/3   | bert-base-uncased | 5    | 61.58±0.04 | 61.09±0.23 | 60.92±0.3  | 2352.16±4.92   | 0.615848398 |
| Classical FFNN | ember-v1          | 3    | 86.65±0.01 | 86.76±0.02 | 86.28±0.12 | 54.39±6.21     | 0.866485393 |
| QNLP Beta 2/3   | ember-v1          | 3    | 86.29±0.01 | 86.45±0.08 | 86.03±0.08 | 4318.78±121.79 | 0.862907666 |
| Classical FFNN | fasttext          | 3    | 58.08±0.02 | 58.28±0.01 | 58.25±0.02 | 47.88±6.86     | 0.58076187  |
| QNLP Beta 2/3   | fasttext          | 3    | 57.83±0.03 | 58.06±0.09 | 57.99±0.1  | 2684.3±43.47   | 0.578297833 |
| Classical FFNN | bert-base-uncased | 3    | 57.74±0.06 | 58.04±0.05 | 57.41±0.04 | 33.26±5.22     | 0.577430884 |
| QNLP Beta 2/3   | bert-base-uncased | 3    | 57.32±0.04 | 57.05±0.11 | 56.48±0.18 | 2394.51±121.78 | 0.573151552 |

*This project has received funding from the European Union’s Horizon
2020 research and innovation programme under grant agreement No 951821.*

