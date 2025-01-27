# Vectorizer

Services used for vectorizing using pretrained word embeddings.

The aim is to have vectorizing service detached from the rest of the library so that different vectorizing methods can easily be tested using the same interface.

Currently, vectorizing with `BERT`, `sentence transformer` and `fastText` models are implemented.

### Setup
Some Python libraries must be installed for vectorization.

- Python fastText library can be installed with command `pip install fasttext-wheel`

- Python sentence transformer library can be installed with command `pip install -U sentence-transformers`

- Python BERT transformer library can be installed with command `pip install -U transformers`

Use of vectorizing services using Python class *Embeddings* is demonstrated in Jupyter notebook *../../doc/tutorials/Prepare_datasets_4classifier.ipynb*

BERT models can also be employed as the vectorizing services. They are built as a `Docker` containers and instructions for building the images are contained in the `Dockerfile`.

- Download and start *Docker Desktop*

- Download uncased BERT model from https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

- Download cased BERT model from https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip

- From the directory containing docker file (*BertVectorizer*) build docker image with command `docker build -t bertvec .`

- Start docker and bind mount directory where downoaded BERT models reside; as environment variables specify the port for the Web Service and the model's file (in example Bert models reside in the directory local E:\BERTModels that is binded to the container's directory /app/data/BERTModels):

`docker run -it -p 12345:12345 --name bv1 --rm  -v E:\BERTModels:/app/data/BERTModels --env port="12345" --env model_file=/app/data/BERTModels/cased_L-12_H-768_A-12 bertvec`

`docker run -it -p 22222:22222 --name bv2 --rm  -v E:\BERTModels:/app/data/BERTModels --env port="22222" --env model_file=/app/data/BERTModels/uncased_L-12_H-768_A-12 bertvec`

### Usage example
Script *GenerateEmbeddings.py* allows to use models of all types. You have to specify input and output file, model type and model name.

`python ./GenerateEmbeddings.py -i ../../datasets/reviews_traintest.json -o ../../datasets/reviews_FASTTEXT.json -t "fasttext" -m "cc.en.300.bin"`

`python ./GenerateEmbeddings.py -i ../../datasets/reviews_traintest.json -o ../../datasets/reviews_BERT_UNCASED.json -t "bert" -m "bert-base-uncased"`

`python ./GenerateEmbeddings.py -i ../../datasets/reviews_traintest.json -o ../../datasets/reviews_all-distilroberta.json -t "transformer" -m "all-distilroberta-v1"`

