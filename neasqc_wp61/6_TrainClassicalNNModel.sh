#!/bin/bash

echo 'This script trains neural network classifier model.'

tfile='-'
dfile='-'
field='class'
etype='-'
modeldir='-'
gpu='-1'
 
while getopts t:d:f:e:m:g:r:h: flag
do
    case "${flag}" in
        t) tfile=${OPTARG};;
        d) dfile=${OPTARG};;
        h) efile=${OPTARG};;
        f) field=${OPTARG};;
        e) etype=${OPTARG};;
        m) modeldir=${OPTARG};;
        g) gpu=${OPTARG};;
        r) runs=${OPTARG};;
    esac
done

if [[ "$tfile" == "-" ]] || [[ "$dfile" == "-" ]] || [[ "$modeldir" == "-" ]]
then
__usage="
Usage: $(basename $0) [OPTIONS]

Options:
  -t <train data file> Json data file for classifier training (with embeddings) or tsv file (if not using pre-trained embeddings, acquired using script 3_SplitTrainTestDev.sh)
  -d <dev data file>   Json data file for classifier validation (with embeddings) or tsv file (if not using pre-trained embeddings, acquired using script 3_SplitTrainTestDev.sh)
  -f <field>           Classify by field
  -e <embedding type>  Embedding type: 'sentence', 'word', or '-' (if not using pre-trained embeddings)
  -m <model directory> Directory where to save trained model
  -g <use gpu>		   Number of GPU to use (from 0 to available GPUs), -1 if use CPU (default is -1)
"
	echo "$__usage"
else
	python ./data/data_processing/train_classifier.py -it "${tfile}" -id "${dfile}" -ie "${efile}" -f "${field}" -e "${etype}" -m "${modeldir}"  -g "${gpu}" -r "${runs}"
fi
