#!/bin/bash

echo 'This script gets embedding vectors for every sample in train and dev datasets using specified embedding model.'

infile='-'
outfile='-'
embmodeltype='-'
embname='-'
column='0'
gpu='-1'
embtype='sentence'
 
while getopts i:o:c:m:t:g:e: flag
do
    case "${flag}" in
        i) infile=${OPTARG};;
        o) outfile=${OPTARG};;
        c) column=${OPTARG};;
        m) embname=${OPTARG};;
        t) embmodeltype=${OPTARG};;
        e) embtype=${OPTARG};;
        g) gpu=${OPTARG};;
    esac
done

if [[ "$infile" == "-" ]] || [[ "$outfile" == "-" ]] || [[ "$embmodeltype" == "-" ]] || [[ "$embname" == "-" ]]
then
__usage="
Usage: $(basename $0) [OPTIONS]

Options:
  -i <input file>      input file with text examples
  -o <output file>     output json file with embeddings
  -c <column>          '2' - if 2-column input file containing class and text columns, '0' - if the whole line is a text example
  -m <embedding name>  Name of the embedding model
  -t <embedding model type>  Type of the embedding model - 'fasttext', 'transformer' or 'bert'
  -e <embedding type>  Embedding type: 'sentence' or 'word'
  -g <use gpu>		   Number of GPU to use (from 0 to available GPUs), -1 if use CPU (dfault is -1)
"
	echo "$__usage"
else
if [[ "$gpu" == "-1" ]]
then
	visibledev=""
else
	visibledev=$gpu
fi
echo $visibledev
	CUDA_VISIBLE_DEVICES=$visibledev python ./data/data_processing/data_vectorisation/Embeddings.py -i "${infile}" -o "${outfile}" -c "${column}" -m "${embname}" -t "${embmodeltype}" -e "${embtype}" -g "${gpu}"
fi
