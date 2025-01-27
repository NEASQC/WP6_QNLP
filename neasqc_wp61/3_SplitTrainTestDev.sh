#!/bin/bash

echo 'This script splits examples in train/test/dev parts.'

infile='-'
randomsplit='-'
while getopts i:r flag
do
    case "${flag}" in
        i) infile=${OPTARG};;
        r) randomsplit='set';;
    esac
done

if [[ "$infile" == "-" ]]
then
__usage="
Usage: $(basename $0) [OPTIONS]

Options:
  -i <input file>              3-column file containing class, text and syntactical tree of the text
  -r            Split method: random stratified. If parameter omitted then words in test/dev must be also in train.
"
	echo "$__usage"
else
if [[ "$randomsplit" == "set" ]]
then
	python ./data/data_processing/train_test_dev_split.py -i "${infile}" -r
else
	python ./data/data_processing/train_test_dev_split.py -i "${infile}"
fi
fi
