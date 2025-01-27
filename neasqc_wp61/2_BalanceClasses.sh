#!/bin/bash

echo 'This script sub-selects data such that classes would be balanced.'

infile='-'
filterfile='-'
classes='-'

while getopts i:c: flag
do
    case "${flag}" in
        i) infile=${OPTARG};;
        c) classes=${OPTARG};;
    esac
done

replace="_balanced.tsv"
outfile=${infile//.tsv/$replace}
	
if [[ "$infile" == "-" ]]
then
__usage="
Usage: $(basename $0) [OPTIONS]

Options:
  -i <input file>               2-column file containing class and text
  -c <classes to ignore>		Optionally classes to ignore in dataset separated by ',' 
"
	echo "$__usage"
else
	if [[ "$classes" == "-" ]]
	then
		python ./data/data_processing/balance_classes.py -i "${infile}" -o "${outfile}"
	else
		python ./data/data_processing/balance_classes.py -i "${infile}" -o "${outfile}" -c 	"${classes}"
	fi
	wc -l $outfile | awk '{ print $1, " lines added to the balanced data file."}'
fi
