#!/bin/bash

echo 'This script tokenizes a dataset.'

infile='-'
delimiter=','
classfield='-'
txtfield='-'

while getopts i:o:d:c:t: flag
do
    case "${flag}" in
        i) infile=${OPTARG};;
        o) outfile=${OPTARG};;
        d) delimiter=${OPTARG};;
        c) classfield=${OPTARG};;
        t) txtfield=${OPTARG};;
    esac
done


if [[ "$infile" == "-" ]]
then
__usage="
Usage: $(basename $0) [OPTIONS]

Options:
  -i <dataset>            Dataset file (with path)
  -o <result>             Out file (with path)
  -d <delimiter>          Field delimiter symbol
  -c <class fiels>        Name of the class field (only if the first line in the file contains field names)
  -t <text field>         Name of the text field (only if the first line in the __usagefile contains field names)
"
	echo "$__usage"
else
	if [[ "$classfield" == "-" ]]
	then
		python ./data/data_processing/filter-with-spacy-preprocessing.py -i "${infile}" -o "${outfile}" -d "${delimiter}"
	else
		python ./data/data_processing/filter-with-spacy-preprocessing.py -i "${infile}" -o "${outfile}" -d "${delimiter}" -c "${classfield}" -t "${txtfield}"
	fi
fi
