#!/bin/bash

echo 'This script compares expected results with results acquired using classifier.'

efile='-'
cfile='-'
ofile='-'
 
while getopts e:c:o: flag
do
    case "${flag}" in
        e) efile=${OPTARG};;
        c) cfile=${OPTARG};;
        o) ofile=${OPTARG};;
    esac
done

if [[ "$efile" == "-" ]] || [[ "$cfile" == "-" ]] || [[ "$ofile" == "-" ]]
then
__usage="
Usage: $(basename $0) [OPTIONS]

Options:
  -e <expected results file>     Expected results file containing class in the first column and optionaly other columns
  -c <classifier results file>   Results acquired using classifier.
  -o <accuracy file>             File for calculated test accuracy.
"
	echo "$__usage"
elif [ "$(wc -l < $efile)" -eq "$(wc -l < $cfile)" ]
then
python3 -c '
import sys, csv, numpy
file1, file2, file3 = sys.argv[1:]
with open(file1) as f1:
    expectedClass = [line.rstrip().split()[0] for line in f1]
with open(file2) as f2:
    predictedClass = [line.rstrip().split()[0] for line in f2]
with open(file3, "w", encoding="utf-8") as f3:
	f3.write("Test accuracy: " + str(sum(x == y for x, y in zip(expectedClass, predictedClass))/len(predictedClass)))
' $efile $cfile $ofile
else
	echo 'Different number of lines in files! Can not compare.'
fi
