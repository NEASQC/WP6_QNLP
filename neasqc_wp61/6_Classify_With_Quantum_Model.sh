#!/bin/bash

echo 'This script classifies examples using quantum classifier model.'

model='-'
iterations='-'
seed='-'
epochs='-'
runs='-'
train='-'
test='-1'
outfile='-'
 
while getopts tr:te:s:m:e:r:i:o: flag
do
    case "${flag}" in
        tr) train=${OPTARG};;
        te) test=${OPTARG};;
        s) seed=${OPTARG};;
        m) model=${OPTARG};;
        e) epochs=${OPTARG};;
        r) runs=${OPTARG};;
        i) iterations=${OPTARG};;
        o) outfile=${OPTARG};;
    esac
done

if [[ "$train" == "-" ]] || [[ "$model" == "-" ]] || [[ "$seed" == "-" ]] || [[ "$outfile" == "-" ]]
then
__usage="
Usage: $(basename $0) [OPTIONS]
Options:
  -tr <train file> 	   Json data file for classifier training
  -te <test file> 	   Json data file for classifier testing
  -o <output file>     Result file with predicted classes
  -e <epochs>          Number of epochs of training for alpha
  -m <model type>      Choose \"alpha\" model or \"pre_alpha\" model
  -r <runs>		       Number of training runs for alpha
  -s <seed>		       Seed for random parameters
  -i <iterations>      Number of optimiser iterations for pre_alpha
"
	echo "$__usage"
else
    if [["$model" == "alpha"]]
    then
    python ./data/data_processing/use_alpha.py -s "${seed}" -e "${epochs}" -r "${runs}" -tr "${train}" -te "${test}" -o "${outfile}"
    else
    python ./data/data_processing/use_pre_alpha.py -s "${seed}" -i "${iterations}" -tr "${train}" -te "${test}" -o "${outfile}"
    fi
fi
