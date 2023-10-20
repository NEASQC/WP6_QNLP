#!/bin/bash

echo 'This script classifies examples using beta neighbours model.'


while getopts "t:v:k:o:" flag
do
    case "${flag}" in
        t) train=${OPTARG};;
        v) test=${OPTARG};;
        k) IFS=' ' read -r -a k <<< "${OPTARG}";;
        o) output=${OPTARG};;
    esac
done

echo "train: $train";
echo "test: $test";
echo "K values for KNN algorithm: ${k[*]}";
echo "output: $output";

echo "running beta_1"

python3.10 ./data/data_processing/use_beta_1.py -tr ${train} -te ${test} -k ${k[*]} -o ${output}