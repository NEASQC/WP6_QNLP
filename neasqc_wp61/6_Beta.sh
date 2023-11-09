#!/bin/bash

echo 'This script classifies examples using beta_1 model.'


while getopts "t:v:d:k:o:" flag
do
    case "${flag}" in
        t) train=${OPTARG};;
        v) test=${OPTARG};;
        d) dimension=${OPTARG};;
        k) IFS=' ' read -r -a k <<< "${OPTARG}";;
        o) output=${OPTARG};;
    esac
done

echo "train: $train";
echo "test: $test";
echo "K values for KNN algorithm: ${k[*]}";
echo "output: $output";

echo "running beta_1"
python3.10 ./data/data_processing/use_beta_1.py -tr ${train} -te ${test} -d ${dimension} -k ${k[*]} -o ${output}s