#!/bin/bash

echo 'This script classifies examples using beta neighbours model.'


while getopts l:t:v:k:r:o: flag
do
    case "${flag}" in
        l) labels=${OPTARG};;
        t) train=${OPTARG};;
        v) test=${OPTARG};;
        k) k=${OPTARG};;
        o) output=${OPTARG};;
    esac
done

echo "labels: $labels";
echo "train: $train";
echo "test: $test";
echo "k: $k";
echo "runs: $runs";
echo "output: $output";




echo "running beta neighbours"
python3.10 ./data/data_processing/use_beta_neighbours.py -l ${labels} -tr ${train} -te ${test} -k ${k} -o ${outfile}