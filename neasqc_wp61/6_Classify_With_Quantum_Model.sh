#!/bin/bash

echo 'This script classifies examples using quantum classifier model.'


while getopts t:v:s:m:e:r:i:o: flag
do
    case "${flag}" in
        t) train=${OPTARG};;
        v) test=${OPTARG};;
        s) seed=${OPTARG};;
        m) model=${OPTARG};;
        e) epochs=${OPTARG};;
        r) runs=${OPTARG};;
        i) iterations=${OPTARG};;
        p) optimiser=${OPTARG};;
        o) outfile=${OPTARG};;
    esac
done
echo "train: $train";
echo "test: $test";
echo "seed: $seed";
echo "model: $model";
echo "epochs: $epochs";
echo "runs: $runs";
echo "optimiser: $optimiser";
echo "iterations: $iterations";
echo "outfile: $outfile";

python3.10 ./data/data_processing/use_pre_alpha.py -s ${seed} -op ${optimiser} -i ${iterations} -r ${runs} -tr ${train} -te ${test} -o ${outfile}
