#!/bin/bash

echo 'This script classifies examples using quantum classifier model.'


while getopts t:v:s:m:e:r:i:p:o:a:q:n:x:l:k flag
do
    case "${flag}" in
        t) train=${OPTARG};;
        v) test=${OPTARG};;
        s) seed=${OPTARG};;
        m) model=${OPTARG};;
        r) runs=${OPTARG};;
        i) iterations=${OPTARG};;
        p) optimiser=${OPTARG};;
        o) outfile=${OPTARG};;
        a) ansatz=${OPTARG};;
        q) qn=${OPTARG};;
        n) nl=${OPTARG};;
        x) np=${OPTARG};;
        l) labels=${OPTARG};;
        k) k=${OPTARG};;
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
echo "ansatz: $ansatz";
echo "Number of qubits per noun: $qn";
echo "number of circuit layers: $nl";
echo ":number of single qubit parameters $np";
echo "labels: $labels";
echo "k: $k";


if [[ "${model}" == "pre_alpha" ]]
then
echo "running pre_alpha"
python3.10 ./data/data_processing/use_pre_alpha.py -s ${seed} -op ${optimiser} -i ${iterations} -r ${runs} -tr ${train} -te ${test} -o ${outfile}
elif [[ "${model}" == "pre_alpha_lambeq" ]]
then
echo "running pre_alpha_lambeq"
python3.10 ./data/data_processing/use_pre_alpha_lambeq.py -s ${seed} -op ${optimiser} -i ${iterations} -r ${runs} -tr ${train} -te ${test} -o ${outfile} -an ${ansatz} -qn ${qn} -nl ${nl} -np ${np}
elif [[ "${model}" == "beta_quantum" ]]
then
echo "running quantum beta neighbours"
python3.10 ./data/data_processing/use_beta_neighbours.py -l ${labels} -tr ${train} -te ${test} -k ${k} -o ${outfile}
else
echo "no model ran";
fi