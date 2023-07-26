#!/bin/bash

echo 'This script classifies examples using quantum classifier model.'


while getopts t:v:s:m:e:r:i:p:o:a:q:n:x:u:d:b:l:w:z:g:y:c:e flag
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

        u) nq=${OPTARG};;
        d) qd=${OPTARG};;
        b) b=${OPTARG};;
        l) lr=${OPTARG};;
        w) wd=${OPTARG};;
        z) slr=${OPTARG};;
        g) g=${OPTARG};;

        y) version=${OPTARG};;
        c) pca=${OPTARG};;
        e) qs=${OPTARG};;
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

echo "Number of qubits in our circuit: $nq";
echo "Initial spread of the parameters: $qd";
echo "Batch size: $b";
echo "Learning rate: $lr";
echo "Weight decay: $wd";
echo "Step size for the learning rate scheduler: $slr";
echo "Gamma for the learning rate scheduler: $g";

echo "Version between alpha_pennylane_lambeq and alpha_pennylane_lambeq_original: $version";
echo "Reduced dimension for the word embeddings: $pca";
echo "Number of qubits per SENTENCE type: $qs";



if [[ "${model}" == "pre_alpha" ]]
then
echo "running pre_alpha"
python3.10 ./data/data_processing/use_pre_alpha.py -s ${seed} -op ${optimiser} -i ${iterations} -r ${runs} -tr ${train} -te ${test} -o ${outfile}
elif [[ "${model}" == "pre_alpha_lambeq" ]]
then
echo "running pre_alpha_lambeq"
python3.10 ./data/data_processing/use_pre_alpha_lambeq.py -s ${seed} -op ${optimiser} -i ${iterations} -r ${runs} -tr ${train} -te ${test} -o ${outfile} -an ${ansatz} -qn ${qn} -nl ${nl} -np ${np}
elif [[ "${model}" == "alpha_pennylane" ]]
then
echo "running alpha_pennylane"
python3.10 ./data/data_processing/use_alpha_pennylane.py -s ${seed} -i ${iterations} -r ${runs} -tr ${train} -te ${test} -o ${outfile} -nq ${nq} -qd ${qd} -b ${b} -lr ${lr} -wd ${wd} -slr ${slr} -g ${g}
elif [[ "${model}" == "alpha_lambeq" ]]
then
echo "running alpha_lambeq"
python3.10 ./data/data_processing/use_alpha_lambeq.py -s ${seed} -i ${iterations} -r ${runs} -v ${version} -pca ${pca} -tr ${train} -te ${test} -o ${outfile} -an ${ansatz} -qn ${qn} -qs ${qs} -nl ${nl} -np ${np} -b ${b} -lr ${lr} -wd ${wd} -slr ${slr} -g ${g}
elif [[ "${model}" == "alpha_pennylane_lambeq" ]]
then
else
echo "no model ran";
fi