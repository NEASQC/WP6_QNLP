#!/bin/bash

echo 'This script classifies examples using quantum classifier model.'


while getopts t:v:s:m:e:r:i:p:o:a:q:n:x:u:d:b:l:w:z:g: flag
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

# -s : 
# -i : 
# -r : 
# -tr : -t
# -te : -v
# -o : 
# -nq : Number of qubits in our circuit
# -qd : Initial spread of the parameters
# -b : Batch size
# -lr : Learning rate
# -wd : Weight decay
# -slr : Step size for the learning rate scheduler
# -g : Gamma for the learning rate scheduler


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
python3.10 ./data/data_processing/use_alpha_lambeq.py -s ${seed} -op ${optimiser} -i ${iterations} -r ${runs} -tr ${train} -te ${test} -o ${outfile} -an ${ansatz} -qn ${qn} -nl ${nl} -np ${np}
else
echo "no model ran";
fi
