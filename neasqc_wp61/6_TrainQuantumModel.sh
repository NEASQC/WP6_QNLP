#!/bin/bash

echo 'This script classifies examples using quantum classifier model.'


while getopts N:x:t:v:j:s:r:i:p:o:b:l:w:z:g: flag
do
    case "${flag}" in

        t) train=${OPTARG};;
        v) test=${OPTARG};;
        j) validation=${OPTARG};;
        o) outfile=${OPTARG};;
        N) nq=${OPTARG};;
        s) qd=${OPTARG};;
        i) iterations=${OPTARG};;
        b) batch=${OPTARG};;
        w) wd=${OPTARG};;
        x) seed=${OPTARG};;
        p) optimiser=${OPTARG};;
        l) lr=${OPTARG};;
        z) slr=${OPTARG};;
        g) g=${OPTARG};;
        r) runs=${OPTARG};;
        *) 
           echo "Invalid option: -$OPTARG" 
           exit 1 ;;

    esac
done

echo "Optimiser: $optimiser";

echo "Number of qubits: $nq";

echo "Training dataset: $train";
echo "Test dataset: $test";
echo "Validation dataset: $validation";
echo "Output file path: $outfile";

echo "Number of runs: $runs";
echo "Number of iterations: $iterations";
echo "Batch size: $batch";

echo "Running BETA3-tests"
python3.10 ./data/data_processing/use_beta_2_3_tests.py -op ${optimiser} -s ${seed} -i ${iterations} -r ${runs} -dat ${train} -te ${test} -va ${validation} -o ${outfile} -nq ${nq} -qd ${qd} -b ${batch} -lr ${lr} -wd ${wd} -slr ${slr} -g ${g}

