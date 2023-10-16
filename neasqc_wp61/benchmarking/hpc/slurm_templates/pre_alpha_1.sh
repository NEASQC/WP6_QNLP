#!/bin/sh

# Slurm flags
#SBATCH -p ProdQ
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --job-name=pre_alpha_1_{{ s }}_{{ p }}_{{ i }}_{{ r }}
   
# Charge job to my project 
#SBATCH -A iccom018c

# Write stdout+stderr to file
#SBATCH -o ./benchmarking/hpc/slurm_output/pre_alpha_1_{{ s }}_{{ p }}_{{ i }}_{{ r }}_output.txt

# Mail me on job start & end
#SBATCH --mail-user=pablo.suarez@ichec.ie
#SBATCH --mail-type=BEGIN,END

cd /ichec/work/iccom018c

cd WP6_QNLP/neasqc_wp61

module load conda

source activate /ichec/work/iccom018c/.conda/pre_alpha

# -t : path of the training dataset
# -v : path of the testing dataset
# -s : seed for the optimisers
# -r : number of runs of the model
# -i : number of iterations of the optmiser
# -p : choice of optimiser
# -o : path of output directory

echo "`date +%T`"

bash 6_Classify_With_Quantum_Model.sh -m pre_alpha_1 -t ./data/datasets/reduced_amazonreview_pre_alpha_train.tsv -v ./data/datasets/reduced_amazonreview_pre_alpha_test.tsv -s {{ s }} -r {{ r }} -i {{ i }} -p {{ p }} -o ./benchmarking/results/raw/

echo "`date +%T`"
