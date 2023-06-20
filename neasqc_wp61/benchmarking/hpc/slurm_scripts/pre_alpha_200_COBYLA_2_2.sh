#!/bin/sh

# Slurm flags
#SBATCH -p DevQ
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --job-name=pre_alpha_200_COBYLA_2_2
   
# Charge job to my project 
#SBATCH -A iccom018c

# Write stdout+stderr to file
#SBATCH -o slurm_output/pre_alpha_200_COBYLA_2_2_output.txt

# Mail me on job start & end
#SBATCH --mail-user=pablo.lauret@ichec.ie
#SBATCH --mail-type=BEGIN,END

cd $SLURM_SUBMIT_DIR

cd WP6_QNLP/neasqc_wp61

module load conda

source activate /ichec/work/iccom018c/.conda/qnlp

# -t : path of the training dataset
# -v : path of the testing dataset
# -s : seed for the optimisers
# -r : number of runs of the model
# -i : number of iterations of the optmiser
# -p : choice of optimiser
# -o : path of output directory

echo "`date +%T`"

bash 6_Classify_With_Quantum_Model.sh -t ./data/datasets/reduced_amazonreview_pre_alpha_train.tsv -v ./data/datasets/reduced_amazonreview_pre_alpha_test.tsv -s 200 -r 2 -i 2 -p COBYLA -o ./benchmarking/results/raw/

echo "`date +%T`"