#!/bin/sh

# Slurm flags
#SBATCH -p DevQ
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --job-name=beta_neighbours_{{ k }}
   
# Charge job to my project 
#SBATCH -A iccom018c

# Write stdout+stderr to file
#SBATCH -o ./benchmarking/hpc/slurm_output/beta_neighbours_{{ k }}.txt

# Mail me on job start & end
#SBATCH --mail-user=pablo.lauret@ichec.ie
#SBATCH --mail-type=BEGIN,END

cd $SLURM_SUBMIT_DIR

cd WP6_QNLP/neasqc_wp61

module load conda

source activate /ichec/work/iccom018c/.conda/qnlp

# -l : path of dataset containing labels
# -t : path of the training dataset
# -v : path of the testing dataset
# -k : number of nearest neighbours
# -o : path of output directory

echo "`date +%T`"

6_Beta_Neighbours.sh -l INSERT_PATH_HERE -t ./data/datasets/reduced_amazonreview_pre_alpha_train.tsv -v ./data/datasets/reduced_amazonreview_pre_alpha_test.tsv -k {{ k }} -o ./benchmarking/results/raw/

echo "`date +%T`"
