#!/bin/sh

# Slurm flags
#SBATCH -p ProdQ
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --job-name=beta_1_{{ k }}
   
# Charge job to my project 
#SBATCH -A iccom018c

# Write stdout+stderr to file
#SBATCH -o ./benchmarking/hpc/slurm_output/beta_1_{{ k }}.txt

# Mail me on job start & end
#SBATCH --mail-user=pablo.suarez@ichec.ie
#SBATCH --mail-type=BEGIN,END

cd /ichec/work/iccom018c

cd WP6_QNLP/neasqc_wp61

module load conda

source activate /ichec/work/iccom018c/.conda/beta

# -t : path of the training dataset
# -v : path of the testing dataset
# -d : dimension of PCA-reduced BERT embeddings
# -k : number of nearest neighbours
# -o : path of output directory

echo "`date +%T`"

bash 6_Beta.sh -t ./data/datasets/reduced_amazonreview_train_sentence.csv -v ./data/datasets/reduced_amazonreview_test_sentence.csv -d {d} -k {k} -o ./benchmarking/results/raw/

echo "`date +%T`"
