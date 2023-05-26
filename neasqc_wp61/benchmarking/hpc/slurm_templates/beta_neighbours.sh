#!/bin/sh

# Slurm flags
#SBATCH -p DevQ
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --job-name=test_job
   
# Charge job to my project 
#SBATCH -A iccom018c

# Write stdout+stderr to file
#SBATCH -o tes_job_output.txt

# Mail me on job start & end
#SBATCH --mail-user=pablo.lauret@ichec.ie
#SBATCH --mail-type=BEGIN,END

cd $SLURM_SUBMIT_DIR

cd WP6_QNLP/neasqc_wp61

module load conda

source activate /ichec/work/iccom018c/.conda/qnlp

# -l : path of dataset containing labels
# -tr : path of the training dataset
# -te : path of the testing dataset
# -k : number of nearest neighbours
# -r : number of runs of the model
# -o : path of output directory

echo "`date +%T`"

#Call step 6 once it has been edited using Jinja formatting

echo "`date +%T`"
