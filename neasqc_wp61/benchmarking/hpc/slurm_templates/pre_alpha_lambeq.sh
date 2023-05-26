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

# -tr : path of the training dataset
# -te : path of the testing dataset
# -s : seed for the optimisers
# -r : number of runs of the model
# -i : number of iterations of the optmiser
# -p : choice of optimiser
# -an : choice of ansatz
# -qn : number of qubits per noun type
# -nl : number of layers in the quantum circuit
# -np : number of parameters per qubit
# -qs: number of qubits per sentence, currently always set to 1
# -o : path of output directory

echo "`date +%T`"

#Here call step 6 once it has been edited to include pre-alpha lambeq. Use Juja formatting.

echo "`date +%T`"
