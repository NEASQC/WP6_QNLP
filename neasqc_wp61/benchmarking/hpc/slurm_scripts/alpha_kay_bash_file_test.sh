#!/bin/sh

# Slurm flags
#SBATCH -p DevQ
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --job-name=alpha_kay_bash_file_test
   
# Charge job to my project 
#SBATCH -A iccom018c

# Write stdout+stderr to file
#SBATCH -o slurm_output/alpha_kay_bash_file_test_output.txt

# Mail me on job start & end
#SBATCH --mail-user=yanis.lalou@ichec.ie
#SBATCH --mail-type=BEGIN,END

cd $SLURM_SUBMIT_DIR

cd WP6_QNLP/neasqc_wp61

module load conda

source activate /ichec/work/iccom018c/.conda/qnlp

# -s : Seed for the initial parameters
# -i : Number of iterations of the optimiser
# -r : Number of runs
# -tr : Directory of the train dataset
# -te : Directory of the test datset
# -o : Output directory with the predictions
# -nq : Number of qubits in our circuit
# -qd : Initial spread of the parameters
# -b : Batch size
# -lr : Learning rate
# -wd : Weight decay
# -slr : Step size for the learning rate scheduler
# -g : Gamma for the learning rate scheduler


echo "`date +%T`"

bash 6_Classify_With_Quantum_Model.sh -m alpha_pennylane -t ./data/toy_dataset/toy_dataset_bert_sentence_embedding_train.csv -v ./data/toy_dataset/toy_dataset_bert_sentence_embedding_test.csv -s 200 -r 5 -i 100 -o ./benchmarking/results/raw/

echo "`date +%T`"