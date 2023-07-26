#!/bin/sh

# Slurm flags
#SBATCH -p GpuQ
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --job-name=alpha_pennylane200_5_100_4_0.01_2048_2e-3_0.5_20_0
   
# Charge job to my project 
#SBATCH -A iccom018c

# Write stdout+stderr to file
#SBATCH -o ./benchmarking/hpc/slurm_output/alpha_pennylane200_5_100_4_0.01_2048_2e-3_0.5_20_0.txt

# Mail me on job start & end
#SBATCH --mail-user=yanis.lalou@ichec.ie
#SBATCH --mail-type=BEGIN,END

cd /ichec/work/iccom018c

cd WP6_QNLP/neasqc_wp61

module load conda

source activate /ichec/work/iccom018c/.conda/qnlp

module load cuda/11.4

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

bash 6_Classify_With_Quantum_Model.sh -m alpha_pennylane -t ./data/toy_dataset/toy_dataset_bert_sentence_embedding_train.csv -v ./data/toy_dataset/toy_dataset_bert_sentence_embedding_test.csv -s 200 -r 5 -i 100 -u 4 -d 0.01 -b 2048 -l 2e-3 -w 0.5 -z 20 -g 0 -o ./benchmarking/results/raw/
echo "`date +%T`"