#!/bin/sh

# Slurm flags
#SBATCH -p GpuQ
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --job-name=alpha_lambeq200_5_100_alpha_pennylane_lambeq_22_IQP_1_1_1_2_2048_2e-3_0.5_20_0
   
# Charge job to my project 
#SBATCH -A iccom018c

# Write stdout+stderr to file
#SBATCH -o ./benchmarking/hpc/slurm_output/alpha_lambeq200_5_100_alpha_pennylane_lambeq_22_IQP_1_1_1_2_2048_2e-3_0.5_20_0.txt

# Mail me on job start & end
#SBATCH --mail-user=yanis.lalou@ichec.ie
#SBATCH --mail-type=BEGIN,END

cd /ichec/work/iccom018c

cd WP6_QNLP/neasqc_wp61

module load conda

source activate /ichec/work/iccom018c/.conda/qnlp

module load cuda/11.4


# -s : Seed for the initial parameters
# -r : Number of runs
# -i : Number of iterations of the optimiser
# -v : Choose between alpha_pennylane_lambeq and alpha_pennylane_lambeq_original
# -pca : Choose the reduced dimension for the word embeddings
# -tr : Directory of the train dataset
# -te : Directory of the test datset
# -o : Output directory with the predictions
# -an : Ansatz to be used in quantum circuits
# -qn : Number of qubits per NOUN type
# -qs : Number of qubits per SENTENCE type
# -nl : Number of layers for the circuits
# -np : Number of parameters per qubit
# -b : Batch size
# -lr : Learning rate
# -wd : Weight decay
# -slr : Step size for the learning rate scheduler
# -g : Gamma for the learning rate scheduler

echo "`date +%T`"

bash 6_Classify_With_Quantum_Model.sh -m alpha_lambeq -t ./data/toy_dataset/toy_dataset_bert_sentence_embedding_train.csv.csv -v ./data/toy_dataset/toy_dataset_bert_sentence_embedding_test.csv -s 200 -r 5 -i 100 -b 2048 -l 2e-3 -w 0.5 -z 20 -g 0 -y alpha_pennylane_lambeq -c 22 -e 1 -a IQP -q 1 -n 1 -x 2 -o ./benchmarking/results/raw/
echo "`date +%T`"