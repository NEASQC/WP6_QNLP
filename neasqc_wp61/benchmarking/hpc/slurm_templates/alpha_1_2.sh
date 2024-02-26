#!/bin/sh

# Slurm flags
#SBATCH -p ProdQ
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH --job-name=alpha_1_2_{{ s }}_{{ r }}_{{ i }}_{{ v }}_{{ pca }}_{{ an }}_{{ qn }}_{{ qs }}_{{ nl }}_{{ np }}_{{ sb }}_{{ lr }}_{{ wd }}_{{ slr }}_{{ g }}
   
# Charge job to my project 
#SBATCH -A iccom018c

# Write stdout+stderr to file
#SBATCH -o ./benchmarking/hpc/slurm_output/alpha_1_2_{{ s }}_{{ r }}_{{ i }}_{{ v }}_{{ pca }}_{{ an }}_{{ qn }}_{{ qs }}_{{ nl }}_{{ np }}_{{ sb }}_{{ lr }}_{{ wd }}_{{ slr }}_{{ g }}.txt

# Mail me on job start & end
#SBATCH --mail-user=yanis.lalou@ichec.ie
#SBATCH --mail-type=BEGIN,END

cd /ichec/work/iccom018c

cd WP6_QNLP/neasqc_wp61

module load conda

source activate /ichec/work/iccom018c/.conda/alpha

#Uncomment line below for GPU jobs
#module load cuda/11.4


# -s : Seed for the initial parameters
# -r : Number of runs
# -i : Number of iterations of the optimiser
# -v : Choose between alpha_1 and alpha_2
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

bash 6_Classify_With_Quantum_Model.sh -m alpha_1_2 -t ./data/toy_dataset/toy_dataset_bert_sentence_embedding_train.csv.csv -v ./data/toy_dataset/toy_dataset_bert_sentence_embedding_test.csv -s {{ s }} -r {{ r }} -i {{ i }} -b {{ sb }} -l {{ lr }} -w {{ wd }} -z {{ slr }} -g {{ g }} -y {{ v }} -c {{ pca }} -e {{ qs }} -a {{ an }} -q {{ qn }} -n {{ nl }} -x {{ np }} -o ./benchmarking/results/raw/
echo "`date +%T`"