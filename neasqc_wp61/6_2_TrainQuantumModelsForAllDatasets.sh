#!/bin/bash

declare -A models
models=(
    ["llmrails/ember-v1"]="transformer"
    ["google-bert/bert-base-uncased"]="bert"
    ["fasttext"]="fasttext"
)

datasets=("labelled_newscatcher_dataset_tok_balanced" "amazon-fine-food-reviews_tok_balanced" "ag_news_tok_balanced" "food-com-recipes-user-interactions_tok_balanced" "huffpost-news_tok_balanced" "amazon-reviews_tok_balanced")
dims=("_3" "_5" "_8")
input_dir="./data/datasets"
out_dir="benchmarking/results/raw/quantum"

sanitize_model_name() {
    echo "$1" | tr '/' '_'
}

for dim in ${dims[@]}; do
    for dataset_base_name in ${datasets[@]}; do
        for model in "${!models[@]}"; do
            sanitized_model=$(sanitize_model_name "${model}")
            train_file="${input_dir}/${dataset_base_name}_train_${sanitized_model}${dims}.json"
            dev_file="${input_dir}/${dataset_base_name}_dev_${sanitized_model}${dims}.json"
            test_file="${input_dir}/${dataset_base_name}_test_${sanitized_model}${dims}.json"
            model_dir="${out_dir}/${dataset_base_name}_test_${sanitized_model}${dims}_beta_2_3"
            LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:/home/TILDE.LV/marcis.pinnis/anaconda3/envs/neasqc2/lib/python3.10/site-packages/tensorrt_libs/ python3.10 ./data/data_processing/use_beta_2_3_tests.py -op Adam -s 42 -i 10 -r 30 -dat ${train_file} -te ${test_file} -va ${dev_file} -o ${model_dir} -nq 8 -qd "0.01" -b 320 -lr "0.002" -wd 0 -slr 150 -g 1
        done
    done
done
