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
output_dir="./benchmarking/results/raw/classical"
sanitize_model_name() {
    echo "$1" | tr '/' '_'
}

for dim in ${dims[@]}; do
    for dataset_base_name in ${datasets[@]}; do
        for model in "${!models[@]}"; do
            sanitized_model=$(sanitize_model_name "${model}")
            train_file="${input_dir}/${dataset_base_name}_train_${sanitized_model}${dim}.json"
            dev_file="${input_dir}/${dataset_base_name}_dev_${sanitized_model}${dim}.json"
            test_file="${input_dir}/${dataset_base_name}_test_${sanitized_model}${dim}.json"
            LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:/home/TILDE.LV/marcis.pinnis/anaconda3/envs/neasqc2/lib/python3.10/site-packages/tensorrt_libs/ python ./data/data_processing/train_classifier.py -it "${train_file}" -id "${dev_file}" -ie "${test_file}" -f "class" -e "sentence" -m "${output_dir}/ffnn_${dataset_base_name}_${sanitized_model}${dim}"  -g "0"
        done
    done
done

for dataset_base_name in ${datasets[@]}; do
    for model in "${!models[@]}"; do
        sanitized_model=$(sanitize_model_name "${model}")
        train_file="${input_dir}/${dataset_base_name}_train_${sanitized_model}.json"
        dev_file="${input_dir}/${dataset_base_name}_dev_${sanitized_model}.json"
        test_file="${input_dir}/${dataset_base_name}_test_${sanitized_model}.json"
        LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:/home/TILDE.LV/marcis.pinnis/anaconda3/envs/neasqc2/lib/python3.10/site-packages/tensorrt_libs/ python ./data/data_processing/train_classifier.py -it "${train_file}" -id "${dev_file}" -ie "${test_file}" -f "class" -e "sentence" -m "${output_dir}/ffnn_${dataset_base_name}_${sanitized_model}_full"  -g "0"
    done
done
