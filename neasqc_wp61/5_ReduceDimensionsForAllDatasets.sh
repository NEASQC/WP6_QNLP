#!/bin/bash

declare -A models
models=(
    ["llmrails/ember-v1"]="transformer"
    ["google-bert/bert-base-uncased"]="bert"
    ["fasttext"]="fasttext"
)

datasets=("amazon-fine-food-reviews_tok_balanced" "labelled_newscatcher_dataset_tok_balanced" "ag_news_tok_balanced" "food-com-recipes-user-interactions_tok_balanced" "amazon-reviews_tok_balanced" "huffpost-news_tok_balanced")
dims=("8" "5" "3")
input_dir="./data/datasets"

sanitize_model_name() {
    echo "$1" | tr '/' '_'
}

for dim in ${dims[@]}; do
    for dataset_base_name in ${datasets[@]}; do
        for model in "${!models[@]}"; do
            model_type="${models[$model]}"
            sanitized_model=$(sanitize_model_name "${model}")
            input_file="${input_dir}/${dataset_base_name}_train_${sanitized_model}.json"
            output_file="${input_dir}/${dataset_base_name}_train_${sanitized_model}_${dim}.json"
            input_file_val="${input_dir}/${dataset_base_name}_dev_${sanitized_model}.json"
            output_file_val="${input_dir}/${dataset_base_name}_dev_${sanitized_model}_${dim}.json"
            input_file_eval="${input_dir}/${dataset_base_name}_test_${sanitized_model}.json"
            output_file_eval="${input_dir}/${dataset_base_name}_test_${sanitized_model}_${dim}.json"
            python ./data/data_processing/reduce_emb_dim.py -it $input_file -ot $output_file -iv $input_file_val -ov $output_file_val -ie $input_file_eval -oe $output_file_eval -n $dim -a PCA
        done
    done
done
