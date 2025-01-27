#!/bin/bash

declare -A models
models=(
    ["llmrails/ember-v1"]="transformer"
    ["google-bert/bert-base-uncased"]="bert"
    ["fasttext"]="fasttext"
)

datasets=("amazon-fine-food-reviews_tok_balanced" "labelled_newscatcher_dataset_tok_balanced" "ag_news_tok_balanced" "food-com-recipes-user-interactions_tok_balanced" "amazon-reviews_tok_balanced" "huffpost-news_tok_balanced")
suffices=("train" "dev" "test")
input_dir="./data/datasets"

sanitize_model_name() {
    echo "$1" | tr '/' '_'
}

for dataset_base_name in ${datasets[@]}; do
    for model in "${!models[@]}"; do
        model_type="${models[$model]}"
        sanitized_model=$(sanitize_model_name "${model}")
        for suffix in "${suffices[@]}"; do
            input_file="${input_dir}/${dataset_base_name}_${suffix}.tsv"
            output_file="${input_dir}/${dataset_base_name}_${suffix}_${sanitized_model}.json"
            LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:/home/TILDE.LV/marcis.pinnis/anaconda3/envs/neasqc2/lib/python3.10/site-packages/tensorrt_libs/ bash 4_GetEmbeddings.sh \
                -i "${input_file}" \
                -o "${output_file}" \
                -c 2 \
                -m "${model}" \
                -t "${model_type}" \
                -e "sentence" \
                -g "1"
        done
    done
done
