#! /bin/bash

set -ex

# First, extract rule embeddings.
PYTHONPATH=./ python models/transformers/generate_rule_embeddings.py \
    --model_name_or_path tals/roberta_python \
    --output_dir results/roberta_rule_embeddings

# Bootsrapping process. Starting with Uniform
PYTHONPATH=./ python models/transformers/finetune_transformer.py \
    --challenges_path results/uniform/out.json \
    --eval_challenges_path results/uniform/out.json \
    --model_name_or_path tals/roberta_python \
    --output_dir results/bootstrap/roberta_0 \
    --num_train_epochs 20 \
    --do_train \
    --do_infer \
		--max_n_progs 10000 \
		--timeout_secs 3600 \
		--threads 40 \
    --per_gpu_eval_batch_size 128 \
    --per_gpu_train_batch_size 16 \
    --rule_emb_dir results/roberta_rule_embeddings \
    --overwrite_cache \
    --max_ticks 10000

for i in {1..5}; do
    PYTHONPATH=./ python models/transformers/finetune_transformer.py \
        --challenges_path "results/bootstrap/roberta_$(($i-1))/solutions.json" \
        --eval_challenges_path "results/bootstrap/roberta_$(($i-1))/solutions.json" \
        --model_name_or_path tals/roberta_python \
        --output_dir results/bootstrap/roberta_$i \
        --num_train_epochs 20 \
        --do_train \
        --do_infer \
            --max_n_progs 10000 \
            --timeout_secs 3600 \
            --threads 40 \
        --per_gpu_eval_batch_size 128 \
        --per_gpu_train_batch_size 16 \
        --rule_emb_dir results/roberta_rule_embeddings \
        --overwrite_cache \
        --max_ticks 10000
done

# Last run until 1M.
PYTHONPATH=./ python models/transformers/finetune_transformer.py \
    --challenges_path "results/bootstrap/roberta_5/solutions.json" \
    --eval_challenges_path "results/bootstrap/roberta_5/solutions.json" \
    --model_name_or_path tals/roberta_python \
    --output_dir results/bootstrap/roberta_6 \
    --num_train_epochs 20 \
    --do_train \
    --do_infer \
		--max_n_progs 1000000 \
        --timeout_secs 3600 \
        --threads 40 \
    --per_gpu_eval_batch_size 128 \
    --per_gpu_train_batch_size 16 \
    --rule_emb_dir results/roberta_rule_embeddings \
    --overwrite_cache \
    --max_ticks 10000


# Run without self-bootrapping (only over unifrom) until 1M.
PYTHONPATH=./ python models/transformers/finetune_transformer.py \
    --challenges_path results/uniform/out.json \
    --eval_challenges_path results/uniform/out.json \
    --model_name_or_path tals/roberta_python \
    --output_dir results/roberta \
    --num_train_epochs 20 \
    --do_train \
    --do_infer \
		--max_n_progs 1000000 \
        --timeout_secs 3600 \
		--threads 40 \
    --per_gpu_eval_batch_size 128 \
    --per_gpu_train_batch_size 16 \
    --rule_emb_dir results/roberta_rule_embeddings \
    --overwrite_cache \
    --max_ticks 10000
