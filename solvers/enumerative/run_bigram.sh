#! /bin/bash

set -ex

# Bootsrapping process. Starting with Uniform
python solve_challenges.py \
    --challenges_path "results/uniform/out.json" \
    --eval_challenges_path "results/uniform/out.json" \
    -m ml_bow_bigram \
    --ml_model rf \
    --copy_prob 0.5 \
    --max_n_progs 10000 \
    --timeout_secs 3600 \
    --threads 40 \
    --learn_from_train_sols \
    --logging_dir results/bootstrap/ml_bigram_0 \
    --out_file results/bootstrap/ml_bigram_0/out.json 


for i in {1..5}; do
    python solve_challenges.py \
        --challenges_path "results/bootstrap/ml_bigram_$(($i-1))/out.json" \
        --eval_challenges_path "results/bootstrap/ml_bigram_$(($i-1))/out.json" \
        -m ml_bow_bigram \
        --ml_model rf \
        --copy_prob 0.5 \
        --max_n_progs 10000 \
        --timeout_secs 3600 \
        --threads 40 \
        --learn_from_train_sols \
        --logging_dir results/bootstrap/ml_bigram_$i \
        --out_file results/bootstrap/ml_bigram_$i/out.json 
done

# Last run until 1M.
python solve_challenges.py \
    --challenges_path "results/bootstrap/ml_bigram_5/out.json" \
    --eval_challenges_path "results/bootstrap/ml_bigram_5/out.json" \
    -m ml_bow_bigram \
    --ml_model rf \
    --copy_prob 0.5 \
    --max_n_progs 1000000 \
    --timeout_secs 3600 \
    --threads 20 \
    --learn_from_train_sols \
    --logging_dir results/bootstrap/ml_bigram_6 \
    --out_file results/bootstrap/ml_bigram_6/out.json 


# Run without self-bootrapping (only over unifrom) until 1M.
python solve_challenges.py \
    --challenges_path "results/uniform/out.json" \
    --eval_challenges_path "results/uniform/out.json" \
    -m ml_bow_bigram \
    --ml_model rf \
    --copy_prob 0.5 \
    --max_n_progs 1000000 \
    --timeout_secs 3600 \
    --threads 40 \
    --learn_from_train_sols \
    --logging_dir results/ml_bigram \
    --out_file results/ml_bigram/out.json 

