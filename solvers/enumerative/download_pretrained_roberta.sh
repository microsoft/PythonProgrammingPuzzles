#! /bin/bash

# Linux commands to download our Roberta model pretrained on Python code.
# Newer vesrions of huggingface transformers don't require this but we need to adjust the rest of the code for them.

set -ex

mkdir tals
mkdir tals/roberta_python

cd tals/roberta_python

wget https://huggingface.co/tals/roberta_python/resolve/main/config.json
wget https://huggingface.co/tals/roberta_python/resolve/main/merges.txt
wget https://huggingface.co/tals/roberta_python/resolve/main/pytorch_model.bin
wget https://huggingface.co/tals/roberta_python/resolve/main/special_tokens_map.json
wget https://huggingface.co/tals/roberta_python/resolve/main/tokenizer_config.json
wget https://huggingface.co/tals/roberta_python/resolve/main/training_args.bin
wget https://huggingface.co/tals/roberta_python/resolve/main/vocab.json

cd ../..
