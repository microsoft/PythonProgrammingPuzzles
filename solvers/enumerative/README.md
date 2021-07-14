# Enumerative puzzle solvers

This folder contains the code for the enumerative models used in our Programming Puzzles paper.
We used python 3.8.0 and the libraries in the `requirements.txt` file.

In a linux machine with python3.8.0 installed, the following commands will set up the environment:
```
virtualenv -p /usr/bin/python3.8 env_solvers
source env_solvers/bin/activate
pip install -r requirements.txt
```

## Uniform solver
```
bash run_uniform.sh
```
This will run the uniform solver for a maximum of 10k trials per puzzle. This is required before training the other parameterized solvers.

To run the uniform with 1M trials per puzzle, simply change the `max_n_progs` argument in the bash script.

## Bigram random forest solver
```
bash run_bigram.sh
```
This will first train a parameterized model with self-bootsrapping (first iteration is based on the unifrom solutions). The last command will train a model without self-bootsrapping.

## Transformers solver
```
bash download_pretrained_roberta.sh
bash run_transformer.sh
```
The first script will download the RoBERTa-Base model that we trained on Python code.

The second script will first train a parameterized model with self-bootsrapping (first iteration is based on the unifrom solutions). The last command will train a model without self-bootsrapping.
