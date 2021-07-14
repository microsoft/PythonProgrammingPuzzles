# Running GPT-3 experiments

These are instructions for re-running the GPT-3 experiments. The results will be slightly different than those in 
the paper because the API is non-deterministic.

The requirements can be installed with `pip3 install -r requirements.txt`.

This script runs the GPT-3 experiments and prints the results to stdout.

## Installation and execution.
You will need an open-ai GPT-3 access key which can be signed up for [here](https://openai.com/join/). 
You will then need to set it as the `OPENAI_API_KEY` environmnet variable.

The requirements can be installed with `pip3 install -r requirements.txt`.

It was run with Python 3.6.9, sys.version = '3.6.9 (default, Jan 26 2021, 15:33:00) \n[GCC 8.4.0]', but should
be compatible with later versions as well. 

Then you simply run
`python run_gpt3_experiments.py` and the results are written to stdout. It uses cacheing mechanisms with the first run 
being quite slow and verbose, querying the API. However you can subsequently run it again and it will be
much faster and just output the results. The cacheing makes it deterministic so it should give the same
exact results when re-run. 
 

## Contact

If you are interested in reproducing the exact results of the paper, please contact the authors to ensure the exact
same query results.