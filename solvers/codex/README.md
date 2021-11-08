# Running GPT-3 experiments

These are instructions for re-running the Codex experiments. The results will be slightly different than those in 
the paper because the API is non-deterministic.

The requirements can be installed with `pip3 install -r requirements.txt`.

`run_codex_experiments.py` runs the Codex experiments and prints the results to stdout. Change the 
parameters in that file to run it on the 397puzzles (v0.2) or 138puzzles.json (v0.1 used in 
first experiment) or 30puzzles.json (study)
or to use the davinci-codex engine vs cushman-codex.

## Installation and execution.
You will need an open-ai Codex API access key which can be signed up for [here](https://openai.com/join/). 
You will then need to set it as the `OPENAI_API_KEY` environmnet variable. If you want an extension added 
to the engines such as "-msft", set the environment variable `export OPEN_AI_ENGINE_SUFFIX=-msft`. 
We also recommend that you set the environment variable `export PYTHONHASHSEED=0` for determinism.

The requirements can be installed with `pip3 install -r requirements.txt`.

It was run with Python 3.6.9, sys.version = '3.6.9 (default, Jan 26 2021, 15:33:00) \n[GCC 8.4.0]', but should
be compatible with later versions as well. 

Then you simply run
`python run_codex_experiments.py`. It uses cacheing mechanisms with the first run 
being quite slow and verbose, querying the API. However you can subsequently run it again and it will be
much faster and just output the results. The cacheing makes it deterministic so it should give the same
exact results when re-run. 
 
