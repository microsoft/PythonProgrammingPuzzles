"""
This script runs the codex experiments.
For GPT-3 experiments see run_gpt3_experiments.py in https://github.com/microsoft/PythonProgrammingPuzzles/tree/v0.1
It uses cacheing mechanisms so that if run twice with the same parameters, it will give exactly the same
results and will not query the API again and will not judge the resulting solutions again. Hence, the first
time you run it, it will be slow, but you can subsequently run it again and it will be fast. It will run the
experiment three times, with different seeds to get different results.
"""

import lm_solve
import utils
import math
import numpy as np

OUTPUT_FILENAME = "results_30_cushman_codex_32.json"
SEEDS = 1 # number of times to run it
PARAMS = dict(
    temp=0.9,
    timeout=1.0,  # seconds to judge
    n=32, # number of attempts per puzzle, usually 1000, or set small for a fast run
    filename="30puzzles.json",  # set to 397puzzles.json for a run on full v0.2 dataset
    cache_only=False,  # change this to True if you want to run a 2nd time without risking hitting API
    engine="cushman-codex",  # FAST-CODEX: "cushman-codex" CODEX: "davinci-codex"  GPT3: "davinci"
)

BOOTSTRAP_PARAMS = dict(
    temp=PARAMS["temp"],
    n=PARAMS["n"],
    timeout=PARAMS["timeout"],
    filename=PARAMS["filename"],
    cache_only=PARAMS["cache_only"],
    ppi=32,  # puzzles per iteration
    engine=PARAMS["engine"],
    prefix="from typing import List\n\n",
)

PREFIX = """from typing import List

def f1(s: str):
    return "Hello " + s == "Hello world"

def g1():
    return "world"

assert f1(g1())

def f2(s: str):
    return "Hello " + s[::-1] == "Hello world"

def g2():
    return "world"[::-1]

assert f2(g2())

def f3(x: List[int]):
    return len(x) == 2 and sum(x) == 3

def g3():
    return [1, 2]

assert f3(g3())

def f4(s: List[str]):
    return len(set(s)) == 1000 and all((x.count("a") > x.count("b")) and ('b' in x) for x in s)

def g4():
    return ["a"*(i+2)+"b" for i in range(1000)]

assert f4(g4())

def f5(n: int):
    return str(n * n).startswith("123456789")

def g5():
    return int(int("123456789" + "0"*9) ** 0.5) + 1

assert f5(g5())

"""  # trailing newlines important


PREFIX_DOCSTR = '''from typing import List

def f1(s: str):
    return "Hello " + s == "Hello world"

def g1():
    """Find a string that when concatenated onto 'Hello ' gives 'Hello world'."""
    return "world"

assert f1(g1())

def f2(s: str):
    return "Hello " + s[::-1] == "Hello world"

def g2():
    """Find a string that when reversed and concatenated onto 'Hello ' gives 'Hello world'."""
    return "world"[::-1]

assert f2(g2())

def f3(x: List[int]):
    return len(x) == 2 and sum(x) == 3

def g3():
    """Find a list of two integers whose sum is 3."""
    return [1, 2]

assert f3(g3())

def f4(s: List[str]):
    return len(set(s)) == 1000 and all((x.count("a") > x.count("b")) and ('b' in x) for x in s)

def g4():
    """Find a list of 1000 distinct strings which each have more 'a's than 'b's and at least one 'b'."""
    return ["a"*(i+2)+"b" for i in range(1000)]

assert f4(g4())

def f5(n: int):
    return str(n * n).startswith("123456789")

def g5():
    """Find an integer whose perfect square begins with 123456789 in its decimal representation."""
    return int(int("123456789" + "0"*9) ** 0.5) + 1

assert f5(g5())

'''  # trailing newlines important


def pass_at_k(k: int, successes: int, attempts: int):
    fail_prob = 1.0
    for i in range(k):
        fail_prob *= (attempts - successes)/attempts # gets right answer of 0 when attempts == successes
        attempts -= 1
    return 1.0 - fail_prob



def run(seed=0):
    PARAMS_0 = {**PARAMS, "n": 0}
    BOOTSTRAP_PARAMS_0 = {**BOOTSTRAP_PARAMS, "n": 0}
    sols = [lm_solve.prompt_experiment(**PARAMS, experiment="short", prefix="", seed=seed),
            lm_solve.prompt_experiment(**PARAMS, experiment="med",  prefix=PREFIX, remove_docstring=True, seed=seed),
            lm_solve.prompt_experiment(**PARAMS, experiment="long", prefix=PREFIX_DOCSTR, remove_docstring=False, seed=seed),
            ]
    num_puzzles = len(sols[0]["sat_sols"])
    assert all(len(s["sat_sols"]) == num_puzzles for s in sols)

    n = PARAMS["n"]
    ks = [1]
    while ks[-1] < n:
        ks += [ks[-1] * i for i in [10]] # for i in [2, 5, 10]]
    ks = [k for k in ks if k <= n]
    if ks[-1] != n:
        ks.append(n)
    for s in sols:
        s["pass@k"] = [
            (
                k,
                np.mean([pass_at_k(k, s_s["n_sols"], n) for s_s in s["sat_sols"]])
                )
            for k in ks]

    bootstrap = lm_solve.bootstrap(**BOOTSTRAP_PARAMS, seed=seed)
    bootstrap["pass@k"] = [(k, np.mean([s_s["failures"] < k for s_s in bootstrap["sat_sols"]])) for k in ks]
    sols.append(bootstrap)

    print(f"run={seed} ALL DONE!\n\n")
    print(f"run={seed} RESULTS " + "=" * 50)
    print()

    for s in sols:
        print(s["experiment"], "prefix:", s["prefix"].replace("\n", "\\n")[:250])
        print("   ", s["tot_solved"], "solved, pass@k", " ".join(f'{k} {p:.5f}' for k, p in s["pass@k"]))

    print(f"Pass at k [(k, {', '.join(s['experiment'] for s in sols)}) ...]")
    print(list(zip([k for k, _p in sols[0]["pass@k"]], *[[p for _k, p in s["pass@k"]] for s in sols])))

    return sols

def main():
    res = [s for seed in range(SEEDS) for s in run(seed)]

    if OUTPUT_FILENAME:
        FULL_FILENAME = OUTPUT_FILENAME.replace(".json", "_full.json.gz")
        utils.save_json(res, FULL_FILENAME)
        for s in res:
            if "sat_sols" in s:
                for t in s["sat_sols"] :
                    if "sol_counts" in t:
                        if t["sol_counts"]:
                            t["shortest_sol"] = min([s for s, c in t["sol_counts"]], key=len)
                            t["longest_sol"] = max([s for s, c in t["sol_counts"]], key=len)
                            t["common_sol"] = max(t["sol_counts"], key=lambda z: z[1])[0]
                        del t["sol_counts"]
        utils.save_json(res, OUTPUT_FILENAME)
        print(f"saved results to {OUTPUT_FILENAME} and {FULL_FILENAME}")


if __name__ == "__main__":
    main()

