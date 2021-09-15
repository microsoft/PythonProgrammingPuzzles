"""
This script runs the GPT_3 experiments and prints the results to stdout.
It uses cacheing mechanisms so that if run twice with the same parameters, it will give exactly the same
results and will not query the GPT3 API again and will not judge the resulting solutions again. Hence, the first
time you run it, it will be slow, but you can subsequently run it again and it will be fast. It will run the
experiment three times, with different seeds to get different results.

"""

import lm_solve
import numpy as np

PARAMS = dict(
    temp=0.9,
    timeout=1.0,  # seconds to judge
    n=10 * 1000,
    filename="puzzles_with_descriptions.json",
    stop="\n",
    cache_only=False,  # change this to True if you want to run a 2nd time without risking hitting API
)

BOOTSTRAP_PARAMS = dict(
    temp=PARAMS["temp"],
    timeout=PARAMS["timeout"],
    filename=PARAMS["filename"],
    stop=PARAMS["stop"],
    cache_only=PARAMS["cache_only"],
    ppi=32,  # puzzles per iteration
    iterations=(PARAMS["n"] + 31) // 32,
)

STUDY = range(107, 137)  # the range of puzzles used in the study

PREFIX = """
def f1(s: str):
    return "Hello " + s == "Hello world"

assert True == f1("world")

---

def f2(s: str):
    return "Hello " + s[::-1] == "Hello world"

assert True == f2("world"[::-1])

---

def f3(x: List[int]):
    return len(x) == 2 and sum(x) == 3

assert True == f3([1, 2])

---

def f4(s: List[str]):
    return len(set(s)) == 1000 and all((x.count("a") > x.count("b")) and ('b' in x) for x in s)

assert True == f4(["a"*(i+2)+"b" for i in range(1000)])

---

def f5(n: int):
    return str(n * n).startswith("123456789")

assert True == f5(int(int("123456789" + "0"*9) ** 0.5) + 1)

---

"""  # trailing newlines important

PREFIX_DOCSTR = '''
def f1(s: str):
    """Find a string that when concatenated onto 'Hello ' gives 'Hello world'."""
    return "Hello " + s == "Hello world"

assert True == f1("world")

---

def f2(s: str):
    """Find a string that when reversed and concatenated onto 'Hello ' gives 'Hello world'."""
    return "Hello " + s[::-1] == "Hello world"

assert True == f2("world"[::-1])

---

def f3(x: List[int]):
    """Find a list of two integers whose sum is 3."""
    return len(x) == 2 and sum(x) == 3

assert True == f3([1, 2])

---

def f4(s: List[str]):
    """Find a list of 1000 distinct strings which each have more 'a's than 'b's and at least one 'b'."""
    return len(set(s)) == 1000 and all((x.count("a") > x.count("b")) and ('b' in x) for x in s)

assert True == f4(["a"*(i+2)+"b" for i in range(1000)])

---

def f5(n: int):
    """Find an integer whose perfect square begins with 123456789 in its decimal representation."""
    return str(n * n).startswith("123456789")

assert True == f5(int(int("123456789" + "0"*9) ** 0.5) + 1)

---

'''  # trailing newlines important


def run(seed=0):
    sols = [lm_solve.prompt_experiment(**PARAMS, prefix="", seed=seed),
            lm_solve.prompt_experiment(**PARAMS, prefix=PREFIX, seed=seed),
            lm_solve.prompt_experiment(**PARAMS, prefix=PREFIX_DOCSTR, add_docstring=True, seed=seed)]
    problems_solved = [sorted([i for i, (f, gs) in enumerate(s) if gs]) for s in sols]
    bootstrap = lm_solve.bootstrap(**BOOTSTRAP_PARAMS, seed=seed)
    print(f"run={seed} ALL DONE!\n\n")
    print(f"run={seed} RESULTS " + "=" * 50)
    print()

    # Instead of running until first success, and outputting number of attempts, we do something more accurate.
    # We run for N tries for each problem and do not stop on first success. Then, we use the number of successes
    # for a better estimate of the average number of attempts required for first success. If we have s successes
    # out of N attempts, then the expected number of attempts is (N - s) / (1 + s). This is the expectation of the
    # random variable that is: when you permute the attempts uniformly at random, how many attempts before the
    # first success. If s=N, it's 0, if s=1, it's (N-1)/2, etc.
    counts = [[(PARAMS["n"] - len(gs)) / (1 + len(gs)) for f, gs in s if gs] for s in sols]
    counts.append([m for m, _i, _f, _a in bootstrap])
    counts = [[1 + z for z in c] for c in counts]  # add 1 to make it 1-based
    for c in counts:
        c.sort()
    print(f"run={seed} (Expected) number of attempts before a problem is solved [short, med, long, bootstrap]:")
    print(counts)
    problems_solved.append([i for _m, i, _f, _a in bootstrap])
    print()
    print(f"run={seed} Which problems were solved [short, med, long, bootstrap]:")
    print(problems_solved)
    print()
    print(f"run={seed} Number of problems solved [short, med, long, bootstrap]:")
    print([len(c) for c in counts])
    print()
    print(f"run={seed} Number of 30 study problems solved [short, med, long, bootstrap]:")
    print([len([i for i in s if i in STUDY]) for s in problems_solved])
    print()
    difficulties = [1.0 for _ in range(len(sols[0]))]

    k = 1
    for m, i, f, a in bootstrap:
        difficulties[i] = np.log(m + 1) / np.log(PARAMS["n"])

        # These commented lines print the problems that bootstrap solved
        # print()
        # print(f"# Bootstrap solved after {m + 1} tries:")
        # print(f.replace("def f", "def f" + str(k)))
        # import json
        #
        # print(f"SOL:", json.dumps(a))
        k += 1
    print(f"run={seed} Bootstrap difficulties for study puzzles:")
    print([difficulties[i] for i in STUDY])


if __name__ == "__main__":
    for seed in range(3):
        run(seed)
