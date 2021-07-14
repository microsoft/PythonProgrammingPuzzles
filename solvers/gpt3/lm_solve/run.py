"""
A simple (non-bootstrap) version where you give a (possibly empty) tutorial.

"""

from typing import List, Tuple, Dict
from collections import Counter
import utils
import ast
import time
import re
import astor
import json
import random
from tqdm import tqdm
from . import gpt3_lib

from . import judge


def get_prompts(prefix, fs, test_prefix=True):
    """adds function numbers after prompt"""

    ans = []
    if test_prefix:
        if "---" in prefix:
            for p in prefix.split("---"):
                exec(p)

    if "def f1(" in prefix:
        i = 1
        while f"def f{i}(" in prefix:
            i += 1
    else:
        i = ""
    for f in fs:
        f = f.replace("def f(", f"def f{i}(")
        ans.append(f"{prefix}{f}\n\nassert True == f{i}(")
    return ans


def strip_param_annotations(f):
    a = ast.parse(f)
    args = a.body[0].args.args
    for arg in args[1:]:
        arg.annotation = None
    new_f = astor.to_source(a,
                            pretty_source=
                            lambda source: ''.join(astor.source_repr.split_lines(source, maxline=10 ** 10)))
    line_0 = new_f.strip().split("\n")[0]
    return "\n".join([line_0] + f.strip().split("\n")[1:])


def load_puzzles(filename, add_docstring=False):
    JS = utils.load_json(filename)
    ans = []
    seen = set()

    for j in JS:
        name = "_".join(j["name"].split("_")[:-1])
        if name in seen:
            continue
        seen.add(name)
        ft = j["sat"].replace("def sat", "def f")
        f = "\n".join(ft.split("\n")[:1] + ft.split("\n")[2:])  # remove assert
        if add_docstring:
            desc = j["desc"]
            desc = "\n".join(" " * 4 + line if i else line for i, line in enumerate(desc.split("\n")))  # add indent
            f = "\n".join(f.split("\n")[:1] + [f'    """{desc}"""'] + f.split("\n")[1:])

        f = strip_param_annotations(f)

        ans.append((ft, f))

    return ans


def find_close_paren(st):
    """Takes a solution and looks for the close parenthesis that would make it parse"""
    for i, c in enumerate(st):
        if c == ")":
            try:
                ast.parse("(" + st[:i + 1])
                return st[:i].strip()
            except:
                pass
    return None


_tokenizer = None  # used only in num_tokens


def num_tokens(s: str):
    """Compute the number of tokens according to GPT-3"""
    global _tokenizer
    if _tokenizer is None:
        import transformers
        _tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
    return len(_tokenizer.tokenize(s)) + s.count("\n\n")  # not sure why this is needed but seems to work


def prompt_experiment(filename, prefix="", n=1000, temp=0.9, stop="\n", timeout=1.0,
                      cache_only=False, add_docstring=False, seed=0):
    """
    Just run n attempts per puzzle

    n is the number of attempts per puzzle
    temp is like 0.9
    timeout is judging timeout
    cache_only means do not call GPT-3 LM but instead insist on loading from cache
    seed makes it so that you can run the experiment more than once without hitting the cache again.

    returns a list of (f, correct answers) for each puzzle string f
    where correct answers is a list of (string, index found)
    """

    if seed == 0:
        seed = None

    print("=" * 100)
    print(f"Running prompt experiment with {num_tokens(prefix)} prefix tokens, add_docstring={add_docstring}")
    for k in locals().copy():
        print(f"param {k}: {json.dumps(locals()[k])[:100]}")
    time0 = time.time()

    puzzles = load_puzzles(filename, add_docstring)
    prefix = re.sub(r" +$", "", (prefix or "").lstrip(), flags=re.M)  # delete leading/trailing whitespace on each line
    prompts = get_prompts(prefix, [f for ft, f in puzzles])

    successes = []
    for p_num, ((ft, f), prompt) in tqdm(enumerate(zip(puzzles, prompts)), total=len(puzzles)):
        res = gpt3_lib.query(prompt=prompt, temp=temp, n=n, stop=stop, cache_only=cache_only, notes=seed)
        assert len(res) == n
        # print(i, "-" * 80)
        # print(f)
        # print()
        valids = [(find_close_paren(g), i) for i, g in enumerate(res)]
        valids = [(g, i) for (g, i) in valids if g is not None]
        # double parentheses are necessary to avoid cheating where it changes default parameters :-)
        results = judge.judge_parallel([f"{ft}\n\nassert True == f(({g}))" for g, _i in valids], timeout=timeout)
        curr = [(g, i) for (g, i), res in zip(valids, results) if res]
        successes.append((f, curr))
        if curr:
            ans1 = [a for a, _i in curr]
            # if verbose:
            #     print(p_num, "-" * 80)
            #     print(strip_param_annotations(f))
            #     summary = [(a if c == 1 else f"{a} ({c} times)") for a, c in Counter(ans1).most_common(10)]
            #     print(f"{len(curr)} sols, first at attempt #{curr[0][1]}:: {' | '.join(summary)}"[:200])

    n_sol = sum(bool(s) for f, s in successes)
    n_suc = sum(len(s) for f, s in successes)
    print(f"Solved {n_sol:,}/{len(puzzles):,} puzzles with a total of {n_suc:,} solutions.")
    print()

    return successes


def bootstrap(filename, iterations, ppi=32, temp=0.9, stop="\n", seed=0, timeout=1.0,
              max_tokens=2048, gen_tokens=150,
              verbose=False, cache_only=False):
    """
    Run the bootstrapping experiment

    iterations is the number of iterations
    ppi is the number of attempts per puzzle per iteration
    stop is the token to stop generating on
    seed is the seed of the random number generator
    temp is like 0.9
    timeout is judging timeout
    max_tokens is the maximum number of tokens allowed in a prompt
    gen_tokens is how many tokens to generate at most
    cache_only means do not call GPT-3 LM but instead insist on loading from cache

    returns a list of (num_gen, i, f, a) for each solved puzzle where f is the puzzle, i is the index of the puzzle,
    a is the answer, and num_gen is number of attempts before generation (0-indexed)
    """

    print("=" * 100)
    print(f"Running GPT3-bootstrap experiment")
    for k in locals().copy():
        print(f"param {k}: {json.dumps(locals()[k])[:100]}")
    time0 = time.time()



    rand = random.Random(seed)


    def get_prompt(f, sep="\n\n---\n\n"):
        nonlocal solutions, rand, max_tokens, gen_tokens
        cur = f"{f}\n\nassert True == f("
        if not solutions:
            return cur
        s2 = solutions[:]
        rand.shuffle(s2)
        entries = [f"{f}\n\nassert True == f({g})" for (_it, _i, f, g) in s2]

        assert f not in "".join(entries)

        last_prompt = None
        for k in range(len(entries) + 1):
            prompt = sep.join([e.replace(' f(', f' f{i + 1}(') for i, e in enumerate(entries[:k] + [cur])])
            # print(k, num_tokens(prompt))
            if num_tokens(prompt) >= max_tokens - gen_tokens:
                # print("TOO MANY TOKENS", num_tokens(prompt), num_tokens(last_prompt))
                break
            last_prompt = prompt

        return last_prompt

    puzzles = load_puzzles(filename)
    assert len(puzzles) == len({f for ft, f in puzzles}), "Duplicate puzzles"

    time0 = time.time()

    tot = 0
    stats = [dict(f=f, ft=ft, gs=[], i=i, raw=[]) for i, (ft, f) in enumerate(puzzles)]
    solved_by_iteration = []
    solutions = []

    for it in tqdm(range(iterations)):
        it_time0 = time.time()
        solved_by_iteration.append(0)


        rand.shuffle(stats)  # do each iteration in a random order

        for count, s in enumerate(stats):
            if s["gs"]:
                continue  # already solved
            i = s["i"]
            f = s["f"]
            ft = s["ft"]
            prompt = get_prompt(f)
            num_solved = sum(bool(s['gs']) for s in stats)
            if verbose:
                if count == 0:
                    print("Prompt:" + ":" * 80)
                    print(prompt)
                if count % 10 == 0:
                    print(f"       * It {it}/{iterations} ({count / len(stats):.0%}) puzzle {i} "
                          f"solved {num_solved} temp={temp}",
                          flush=True)

            candidates = gpt3_lib.query(
                prompt=prompt,
                n=ppi,
                temp=temp,
                max_tokens=gen_tokens,
                stop=stop,
                cache_only=cache_only,
                notes=(seed, it)
            )
            s["raw"].append((prompt, candidates))
            assert len(candidates) == ppi
            close_parens = [z for z in [(find_close_paren(s), j) for j, s in enumerate(candidates)] if z[0] is not None]
            # print("Judging:")
            # for g, _i in close_parens:
            #     print(f"`{ft}\n\n_ans_=({g})\nassert True == f(_ans_)`".replace("\n", "\\n"))
            results = judge.judge_parallel([f"{ft}\n\n_ans_=({g})\nassert True == f(_ans_)" for g, _i in close_parens],
                                           timeout=timeout)
            if any(results):
                a, j = next((a, j) for ((a, j), res) in zip(close_parens, results) if res)
                solutions.append((it*ppi + j, i, f, a))
                solved_by_iteration[-1] += 1
                s["gs"].append((a, it, j))
                if verbose:
                    print(prompt)
                    print(f"YAY it {it}, puzzle {s['i']}")
                    print(f)
                    print(a)
        it += 1
        num_solved = sum(bool(s['gs']) for s in stats)
        assert sum(solved_by_iteration) == num_solved
        if verbose:
            print()
            print()
            print(f"+++ Iter {it}: {num_solved} solved {solved_by_iteration}")
            print(f"+++ {time.time() - it_time0:.1f}s it ({time.time() - time0:.1f}s)")
            print()
            print()

    print(f"Solved {len(solutions):,}/{len(puzzles):,} puzzles")

    return sorted(solutions)
# sols[2][-18] [('[] + [1] * 20', 218)])
# sols[2][2]
# [('23', 53),
#   ('24', 110),
#   ('23', 174),
#   ('24', 233),
#   ('23', 251),
#   ('23', 332),
#   ('23', 336),
#   ('24', 404),
#   ('24', 410),
#   ('23', 454),
#   ('23', 456),
#   ('23', 605),
#   ('23', 697)])