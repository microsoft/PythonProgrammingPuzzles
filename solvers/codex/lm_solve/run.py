"""
Run experiments.

"""

from typing import List, Tuple, Dict, Set
from collections import Counter
import utils
import ast
import time
import re
import astor
import json
import random
import inspect
from tqdm import tqdm
import sys
import os
import logging
import math

from . import gpt_lib
from . import judge


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to avoid parallelism warnings from transformer tokenizer
if os.environ["PYTHONHASHSEED"] != '0':
  print("Warning, please set environment variable PYTHONHASHSEED to 0 for determinism")  

def get_prompts(prefix, fs, sol_headers, multi_line=False, test_prefix=True):
    """adds function numbers after prompt"""

    ans = []
    if test_prefix:
        exec(prefix, dict(List=List))

    if "def f1(" in prefix:
        i = 1
        while f"def f{i}(" in prefix:
            i += 1
    else:
        i = ""

    assert len(sol_headers) == len(fs)
    for f, head in zip(fs, sol_headers):
        f = f.replace("def f(", f"def f{i}(")
        head = head.replace("def g(", f"def g{i}(" )
        head = head.replace("def sol(", f"def g{i}(")
        if multi_line:
            ans.append(f"{prefix}{f}\n\n{head}")
        else:
            ans.append(f"{prefix}{f}\n\nassert True == f{i}(")
    return ans


def load_puzzles(filename, remove_docstring):
    """Returns list of functions and solution headers, one puzzle per problem"""
    JS = utils.load_json(filename)
    fs = []
    sol_headers = []
    seen = set()

    for j in JS:
        name = j["name"].split(":")[0] # just one puzzle per problem
        if name in seen:
            continue
        seen.add(name)
        f = j["sat"].replace("def sat", "def f")

        fs.append(f)
        sol_headers.append(j["sol_header"].replace("def sol", "def g")  + ("" if remove_docstring else "\n" +  j["sol_docstring"]))

    return fs, sol_headers

_std_errs = {"orig": os.dup(2), "devnull": os.open(os.devnull, os.O_RDWR)}

def ast_parse_quiet(s: str):
    global _std_errs
    try:
        os.dup2(_std_errs["devnull"], 2) # to avoid printing the s_push parser when parsing stuff with "((((()))))" 
        return ast.parse(s)        
    except:       
        pass
    finally:
        os.dup2(_std_errs["orig"], 2)


def find_end(st: str, multi_line: bool):
    """Takes a solution and looks for the end that would make it parse."""
    if multi_line:
        lines = st.split("\n")
        for i in range(1, len(lines)):
            line = lines[i]
            if line and line[0] not in " \t":
                lines = lines[:i]
                break
        ans = "\n".join(lines)

        if ast_parse_quiet("def g():" + ans):
            return ans
        else:
            return None


    for i, c in enumerate(st):
        if c == ")":
            if ast_parse_quiet("(" + st[:i + 1]):
                return st[:i].strip()            
    return None


_tokenizer = None  # used only in num_tokens
_tokenizer_old_version = False

def num_tokens(s: str):
    """Compute the number of tokens according to GPT"""
    global _tokenizer, _tokenizer_old_version
    if _tokenizer is None:
        import transformers
        _tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        _tokenizer_old_version = (len(_tokenizer.tokenize("\n\nx")) == 2)  # different between transformer versions :-(            
    return len(_tokenizer.tokenize(s)) + (s.count("\n\n") if _tokenizer_old_version else 0)


def get_inputs(sat: str):
    """Extacts arguments past the first from a function string
    def f(a: Dict[int, str], b=12):
       test
    """
    sat = sat.replace(" -> bool", "")
    first_line = sat.split("\n")[0].strip()
    if "#" in first_line:
        first_line = first_line[:first_line.index("#")].strip()
    if not (first_line.endswith("):") and first_line.startswith("def")):
        print("Weird puzzle, cannot extract inputs", json.dumps(sat))
        return ""
    arg_str = first_line[first_line.index("("):-len("):")]
    depth = 0
    for i, c in enumerate(arg_str):
        if c == "," and depth == 0:
            return arg_str[i + 1:].strip()
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
    return ""

def solve_puzzles(filename_or_puzzles, prefix="", n=1000, temp=0.9, timeout=1.0, sep="###",
                  cache_only=False, seed=0, engine=None, 
                  multi_line=True, limit=None):
    if not prefix:
        assert multi_line == False, "Cannot solve multi-line without a prompt"
    if seed == 0:
        seed = None

    stop = "\nassert" if multi_line else "\n"

    print("=" * 100)
    print(f"Solving with {num_tokens(prefix)} prefix tokens, engine={engine}")
    for k in locals().copy():
        print(f"param {k}: {json.dumps(locals()[k])[:100]}")
    time0 = time.time()

    if isinstance(filename_or_puzzles, str):
        puzzles = utils.load_json(filename_or_puzzles)[:limit]
        print(f"Loaded {len(puzzles)} from `{filename_or_puzzles}`")
    else:
        puzzles = filename_or_puzzles[:limit]  # zzz
        print(f"Solving {len(puzzles)} given directly")
    
    sol_headers = [f"def g({get_inputs(f)}):" for f in puzzles]
    prefix = re.sub(r" +$", "", (prefix or "").lstrip(), flags=re.M)  # delete leading/trailing whitespace on each line
    prompts = get_prompts(prefix, puzzles, sol_headers, multi_line)

    successes = []
    for p_num, (f, head, prompt) in tqdm(enumerate(zip(puzzles, sol_headers, prompts)), total=len(puzzles)):
        res = gpt_lib.query(prompt=prompt, temp=temp, n=n, stop=stop, cache_only=cache_only, notes=seed, engine=engine)
        assert len(res) == n

        valids = [(find_end(g, multi_line), i) for i, g in enumerate(res)]
        valids = [(g, i) for (g, i) in valids if g is not None]
        # double parentheses are necessary to avoid cheating where it changes default parameters :-)
        if multi_line:
            if "def f1(" in prompt:
                for kk in range(1, 10000):
                    if f"def f{kk}(" not in prompt:
                        break
                kk -= 1
            else:
                kk = ""
            valids = [(g.replace(f"f{kk}(", "f("), i) for (g, i) in valids]
            results = judge.judge_parallel([f"{f}\n\n{head}{g}\n\nassert test_puzzle(f, g())" for g, _i in valids],
                                           timeout=timeout)
        else:            
            results = judge.judge_parallel([f"{f}\n\nassert test_puzzle(f, ({g}))" for g, _i in valids], timeout=timeout)
        curr = [g for (g, i), res in zip(valids, results) if res]
        successes.append((f, curr))
        # if curr:
        # ans1 = [a for a, _i in curr]
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


def prompt_experiment(filename, experiment="prompt", prefix="", n=1000, temp=0.9, timeout=1.0,
                      cache_only=False, remove_docstring=False, seed=0, engine=None, 
                      verbose=False):
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

    multi_line = (prefix != "")
    if not prefix:
        assert multi_line == False, "Cannot have multi-line bootstrap without a prompt"
    if seed == 0:
        seed = None

    stop = "\nassert" if multi_line else "\n"

    print("=" * 100)
    print(f"{experiment} expt, remove_docstring={remove_docstring}, engine={engine}, n={n:,}")
    for k in locals().copy():
        print(f"param {k}: {json.dumps(locals()[k])[:100]}")
    time0 = time.time()

    fs, sol_headers = load_puzzles(filename, remove_docstring)

    prefix = re.sub(r" +$", "", (prefix or "").lstrip(), flags=re.M)  # delete leading/trailing whitespace on each line
    prompts = get_prompts(prefix, fs, sol_headers, multi_line)

    sat_sols = []
    for p_num, (f, head, prompt) in tqdm(enumerate(zip(fs, sol_headers, prompts)), total=len(fs)):
        # prompt2 = re.sub(r'\bg\d\(', 'sol(', re.sub(r'\bf\d\(', 'sat(', prompt))
        # prompt2 = re.sub(r'\bg(?=\d\()', 'sol', re.sub(r'\bf(?=\d\()', 'sat', prompt))
        # prompt2 = prompt
        res = gpt_lib.query(prompt=prompt, temp=temp, n=n, stop=stop, cache_only=cache_only, 
                            notes=seed, engine=engine, verbose=True)
        assert len(res) == n
        # print(i, "-" * 80)
        # print(f)
        # print
        valids = [(find_end(g, multi_line), i) for i, g in enumerate(res)]
        valids = [(g, i) for (g, i) in valids if g is not None]
        if multi_line:
            # valids = [(g.replace("sat(", "f("), i) for (g, i) in valids]
            # valids = [(g.replace("sat6(", "f("), i) for (g, i) in valids]
            valids = [(g.replace("f6(", "f("), i) for (g, i) in valids]
            results = judge.judge_parallel([f"{f}\n\n{head}{g}\n\nassert test_puzzle(f, g())" for g, _i in valids],
                                           timeout=timeout)
            
        else:
            results = judge.judge_parallel([f"{f}\n\nassert test_puzzle(f, ({g}))" for g, _i in valids], timeout=timeout)
        curr = [g for (g, i), res in zip(valids, results) if res]
        sol_counts = Counter(curr).most_common()
        sat_sols.append({"sat": f, "n_sols": len(curr), "sol_counts": sol_counts})

        if verbose:
            print("=", p_num, "="*10, f"{len(valids)/n:.1%} valid generations:")
            print(f)
            print()
            for v in valids:
                print(v)
                print("-"*100)
            if curr:
                if False:  # len(ans1) < 1:
                    print(p_num, "-" * 80)
                    print(strip_param_annotations(f))
                    summary = [(a if c == 1 else f"{a} ({c} times)") for a, c in Counter(curr).most_common(10)]
                    summary_str = ' | '.join(summary).replace("\n", "\\n")
                    print(f"{len(curr)} sols, first at attempt #{curr[0]}:: {summary_str}"[:200])
                    print(curr[0])
            
    n_sol = sum(bool(s_s["n_sols"]) for s_s in sat_sols)
    n_suc = sum(s_s["n_sols"] for s_s in sat_sols)
    print(f"Solved {n_sol:,}/{len(fs):,} puzzles with a total of {n_suc:,} total solutions.")
    print()

    return dict(
        experiment=experiment,
        filename=filename,
        engine=engine,
        n=n,
        prefix=prefix,
        seed=seed,
        tot_solved = n_sol,
        tot_solutions = n_suc,        
        sat_sols = sat_sols,
    )


def bootstrap(filename, experiment="bootstrap", n = 1000, ppi=32, temp=0.9, seed=0, timeout=1.0, gen_tokens=150, 
              verbose=False, prefix="", cache_only=False, engine=None, remove_docstring=True):
    """
    Run the bootstrapping experiment

    ppi is the number of attempts per puzzle per iteration
    stop is the token to stop generating on
    seed is the seed of the random number generator
    temp is like 0.9
    timeout is judging timeout
    max_tokens is the maximum number of tokens allowed in a prompt
    gen_tokens is how many tokens to generate at most
    cache_only means do not call GPT LM but instead insist on loading from cache

    returns a list of (num_gen, i, f, a) for each solved puzzle where f is the puzzle, i is the index of the puzzle,
    a is the answer, and num_gen is number of attempts before generation (0-indexed)
    """


    iterations = math.ceil(n / ppi)
    if n % ppi != 0:
        print(f"Bootstrap warning: {n} puzzles not divisible by ppi {ppi}, rounding up")

    max_tokens = (4096 if "davinci-codex" in engine else 2048) # zzzzz

    print("=" * 100)
    print(f"Running GPT-bootstrap experiment with engine {engine}")
    for k in locals().copy():
        print(f"param {k}: {json.dumps(locals()[k])[:100]}")
    time0 = time.time()
    rand = random.Random(seed)

    def get_prompt(f, sol_header): 
        """Returns prompt and function number"""
        nonlocal solutions, rand, max_tokens, gen_tokens, prefix
        
        assert f not in [f2 for (_, _, f2, _) in solutions], "Puzzle already solved"

        s2 = solutions[:]
        rand.shuffle(s2)

        entries = []
        for i, (_, _, f2, g) in enumerate(s2):
            j = i + 1
            example = f2.replace('def f(', f'def f{j}(').strip()
            example += "\n\n"
            example += utils.rename_src_var("f", f'f{j}', utils.rename_src_var("g", f'g{j}', g)).strip()
            example += "\n\n"
            example += f"assert f{j}(g{j}())"
            entries.append(example)

        ans = None
        for k in range(len(entries) + 1):
            
            entries2 = ([prefix] if prefix else []) + entries[:k]

            j = k + 1

            entries2.append(f.replace('def f(', f'def f{j}('))
            entries2.append(sol_header.replace("def sol(", "def g(", 1).replace("def g(", f'def g{j}(', 1))
            
            prompt = "\n\n".join([e.strip() for e in entries2])

            # print(k, num_tokens(prompt))
            if num_tokens(prompt) >= max_tokens - gen_tokens:
                # print("TOO MANY TOKENS", num_tokens(prompt), num_tokens(last_prompt))
                break
            ans = (prompt, j)

        return ans

    if isinstance(filename, str):
        fs, sol_headers = load_puzzles(filename, remove_docstring=remove_docstring)
    else:
        fs = filename  # zzz

    assert len(fs) == len(set(fs)) == len(sol_headers)

    time0 = time.time()

    tot = 0
    stats = [dict(f=f, sol_header=h, gs=[], i=i, raw=[]) for i, (f, h) in enumerate(zip(fs, sol_headers))]
    solved_by_iteration = []
    solutions = []
    sat_sols = [{"sat": f, "failures": iterations * ppi} for f in fs] # alternative format


    for it in (range(iterations) if verbose else tqdm(range(iterations))):
        it_time0 = time.time()
        solved_by_iteration.append(0)

        rand.shuffle(stats)  # do each iteration in a random order

        for count, s in enumerate(stats):
            if s["gs"]:
                continue  # already solved
            if verbose:
                print("*"*20, f"Iteration {it+1}/{iterations} #{count} solved {len(solutions)}/{len(fs)}: {solved_by_iteration}")
            i = s["i"]
            f = s["f"]
            sol_header = s["sol_header"]
            prompt, j = get_prompt(f, sol_header)
            num_solved = sum(bool(s['gs']) for s in stats)
            # if verbose:
            #     if count == 0:
            #         print("Prompt:" + ":" * 80)
            #         print(prompt)
            #     if count % 10 == 0:
            #         print(f"       * It {it}/{iterations} ({count / len(stats):.0%}) puzzle {i} "
            #               f"solved {num_solved} temp={temp}",
            #               flush=True)

            res = gpt_lib.query(
                prompt=prompt,
                n=ppi,
                temp=temp,
                max_tokens=gen_tokens,
                stop="\nassert" ,
                cache_only=cache_only,
                notes=(seed, it),
                engine=engine,
                verbose=True             
            )
            s["raw"].append((prompt, res))
            assert len(res) == ppi

            valids = [(find_end(g, multi_line=True), i) for i, g in enumerate(res)]
            valids = [(g, i) for (g, i) in valids if g is not None]            
            valids = [(sol_header + g.replace("f{j}(", "f("), i) for (g, i) in valids]
            results = judge.judge_parallel([f"{f}\n\n{g}\n\nassert test_puzzle(f, g())" for g, _i in valids],
                                           timeout=timeout)

            if any(results):
                a, m = next((a, m) for ((a, m), res) in zip(valids, results) if res) 
                solutions.append((it * ppi + m, i, f, a))
                assert "sol" not in sat_sols[i]
                sat_sols[i]["sol"] = a
                sat_sols[i]["prompt"] = prompt
                sat_sols[i]["failures"] = m + it * ppi
                solved_by_iteration[-1] += 1
                s["gs"].append((a, it, m))
                if verbose:
                    print(f"# {len(solutions)}-th puzzle solved, iteration {it}: puzzle id={i}")
                    print(f)
                    print()
                    print(f"assert True == f({a})")
                    print()
        it += 1
        num_solved = sum(bool(s['gs']) for s in stats)
        assert sum(solved_by_iteration) == num_solved
        # if verbose:
        #     print()
        #     print()
        #     print(f"+++ Iter {it}: {num_solved} solved {solved_by_iteration}")
        #     print(f"+++ {time.time() - it_time0:.1f}s it ({time.time() - time0:.1f}s)")
        #     print()
        #     print()

    print(f"Solved {len(solutions):,}/{len(fs):,} puzzles")
    print("Solved by iteration", solved_by_iteration)

    return dict(
        experiment=experiment,
        filename=filename,
        engine=engine,
        n=n,
        prefix=prefix,
        seed=seed,
        tot_solved = len(solutions),
        sat_sols = sat_sols
    )





def strip_puzzle(puz: str):
    puz = puz.strip()
    match = re.search(r"\n\S", puz)  # newline followed by a non-newline character
    if match:
        return puz[:match.start()]
    return puz
