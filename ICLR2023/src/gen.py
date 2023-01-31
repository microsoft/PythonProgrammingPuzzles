from typing import List
import os
from tqdm import tqdm
import numpy as np
import json
import judge
import inspect
import random
import re
import ast
import time
from collections import Counter
from strictfire import StrictFire as Fire  # aborts early on invalid arguments
import utils
import solve
import torch

def ast_parse_quiet(s: str):
    utils.silence_std_err(True)
    try:
        return ast.parse(s)
    except:
        pass
    finally:
        utils.silence_std_err(False)


def find_end(st: str):
    """Takes a solution and looks for the end that would make it parse."""
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


def strip_puzzle(puz: str):
    """When a puzzle is generated, it will typically be followed by extra code after the def.
    This function strips that extra code, leaving just the puzzle."""
    puz = puz.strip()
    match = re.search(r"\n\S", puz)  # newline followed by a non-newline character
    if match:
        return puz[: match.start()]
    return puz


def good_puzzles(puzzles: List[str], trivial_reject_rate, verbose=True):
    """Find the puzzles that compile, have exactly one required argument of a listful type, and are non-trivial
    meaning that they use the argument somewhere in the puzzle and do not return True on some trivial values.
    Set trivial_reject_rate to 1 if you want to reject all puzzles"""

    # first we make sure they have a return statement and start with 'def f(' and also strip any trailing code

    n = len(puzzles)
    puzzles = [strip_puzzle(p) for p in puzzles]
    puzzles = [p for p in puzzles if p.startswith("def f(") and "return" in p]

    utils.info(f"{len(puzzles):,}/{n:,} = {len(puzzles) / n:.0%} puzzle passed step 1")

    # next we modify the puzzle by inserting a return True as its first line and judge if f(None) is True
    # this removes puzzles with bad signatures or dangerous code as detected by the judge

    def make_true(p):
        lines = p.split("\n")
        lines.insert(1, "    return True")
        lines.append("")
        lines.append("assert f(None)")
        return "\n".join(lines)

    n = len(puzzles)
    puzzles = [p for p, res in zip(puzzles, judge.judge_parallel([make_true(p) for p in puzzles], timeout=1)) if res]

    utils.info(f"{len(puzzles):,}/{n:,} = {len(puzzles) / n:.0%} puzzle passed step 2")

    def get_trivial_tests(p):
        """determine which test to run for trivial tests based on the type, returns None if the spec is invalid"""
        try:
            env = {"List": List}
            exec(p, env)
            f = env["f"]
            spec = inspect.getfullargspec(f)
            ans_var_name = spec.args[0]
            typ = spec.annotations[ans_var_name]
        except:
            return None
        num_var_mentions = len(re.findall(r"\b" + ans_var_name + r"\b", p))
        if (
            len(spec.defaults or []) != len(spec.args) - 1
            or spec.varargs  # need exactly one required parameter
            or spec.varkw
            or spec.kwonlyargs
            or spec.kwonlydefaults  # weird spec: *args, **kwargs
        ):
            return None
        if random.random() > trivial_reject_rate:
            return []  # don't bother to test some small fraction of the puzzles for trivial solution
        if (
            num_var_mentions <= 1
            or typ is bool  # need to use the answer variable other than in the spec  # bool puzzles are all trivial
        ):
            return None
        base_types = {"bool": bool, "float": float, "str": str, "int": int}
        if typ not in base_types.values():
            if str(typ) == "str" and typ is not str:
                return None
            type_str = str(typ).replace("typing.", "")
            inside = type_str.replace("List[", "").replace("]", "")
            if inside not in base_types:
                return None
        if typ is int:
            tests = list(range(-10, 101))
        elif typ is str:
            tests = ["cat", "dog", "aa", "ab", "foo", "bar", "baz", ""]
        elif typ is float:
            tests = [-100.0, -10.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]
        elif typ is bool:
            tests = [True, False]
        else:
            depth = type_str.count("List[")
            if depth == 0:
                return None
            if inside == "int":
                base = list(range(-3, 4))
            elif inside == "str":
                base = ["a", "b", "foo", "bar", "baz"]
            elif inside == "bool":
                base = [True, False]
            elif inside == "float":
                base = [-1.0, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0]
            else:
                return None
            from itertools import product

            tests = []
            for r in range(3):
                tests.extend(list(p) for p in product(base, repeat=r))
            for d in range(depth - 1):
                tests = [[i] for i in tests]
            if [] not in tests:
                tests.append([])
        return tests

    n = len(puzzles)
    tests = [get_trivial_tests(p) for p in puzzles]
    puzzles, testss = zip(*[(p, t) for p, t in zip(puzzles, tests) if t is not None])

    utils.info(f"{len(puzzles):,}/{n:,} = {len(puzzles) / n:.0%} puzzle passed step 3")

    # next remove puzzles with trivial solutions
    # todo: also remove puzzles that raise a NameError exception??
    n = len(puzzles)
    nontrivials = []

    for p, tests in tqdm(list(zip(puzzles, testss))):
        results = judge.judge_parallel(
            [f"{p}\n\ntry:\n    assert f({utils.stringify(t)})\nexcept NameError:\n    pass" for t in tests], timeout=1
        )
        if not any(results):
            nontrivials.append(p)
            if verbose:
                utils.info("*" * 100)
                utils.info(p)

    puzzles = nontrivials
    utils.info(f"{len(puzzles):,}/{n:,} = {len(puzzles) / n:.0%} puzzle passed step 3")

    return puzzles


def load_puzzles(filename, remove_docstring):
    """Returns list of functions and solution headers, one puzzle per problem"""
    JS = utils.load_json(filename)
    fs = []
    sol_headers = []
    seen = set()

    for j in JS:
        name = j["name"].split(":")[0]  # just one puzzle per problem
        if name in seen:
            continue
        seen.add(name)
        f = j["sat"].replace("def sat", "def f")

        fs.append(f)
        sol_headers.append(
            j["sol_header"].replace("def sol", "def g") + ("" if remove_docstring else "\n" + j["sol_docstring"])
        )

    return fs, sol_headers


def gen_from_puzzles(
    filename,
    n,
    per_prompt,
    temp,
    model,
    tokenizer,
    remove_docstring,
    max_tokens,
    trivial_reject_rate,
    gen_tokens=200,    
    stop="\ndef",
    batch_size=16,
):
    """
    Generate based on random selection of puzzles only.
    """
    utils.info(f"Generating puzzles from puzzles")
    time0 = time.time()

    fs, heads = load_puzzles(filename, remove_docstring)

    assert len(fs) == len(set(fs)), "Duplicate puzzles"

    generated = []
    SEPARATOR = "\n\n"

    with tqdm(total=n) as pbar:
        it = 0
        while len(generated) < n:
            # compute prompt
            random.shuffle(fs)

            prompt = None
            for k in range(len(fs) + 1):
                entries = fs[:k] + ["def f("]
                candidate_prompt = SEPARATOR.join([f.replace(" f(", f" f{i + 1}(") for i, f in enumerate(entries)])
                if utils.num_tokens(candidate_prompt, tokenizer) >= max_tokens - gen_tokens:
                    break
                prompt = candidate_prompt

            # candidates = gpt_lib.query(
            #     prompt=prompt,
            #     n=min(per_prompt, n),
            #     temp=temp,
            #     max_tokens=gen_tokens,
            #     stop=stop,
            #     cache_only=cache_only,
            #     notes=(seed, it),
            #     engine=engine,
            # )

            num_gen = min(per_prompt, n)

            while True:  # loop to decrease batch size if necessary
                try:
                    candidates = solve.gen(  # complete prompts
                        prompts=[prompt]*num_gen, 
                        tokenizer=tokenizer, 
                        model=model, 
                        batch_size=batch_size, 
                        temp=temp, 
                        gen_tokens=gen_tokens
                    )
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "CUBLAS_STATUS_ALLOC_FAILED" in str(e):
                        print(str(e))
                        utils.info(f"Out of GPU memory gen.py, reducing batch size {batch_size} -> {batch_size//2}")
                        batch_size //= 2
                        assert batch_size >= 1
                        # torch.cuda.empty_cache()  # not important, just lets nvidia-smi update if anything
                    else:
                        raise
            
            candidates = [c[len(prompt):] for c in candidates]
            assert len(candidates) == num_gen

            generated += [strip_puzzle("def f(" + c) for c in candidates]
            pbar.update(len(candidates))
            it += 1

    return good_puzzles(generated, trivial_reject_rate=trivial_reject_rate)


def get_inputs(sat: str):
    """Extacts arguments past the first from a function string
    def f(a: Dict[int, str], b=12):
       test

    should give 'b=12'
    """
    sat = sat.replace(" -> bool", "")
    first_line = sat.split("\n")[0].strip()
    if not first_line.endswith("):") and "#" in first_line:
        first_line = first_line[: first_line.index("#")].strip()
    if not (first_line.endswith("):") and first_line.startswith("def")):
        # raise WeirdInputsException("Weird puzzle, cannot extract inputs", json.dumps(sat))
        return None
    arg_str = first_line[first_line.index("(") : -len("):")]
    depth = 0
    for i, c in enumerate(arg_str):
        if c == "," and depth == 0:
            return arg_str[i + 1 :].strip()
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
    return ""


def get_prompts(prefix, fs, sol_headers, test_prefix=True):
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
        head = head.replace("def g(", f"def g{i}(")
        head = head.replace("def sol(", f"def g{i}(")
        ans.append(f"{prefix}{f}\n\n{head}")
    return ans


def solve_puzzles(
    puzzles,
    prefix,
    attempts,  # number of attempts to solve each puzzle
    model,
    tokenizer,
    temp,
    solve_tokens=150,
    timeout=1.0,
):

    stop = "\nassert"

    utils.info("=" * 100)
    utils.info(f"Solving with {utils.num_tokens(prefix, tokenizer)} prefix tokens")
    time0 = time.time()

    utils.info(f"Solving {len(puzzles)} given directly")

    sol_headers = [f"def g({get_inputs(f)}):" for f in puzzles]
    prefix = re.sub(r" +$", "", (prefix or "").lstrip(), flags=re.M)  # delete leading/trailing whitespace on each line
    prompts = get_prompts(prefix, puzzles, sol_headers)

    all_results = []
    for p_num, (f, head, prompt) in tqdm(enumerate(zip(puzzles, sol_headers, prompts)), total=len(puzzles)):        
        res = solve.gen(  # complete prompts
                        prompts=[prompt]*attempts, 
                        tokenizer=tokenizer, 
                        model=model, 
                        batch_size=4, 
                        temp=temp, 
                        gen_tokens=solve_tokens
                    )
        res = [r[len(prompt):] for r in res]
        assert len(res) == attempts

        valids = [(find_end(g), i) for i, g in enumerate(res)]
        valids = [(g, i) for (g, i) in valids if g is not None]
        # double parentheses are necessary to avoid cheating where it changes default parameters :-)
        if "def f1(" in prompt:
            for kk in range(1, 10000):
                if f"def f{kk}(" not in prompt:
                    break
            kk -= 1
        else:
            kk = ""
        valids = [(g.replace(f"f{kk}(", "f("), i) for (g, i) in valids]
        results = judge.judge_parallel(
            [f"{f}\n\n{head}{g}\n\nassert test_puzzle(f, g())" for g, _i in valids], timeout=timeout
        )
        successes = [g for (g, i), res in zip(valids, results) if res]
        failures = [g for (g, i), res in zip(valids, results) if not res]
        all_results.append((f, successes, failures))
        # if curr:
        # ans1 = [a for a, _i in curr]
        # if verbose:
        #     utils.info(p_num, "-" * 80)
        #     utils.info(strip_param_annotations(f))
        #     summary = [(a if c == 1 else f"{a} ({c} times)") for a, c in Counter(ans1).most_common(10)]
        #     utils.info(f"{len(curr)} sols, first at attempt #{curr[0][1]}:: {' | '.join(summary)}"[:200])

    n_sol = sum(bool(s) for f, s, _ in all_results)
    n_suc = sum(len(s) for f, s, _ in all_results)
    utils.info(f"Solved {n_sol:,}/{len(puzzles):,} puzzles with a total of {n_suc:,} solutions.")
    utils.info()

    return all_results


def gen(
    out="../outputs/gen/<date>/",
    n=100_000,
    seed=0,
    trivial_reject_rate=0.95,
    temp=0.9,
    temp_solve=None,
    gpu=0,
    train="../data/155_train.json",
    prefix="../data/train_prefix.txt",
    remove_docstring=True,
    model_path="EleutherAI/gpt-neo-125M",
    model_path_solve=None,
    max_tokens=2048,
    per_prompt=64,
    attempts=128,
    only_good=False
):
    """
    Run the generator with given seed

    outfilename: where to write the output
    n: number of puzzles to generate, actual number of puzzles will be smaller after filtering
    seed: random seed
    trivial_reject_rate: what fraction of trival puzzles to reject
    temp: temperature for generation
    temp_solve: temperature for solving (if different than temp, default=None means same as generation temp)
    gpu: the gpu to use (default 0)
    train: path to training data (default: 155_train.json)
    prefix: text filename containing prompt (default: ../data/train_prefix.txt)
    remove_docstring: whether to remove docstrings from puzzles (default=True)
    model_path: path to model for generating puzzles
    model_path_solve: path to model for solving puzzles (if different than model_path for generation)
    max_tokens: maximum number of tokens that can fit in a prompt
    per_prompt: number of puzzles to generate per prompt
    attempts: number of solutions to generate per puzzle
    """
    params = locals().copy()  # store parameters
    utils.info("PARAMETERS:")
    utils.info(params)

    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    torch.cuda.set_device(int(gpu))
    tokenizer = utils.load_tokenizer(model_path)
    model = solve.load_model(model_path, pad_token_id=tokenizer.eos_token_id)  # pad_token_id removes wrnings

    output_path = utils.create_experiment_outpath(out, False)

    prefix = utils.load_text_file(prefix).strip() if prefix else ""
    if prefix:
        prefix += "\n\n"

    time0 = time.time()

    puzzles = gen_from_puzzles(
        filename=train,
        n=n,
        per_prompt=per_prompt,
        temp=temp,
        model=model,
        tokenizer=tokenizer,
        trivial_reject_rate=trivial_reject_rate,
        max_tokens=max_tokens,
        remove_docstring=remove_docstring,
    )
    num_puzzles = len(puzzles)
    utils.info(f"Generated {num_puzzles:,} puzzles.")
    
    puzzles_and_solutions = solve_puzzles(
        puzzles,
        prefix=prefix,
        attempts=attempts,
        model=model,
        tokenizer = tokenizer,
        temp=(temp_solve or temp),
    )

    puzzles_and_solutions.sort(key=lambda z: (len(z[1]), len(z[0])))
    
    if (not only_good):
        out_filename = os.path.join(output_path, "puzzles.json")

        utils.save_json(puzzles_and_solutions, out_filename)

        out_filename2 = out_filename.replace(".json", ".txt").replace(".gz", "")
        with open(out_filename2, "w", encoding="utf8") as file:
            for f, gs, *rest in puzzles_and_solutions:
                if rest:
                    [rest] = rest
                    print(len(gs), len(rest), "/" * 100, file=file)
                else:
                    print(len(gs), "=" * 100, file=file)
                print(f, file=file)
                print(file=file)
                for g in (gs + rest)[:2]:
                    print(g, file=file)
                    print(file=file)

        utils.info("Wrote results to {}.".format(out_filename))
        utils.info("Wrote results to {}.".format(out_filename2))

    # generating puzzles and solutions on cluster we want to limit the amount of data produced
    # save out only the puzzles with their good solutions, forget bad solutions and unsolved problems.
    gp_gs = []
    for f, gs, *rest in puzzles_and_solutions:
        if len(gs) > 0:
            gp_gs.append((f, gs, []))

    out_filename = os.path.join(output_path, "good_puzzles_" + "R0_" + str(gpu) + "_" + time.strftime("%y-%m-%d-%H-%M-%S") + ".json")
    utils.save_json(gp_gs, out_filename)

    time1 = time.time()
    utils.info(f"Took {time1 - time0:.3f} seconds.")
    utils.info(f"Saved as file {out_filename}")

if __name__ == "__main__":
    Fire(gen)
