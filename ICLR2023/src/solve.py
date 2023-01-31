from strictfire import StrictFire as Fire  # aborts early on invalid arguments
import os
import time
import torch
from transformers import GPTNeoForCausalLM
import deepspeed
import gc
import tqdm
from typing import List
import ast
import random
import numpy as np
import judge
import utils


def load_model(model_path, pad_token_id=None, mp_size=1):
    start_time = time.time()
    model = GPTNeoForCausalLM.from_pretrained(model_path, pad_token_id=pad_token_id).half()
    utils.info(f"Loaded model in {time.time()-start_time:.1f}s")
    
    print("deepspeed version", deepspeed.__version__)
    print("mp_size", mp_size)
        
    if (deepspeed.__version__[:5] >= "0.6.0"):
        # This works on deepspeed 0.6.0 and later - deepspeed updated their tutorials and docs 
        return deepspeed.init_inference(model, mp_size=mp_size, dtype=torch.float16, replace_method="auto", replace_with_kernel_inject=True).module        
    else:
        # This works on deepspeed 0.5.1 and 0.5.6
        return deepspeed.init_inference(model, mp_size=mp_size, dtype=torch.float16, replace_method="auto").module

def get_puz_num_str(prefix: str):
    """
    If the prefix has def f1 ... def f5, it returns "6", otherwise it returns ""
    """
    if "def f1(" in prefix:
        i = 1
        while f"def f{i}(" in prefix:
            i += 1
        return str(i)
    else:
        return ""


def gen_prompts(fs: List[str], prefix: str) -> str:
    # extract everything after first argument
    ans = []

    puz_num_str = get_puz_num_str(prefix)

    for f in fs:
        args = f[f.index("(") + 1 : f.index("):\n")]
        if "," in args:
            inputs = args[args.index(",") + 1 :].strip()
        else:
            inputs = ""

        f_new = f.replace("def f(", f"def f{puz_num_str}(").strip()
        prompt = f"{prefix}{f_new}\n\ndef g{puz_num_str}({inputs}):"

        ans.append(prompt)

    return ans


def trim_gen_texts(gen_texts, prefix):
    """
    Trim the generated texts to remove the prefix and find the end of the generated function
    """
    # utils.silence_std_err(True)

    p_num = get_puz_num_str(prefix)

    assert all(t.startswith(prefix) for t in gen_texts)

    texts = [text[len(prefix) :] for text in gen_texts]  # remove prefix
    # for t in texts:
    #     print("====")
    #     print(t)

    texts = [text.replace(f"f{p_num}(", "f(").replace(f"g{p_num}(", "g(") for text in texts]  # remove f<num>

    gs = []
    for t in texts:  # for sat, t in zip(sats, texts):
        # print("-t", t)
        # print("-f", f)
        # f = sat.replace("def sat(", "def f(")
        # assert t.strip().startswith(f.strip())
        # assert t.startswith(f)
        gs.append(t[t.index("def g(") :].strip())

    results = []
    for st in gs:
        lines = st.split("\n")
        for i in range(1, len(lines)):
            line = lines[i]
            if line and line[0] not in " \t":
                lines = lines[:i]
                break
        g = "\n".join(lines).strip()

        try:
            ast.parse(g)
            results.append(g)
        except:
            results.append(None)

    return results


def gen(prompts, tokenizer, model, batch_size, temp, gen_tokens):

    # print("generating")
    start_time = time.time()

    gen_texts = []
    for start_i in range(0, len(prompts), batch_size):
        cur_prompts = prompts[start_i : start_i + batch_size]
        tokens = tokenizer(cur_prompts, padding=True, return_tensors="pt").input_ids.cuda()
        max_length = tokens.shape[1] + gen_tokens  # ids.shape[1] is num_tokens of the longest prompt
        # print(tokenizer.batch_decode(ids)[0])
        with torch.no_grad():
            assert max_length <= 2048
            generated = model.generate(
                tokens,
                do_sample=(temp != 0),
                min_length=max_length,  # total length including prompt
                max_length=max_length,
                temperature=(temp or None),  # use None if temp == 0.0
                use_cache=True,
                # num_return_sequences=num_return,
            )
        # possibly todo: trim all generations to gen_tokens length?
        gen_texts.extend(tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    duration = time.time() - start_time
    utils.info(f"Generated {len(gen_texts)} texts in {duration:.1f}s")
    assert len(gen_texts) == len(prompts)
    assert all(t.startswith(prompt) for t, prompt in zip(gen_texts, prompts))
    return gen_texts


def solve(
    puzzles="../data/test_228.json",
    prefix="../data/train_prefix.txt",
    attempts=10,
    fixed_temp=None,
    model_path="EleutherAI/gpt-neo-2.7B",
    gpu=0,
    batch_size=4, # Sometimes shrinking the batch size doesn't work, like A100 need 8 on 2.7B to run - dies otherwise.
    gen_tokens=150,
    out="../outputs/solve/<date>/",
    seed=0
):
    """
    Solve puzzles. Writes the results in outputs/solve/date-time folder.

    puzzles: the file containing the puzzles to solve (default: ../data/test_228.json)
    prefix: text filename containing prompt (default: ../data/tutorial_prefix.txt)
    attempts: number of attempts to solve each puzzle (default: 10)
    fixed_temp: the temperature to use for the solver, if None it will automatically increase temperature (default: None)
    model_path: the path to the model to fine tune (default "EleutherAI/gpt-neo-2.7B")
    gpu: which gpu to use, currently only supports one gpu (default: 0)
    batch_size: initial GPU batch size, automatically reduced if needed (default: 64)
    gen_tokens: minimum number of tokens to generate per solution (default: 150) # todo make all equal
    out: the path to write the output to, with <date> filled in (default: ../outputs/solved/<date>/)
    seed: random seed to use (default: 0)
    """
    params = locals().copy()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['WORLD_SIZE'] = "4"

    if fixed_temp == 0.0:
        utils.warn("*** Fixed temp is 0.0, boiling instead")

    start_time = time.time()
    sats = utils.load_json(puzzles)
    fs = [s.replace("def sat(", "def f(").strip() for s in sats]
    prefix = utils.load_text_file(prefix).strip() if prefix else ""
    if prefix:
        prefix += "\n\n"
    print("out", out)
    output_path = utils.create_experiment_outpath(out)

    prompts = gen_prompts(fs, prefix)
    prompts_by_f = {f: prompt for f, prompt in zip(fs, prompts)}

    utils.save_json(prompts, os.path.join(output_path, "prompts.json"))
    results_filename = os.path.join(output_path, "results.json")

    # gpu if positive is which gpu to use
    # gpu if negative is how many gpu to use
    print("gpu", gpu)
    if (int(gpu) < 0) : 
        from mpi4py import MPI
        mpi_rank = MPI.COMM_WORLD.Get_rank()
        mpi_size = MPI.COMM_WORLD.Get_size()
        print("mpi_rank and mpi_size", mpi_rank, mpi_size)

        port = 29600
        os.environ["MASTER_ADDR"] = '127.0.0.1'
        os.environ["MASTER_PORT"] = str(port)

        print("calling init_distributed")
        deepspeed.init_distributed()
        mp_size = abs(int(gpu))
    else:
        torch.cuda.set_device(int(gpu))  # problematic line for multiple gpus
        mp_size = 1
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())

    tokenizer = utils.load_tokenizer(model_path)
    model = load_model(model_path, pad_token_id=tokenizer.eos_token_id, mp_size=mp_size)  # pad_token_id removes warnings

    all_gen = {f: [] for f in fs}  # all generated solutions for each puzzle
    solutions = {}  # to record the solutions

    current_fs = fs.copy() # the puzzles we are solving at the current temperature
    next_fs = []  # puzzles to solve at the next temperature

    if fixed_temp:
        temp = fixed_temp
        delta_temp = 0.0
    else:
        temp = 0.0
        delta_temp = 0.2

    while current_fs or next_fs:       
        # filter out solved puzzles and puzzles that have already been tried attempts times, and puzzles to be advanced
        current_fs = [f for f in current_fs if len(all_gen[f]) < attempts and f not in solutions and f not in next_fs]         
        if not current_fs:
            current_fs, next_fs = next_fs, []
            temp += delta_temp
            continue

        potential_attempts = sum([attempts - len(all_gen[f]) for f in current_fs + next_fs])
        utils.info(
            f"{len(solutions):,} solved; {potential_attempts:,} potential remaining attempts; " + 
            f"{len(current_fs):,} at temp={temp:.2f}; " + 
            ("" if fixed_temp else f"{len(next_fs):,} at temp={temp+delta_temp:.2f}")
        )
        while True:  # loop to decrease batch size if necessary
            try:
                gen_texts = gen(
                    [prompts_by_f[f] for f in current_fs],
                    tokenizer,
                    model,
                    batch_size,
                    temp,
                    gen_tokens,
                )
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUBLAS_STATUS_ALLOC_FAILED" in str(e):
                    utils.info(f"Out of GPU memory solve.py, reducing batch size {batch_size} -> {batch_size//2}")
                    batch_size //= 2
                    assert batch_size >= 1
                    torch.cuda.empty_cache()  # not important, just lets nvidia-smi update if anything
                else:
                    raise

        assert len(gen_texts) == len(current_fs)
        gs = trim_gen_texts(gen_texts, prefix)
        for f, g in zip(current_fs, gs):
            if not fixed_temp:
                if (temp==0.0 or (g and any(g == g2 for g2, _temp in all_gen[f]))):
                    next_fs.append(f)  # increase temperature when you see a repeated solution or if temperature is 0.0
                    # this will also cause it to be removed from current_fs at the beginning of the next loop               
            all_gen[f].append([g, temp])
        parsed = [(f, g) for f, g in zip(current_fs, gs) if g]
        judge_srcs = [f"{f}\n{g}\nassert test_puzzle(f, g())" for f, g in parsed]
        judgments = judge.judge_parallel(judge_srcs, timeout=1.0)
        assert len(judgments) == len(parsed) == len(judge_srcs) <= len(current_fs)
        for (f, g), solved in zip(parsed, judgments):
            assert f not in solutions
            if solved:
                solutions[f] = g
    for f in all_gen:
        if f not in solutions:
            assert len(all_gen[f]) == attempts
            all_gen[f].append("# UNSOLVED #")  # makes len(all_gen[f]) > attempts

    scores = sorted([len(all_gen[f]) for f in solutions])
    print(f"{len(solutions):,} solved at:", scores)
    passes = {}
    k = 1
    while True:
        passes[k] = sum(len(gens) <= k for gens in all_gen.values())
        if k == attempts:
            break
        k = min(2 * k, attempts)

    utils.info(f"                         Pass@k out of {len(fs):,} puzzles")
    utils.info("                        k: ", "".join(f"{k:6,}" for k in passes))
    utils.info("# solved in <= k attempts: ", "".join(f"{passes[k]:6,}" for k in passes))

    duration_mins = (time.time() - start_time) / 60
    results = dict(
        duration_mins=duration_mins, passes=passes, scores=scores, params=params, solutions=solutions, all_gen=all_gen
    )
    utils.info(f"Saved results generations to '{results_filename}'. Took {duration_mins:.1f} minutes.")
    utils.save_json(results, results_filename)

    # cleanup model (for multithreaded, this happens anyways when the process dies)
    # here we delte the model and reset worker_data to None (really either of the two should work
    # but just being extra careful)
    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    Fire(solve)
