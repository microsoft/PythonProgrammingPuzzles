import os
import random
import utils
import glob
import json
from typing import List
from strictfire import StrictFire as Fire  # aborts early on invalid arguments

class WeirdInputsException(Exception):
    pass

def get_inputs(sat: str):
    """Extacts arguments past the first from a function string
    def f(a: Dict[int, str], b=12):
       test

    should give 'b=12'
    """
    sat = sat.replace(" -> bool", "")
    first_line = sat.split("\n")[0].strip()
    if not first_line.endswith("):") and "#" in first_line:
        if "):" in first_line:
            n = first_line.index("):")
            if "#"  in first_line[n:]:
                first_line = first_line[:n + first_line[n:].index("#")].strip()
        else:
            first_line = "" # raises exception below
    if not (first_line.endswith("):") and first_line.startswith("def")):
        raise WeirdInputsException("Weird puzzle, cannot extract inputs", json.dumps(sat))        
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

def main(
    path,
    filtered_name="gen_ps_filtered.txt",
    unfiltered_name=None, # "gen_ps_unfiltered.txt",
    max_sols_per_puzzle=8,
    seed=0):
    """
    Merge the puzzles from the given json input files. Examples:
        python preprocess.py -unfiltered_name=gen_ps_unfiltered.txt -- ~/aicoder/data/gen_125M_RL/*.json

    path: path to search for json files
    filtered_name: path to write puzzles, unfiltered (default: gen_ps_filtered.txt)
    unfiltered_name: path to write filtered puzzles (optional)
    max_sols_per_puzzle: maximum number of solutions per puzzle (default 8)
    seed: random seed (default 0) for reproducibility
    infiles: list of files to read puzzles from (like /path/*.json)
    """
    
    # Make the path so enumeration off that path works, even if it doesn't exist yet
    filtered_path = os.path.join(path, filtered_name)
    os.makedirs(os.path.dirname(filtered_path), exist_ok=True)

    codes = []
    all_codes = []

    # grab all the iter_* data for just this experiment
    gen_paths = [os.path.join(path, "../*/*.json")]

    # grab just the data for this iter_# for this experiment
    # gen_paths = [os.path.join(path, "*.json")]

    for gen_path in gen_paths:
        for filename in sorted(glob.glob(gen_path)):
            print("preprocess filename:", filename)
            js = utils.load_json(filename)
            for f, successes, failures in js:
                for body in sorted(utils.dedup(successes), key=len)[:max_sols_per_puzzle]:

                    try:
                        g = f"def g({get_inputs(f)}):{body}".strip("\\").strip()
                        codes.append(f + "\n\n" + g + "\n\n" + "assert f(g())\n\n")
                    except WeirdInputsException:
                        print("failed to create g")
                        pass
            print(f"{len(codes):,}/{len(all_codes):,} puzzles of preprocessing {filename}")

    print("len(codes)", len(codes))
    codes = utils.dedup(codes)
    print("len(codes) after dedup", len(codes))

    random.shuffle(codes)
    random.shuffle(all_codes)

    # Make it the same number of examples as we got from codex
    codes = codes[:950511]
    print("len(codes) after truncation", len(codes))

    code = "".join(codes)

    utils.save_text_file(code, filtered_path)
    print(f"Wrote filtered results to {filtered_path}")

    assert unfiltered_name is None, "Not supported now, go back to earlier version"
    if unfiltered_name:
        unfiltered_path = os.path.join(path, filtered_name)
        utils.save_text_file("".join(all_codes), unfiltered_path)
        print(f"Wrote unfiltered results to {unfiltered_path}")

    
if __name__ == "__main__":
    Fire(main)
