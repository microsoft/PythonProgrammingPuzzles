import argparse
import time
import sys
import inspect

import puzzle_generator
import utils
import generators  # This loads all the problem generators

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--target_num_per_problem',
                    '-n',
                    type=int,
                    default=10,
                    help='Target number of variants to generate per problem.')

parser.add_argument('--json',
                    '-j',
                    type=str,
                    default="puzzles/puzzles.json",
                    help='Filename to store puzzles.')

parser.add_argument('--readme',
                    '-r',
                    type=str,
                    default="puzzles/README.md",
                    help='Filename to store README description of puzzles.')

TOP = """# Python Programming Puzzles: dataset summary
This document summarizes the dataset stored in the `puzzles.json` file in this directory. 
These files are generated from the `generators/*.py` files.
The only import for puzzles is `from typing import List` but you should also pass a candidate solution 
through `check_solution_type` from `puzzle_generator.py` before certifying correctness. 

## Puzzles by module:

{}

----

"""


def save_readme(gen_modules, filename):
    table = ""
    content = ""
    tot_puzzles = 0
    tot_instances = 0
    for name, module_stuff in gen_modules.items():
        section = ""
        sec_name = name.split(".")[-1]
        section += f"## {sec_name}\n\n"
        section += f"{module_stuff['docstring']}\n\n"
        puzzles = module_stuff['examples']
        n = len(puzzles)
        link = f"[{sec_name}](#{sec_name.lower().replace(' ', '-')})"
        n_instances = sum(p["n_instances"] for p in puzzles)
        tot_instances += len(puzzles)
        tot_instances += n_instances
        tot_puzzles += 1
        table += f"- [{sec_name} ({len(puzzles):,} problems, {n_instances:,} instances)](#{sec_name.lower().replace(' ', '-')})\n"
        for i, puzzle in enumerate(puzzles):
            section += f"### {puzzle['name']}\n"
            section += f"{puzzle['desc']}\n\n"
            section += f"```python\n{puzzle['sat']}\n```\n"
            if len(puzzle['sols']) > 0:
                section += "<details><summary>"
            section += f"{len(puzzle['sols'])} solution{'s' if len(puzzle['sols'])!=1 else ''} "
            section += f"to puzzle {link} {i + 1:,}/{n:,}"
            section += f", {puzzle['n_instances']:,} instance{'s' if puzzle['n_instances'] > 1 else ''}"
            if len(puzzle['sols']) > 0:
                section += "</summary>\n\n"
            for sol in puzzle['sols']:
                section += f"```python\n{sol}\n```\n\n"
                if len(puzzle['sols']) > 0:
                    section += "</details>\n\n"

        content += section

    table += f"\nTotal ({tot_puzzles:,} problems, {tot_instances:,} instances)\n"

    content = TOP.format(table) + content

    with open(filename, "w", encoding='utf8') as f:
        f.write(content)


def main(args):
    start_time = time.perf_counter()
    if puzzle_generator.PuzzleGenerator.Debug.subclass_descendents():  # if there are debug problems, don't make the dataset
        puzzle_generator.PuzzleGenerator.debug_problems()
        print("Didn't make dataset because there are `Problem.Debug` problems, remove the `.Debug` to proceed.")
        return

    try:
        last_version = utils.load_json(args.json)
        already_tested_cache = {puz["sat"]: {sol for sol in puz["sols"]} for puz in last_version}
    except:
        already_tested_cache = {}

    utils.info(f"Using {len(already_tested_cache):,} last puzzle testings for speeding up (to avoid re-testing)")

    gens = puzzle_generator.PuzzleGenerator.subclass_descendents()

    gens_by_module = utils.inv_dict({g: g.__module__.split(".")[-1] for g in gens})

    utils.info(f"Python version {sys.version}")
    utils.info(f"Generating from templates: {list(gens_by_module)}")
    puzzles = []
    summaries = {}

    for module_name, gens in gens_by_module.items():  # order determined by generators/__init__.py
        examples = []
        for cls in gens:
            gen = cls()
            gen.build(args.target_num_per_problem, already_tested_cache)
            instances = [
                {
                    "name": i.name,
                    "sat": i.src,
                    "sols": i.sol_srcs,
                    "module": module_name
                }
                for i in gen.instances]
            puzzles.extend(instances)
            examples.append({
                "name": gen.name,
                "desc": gen.desc,
                "sat": gen.instances[0].src,
                "sols": gen.instances[0].sol_srcs,
                "n_instances": len(instances)
            })

        summaries[module_name] = {
            "docstring": inspect.getmodule(gens[0]).__doc__,
            "examples": examples
        }

    utils.save_json(puzzles, args.json, make_dirs_if_necessary=True, indent=2)
    save_readme(summaries, args.readme)
    utils.info(f"Elapsed time: {(time.perf_counter() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
