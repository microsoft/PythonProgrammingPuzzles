import argparse
import time
import sys
import inspect
import re

import puzzle_generator
import utils
import generators  # This loads all the problem generators

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--target_num_per_problem',
                    '-n',
                    type=int,
                    default=5,
                    help='Target number of variants to generate per problem.')

parser.add_argument('--solutions',
                    '-s',
                    default='',
                    help='Filename of AI solutions to add to README, if any.')

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

TOP = """# Summary of Puzzles
This document summarizes the dataset stored in the `puzzles.json` file in this directory. 
These files are generated from the `generators/*.py` files.
The only import for puzzles is `from typing import List` but you should also pass a candidate solution 
through `type_check` from `puzzle_generator.py` before certifying correctness. 

## Puzzles by module: <!-- descriptions come from the module docstring --> 

{}

----

"""


def rename_src_var(orig, new, src, count=0):
    def helper(s):
        return re.sub(r'\b' + orig + r'\b(?!["\'])', new, s, count)
    if src.count('"""') >= 2:
        a = src.index('"""')
        b = src.index('"""', a+1) + 3
        if count == 1:
            h = helper(src[:a])
            if h != src[:a]:
                return h + src[a:]
        return helper(src[:a]) + src[a:b] + helper(src[b:])

    return helper(src)

def anchor(name):
    return name.strip().lower().replace(' ', '-')

def indent(x, spaces=4):
    return (" "*spaces + x.replace("\n", "\n" + " "*spaces))[:-spaces if x.endswith("\n") else None]

def save_readme(gen_modules, filename, sol_filename):
    ai_sols = run_name = run_desc = None
    if sol_filename:
        sols_js = utils.load_json(sol_filename)
        run_name = sols_js["run_name"]
        run_desc = sols_js["run_desc"]
        ai_sols = {}
        for f, entry in sols_js["sols"].items():
            assert f.startswith("def ")
            f_name = f[len("def "):f.index("(")].strip()
            f2 = rename_src_var(f_name, "sat", f, 1)
            entry2 = ai_sols[f2] = entry.copy()
            g = entry2["sol"]
            g_name = g[len("def "):f.index("(")].strip()
            entry2["sol"] = rename_src_var(g_name, "sol", rename_src_var(f_name, "sat", entry2["sol"]))
            entry2["longest_sol"] = rename_src_var(g_name, "sol", rename_src_var(f_name, "sat", entry2["longest_sol"]))

    table = ""
    content = ""
    tot_puzzles = 0
    tot_instances = 0
    for module_name, module_stuff in gen_modules.items():
        section = ""
        sec_name = module_name.split(".")[0]
        section += f"## {sec_name}\n\n"
        section += f"{module_stuff['docstring']}\n\n"
        puzzles = module_stuff['examples']
        if ai_sols:
            puzzles = sorted(puzzles,
                             key=lambda f: (ai_sols.get(f['sat']) or {}).get("num_sols", 0),
                             reverse=True)
        n = len(puzzles)
        link = f"[{sec_name}](#{anchor(sec_name)})"
        n_instances = sum(p["n_instances"] for p in puzzles)
        tot_instances += n_instances
        table += f"- [{module_name}](../generators/{module_name}), [{len(puzzles):,} problems](#{anchor(sec_name)}): {module_stuff['docstring'].strip().rstrip('.')}\n"
        for i, puzzle in enumerate(puzzles):
            tot_puzzles += 1
            f = puzzle['sat']
            puzzle_text = f'* <a name="{anchor(puzzle["name"])}"></a>**{puzzle["name"]}** {puzzle["desc"].strip()}'
            puzzle_text += f' ({puzzle["n_instances"]} instance{"s" if puzzle["n_instances"] != 1 else ""})\n\n'
            puzzle_text += f"```python\n{f}\n\n{puzzle['sol_header']}\n```\n"
            if ai_sols:
                sol_entry = ai_sols.get(f)
                if sol_entry:
                    num_ai_sols = sol_entry['num_sols']
                    if num_ai_sols > 0:
                        puzzle_text += "<details><summary>"
                    puzzle_text += f"{num_ai_sols:,} AI solution{'s' if num_ai_sols != 1 else ''} from {run_name}"
                    if num_ai_sols > 0:
                        if num_ai_sols > 1:
                            puzzle_text += " (shortest and longest ones below)"
                        puzzle_text += "</summary>\n\n"
                        puzzle_text += f"```python\n{sol_entry['sol']}\n```\n\n"
                        if num_ai_sols > 1:
                            puzzle_text += f"```python\n{sol_entry['longest_sol']}\n```\n\n"
                        puzzle_text += "</details>\n\n"
                else:
                    puzzle_text += f"{run_name} was not run on this puzzle\n\n"

            if len(puzzle['sols']) > 0:
                puzzle_text += "<details><summary>"
            puzzle_text += f"{len(puzzle['sols']):,} hand-written solution{'s' if len(puzzle['sols']) != 1 else ''} "
            if len(puzzle['sols']) > 0:
                puzzle_text += "</summary>\n\n"
                for sol in puzzle['sols']:
                    puzzle_text += f"```python\n{sol}\n```\n\n"
                puzzle_text += "</details>\n\n"
            else:
                puzzle_text += "\n\n"
            section += indent(puzzle_text, 4)[4:]

        content += section

    table += f"\nTotal ({tot_puzzles:,} problems, {tot_instances:,} instances)\n"

    content = TOP.format(table) + content
    if run_name:
        content = content.replace(
            "Summary of Puzzles",
            f"Summary of Puzzles and {run_name} solutions\n{run_desc}",
            1
        ).replace(
            "----",
            f"----\n\nThe puzzles in each module are sorted by number of {run_name} solutions\n\n",
            1
        )

    with open(filename, "w", encoding='utf8') as f:
        f.write(content)


def main(args):
    start_time = time.perf_counter()
    if any(p.DEBUG for p in
           puzzle_generator.PuzzleGenerator.subclass_descendents()):  # if there are debug problems, don't make the dataset
        puzzle_generator.PuzzleGenerator.debug_problems()
        print("Didn't make dataset because there are `Problem.Debug` problems, remove the `.Debug` to proceed.")
        return

    try:
        last_version = utils.load_json(args.json)
        already_tested_cache = {puz["sat"]: {sb for sb in puz["sol_bodies"]} for puz in last_version}
    except:
        already_tested_cache = {}

    utils.info(f"Using {len(already_tested_cache):,} last puzzle testings for speeding up (to avoid re-testing)")

    gens = puzzle_generator.PuzzleGenerator.subclass_descendents()

    gens_by_module = utils.inv_dict({g: g.__module__.replace("generators.", "")+".py" for g in gens})

    utils.info(f"Python version {sys.version}")
    utils.info(f"Generating from templates: {list(gens_by_module)}")
    puzzles = []
    summaries = {}

    for module_name, gens in gens_by_module.items():  # order determined by generators/__init__.py
        readme_examples = []
        downweight = 1.0 / len(gens)  # this causes each module to be weighted equally except for multipliers
        for cls in gens:
            gen = cls()
            gen.build(args.target_num_per_problem, already_tested_cache)
            instances = [
                {
                    "name": i.name,
                    "sat": i.src,
                    "ans_type": gen.ans_type,
                    "sol_header": i.sol_header,
                    "sol_docstring": gen.docstring,
                    "sol_bodies": i.sol_bodies,
                    "module": module_name,
                    "notes": gen.desc,
                    # "weight": gen.multiplier * downweight / len(gen.instances)
                }
                for i in gen.instances]
            puzzles.extend(instances)
            readme_examples.append({
                "name": gen.name,
                "desc": gen.desc,
                "sat": gen.instances[0].src,
                "sol_header": f'{gen.instances[0].sol_header}\n{gen.docstring}',
                "sols": gen.instances[0].sol_bodies,
                "n_instances": len(instances)
            })

        summaries[module_name] = {
            "docstring": inspect.getmodule(gens[0]).__doc__,
            "examples": readme_examples
        }

    utils.save_json(puzzles, args.json, make_dirs_if_necessary=True, indent=2)
    save_readme(summaries, args.readme, args.solutions)
    utils.info(f"Elapsed time: {(time.perf_counter() - start_time) / 60:.2f} minutes")
    utils.info(f"Saved {len(puzzles)} to {args.json} and {args.readme}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
