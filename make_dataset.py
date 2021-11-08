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
                    nargs='*',
                    default='',
                    help='Filename(s) of AI solutions to add to README, if any.')

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
    if orig == new:
        return

    def helper(s):
        return re.sub(r'\b' + orig + r'\b(?!["\'])', new, s, count)

    if src.count('"""') >= 2:
        a = src.index('"""')
        b = src.index('"""', a + 1) + 3
        if count == 1:
            h = helper(src[:a])
            if h != src[:a]:
                return h + src[a:]
        return helper(src[:a]) + src[a:b] + helper(src[b:])

    return helper(src)


def anchor(name):
    return name.strip().lower().replace(' ', '-')


def indent(x, spaces=4):
    return (" " * spaces + x.replace("\n", "\n" + " " * spaces))[:-spaces if x.endswith("\n") else None]


def save_readme(gen_modules, filename, sol_filenames):
    ai_sols = {}
    for sol_filename in sol_filenames:
        sols_js = utils.load_json(sol_filename)
        for experiment in sols_js:
            if "short" in experiment["experiment"]:
                continue # skip short experiments
            for s_s in experiment["sat_sols"]:
                f = s_s["sat"]
                assert f.startswith("def ")
                f_name = f[len("def "):f.index("(")].strip()
                f2 = rename_src_var(f_name, "sat", f, 1)
                if f2 not in ai_sols:
                    ai_sols[f2] = dict(n_sols=0, n_attempts=0, sols=set())
                if "failures" in s_s: # bootstrap
                    ai_sols[f2]["n_attempts"] += s_s["failures"]
                    if s_s.get("sol"):
                        ai_sols[f2]["n_sols"] += 1
                        ai_sols[f2]["n_attempts"] += 1
                    cur_sols = [s_s.get("sol")]
                else:
                    ai_sols[f2]["n_attempts"] += experiment["n"]
                    ai_sols[f2]["n_sols"] += s_s["n_sols"]
                    cur_sols = [s_s[k] for k in s_s if k.endswith("_sol")]
                    if "example_sols" in s_s:
                        cur_sols += s_s["example_sols"]
                ai_sols[f2]["sols"].update(rename_src_var(f_name, "sat", sol) for sol in cur_sols if sol)

    for entry in ai_sols.values():
        entry["sol"] = (min(entry["sols"], key=len) if entry["sols"] else "")
        entry['longest_sol'] = (max(entry["sols"], key=len) if len(entry["sols"]) > 1 else "")
        entry["success_rate"] = entry["n_sols"]/entry["n_attempts"]

    table = ""
    content = ""
    tot_puzzles = 0
    tot_instances = 0

    def py(src):
        return f"```python\n{src}\n```\n"

    for module_name, module_stuff in gen_modules.items():
        section = ""
        sec_name = module_name.split(".")[0]
        section += f"## {sec_name}\n\n"
        section += f"{module_stuff['docstring']}\n\n"
        puzzles = module_stuff['examples']
        if ai_sols:
            puzzles = sorted(puzzles,
                             key=lambda f: ai_sols[f['sat']]["success_rate"] if f['sat'] in ai_sols else 0,
                             reverse=True)
        n = len(puzzles)
        link = f"[{sec_name}](#{anchor(sec_name)})"
        n_instances = sum(p["n_instances"] for p in puzzles)
        tot_instances += n_instances
        table += f"- [{module_name}](../generators/{module_name}), [{len(puzzles):,} problems](#{anchor(sec_name)}): {module_stuff['docstring'].strip().rstrip('.')}\n"
        for i, puzzle in enumerate(puzzles):
            tot_puzzles += 1
            f = puzzle['sat']
            puzzle_text = f'* <a name="{anchor(puzzle["name"])}"></a>**{puzzle["name"]}** {puzzle["notes"].strip()}'
            puzzle_text += f' ({puzzle["n_instances"]} instance{"s" if puzzle["n_instances"] != 1 else ""})\n\n'
            puzzle_text += py(f)
            puzzle_text += "<details><summary>"
            num_ai_sols = 0
            if ai_sols:
                sol_entry = ai_sols.get(f)
                if sol_entry:
                    num_ai_sols = sol_entry['n_sols']
                    puzzle_text += f"{sol_entry['success_rate']*100:.2g}% Codex success rate, "
                else:
                    puzzle_text += f"Codex was not run on this puzzle, "
            sol_bodies = puzzle['sol_bodies']
            n_sols = len(sol_bodies)
            puzzle_text += f"{n_sols:,} hand-written solution{'s' if n_sols != 1 else ''} "
            puzzle_text += "</summary>\n\n"
            puzzle_text += "Solution header:\n"
            puzzle_text += py(puzzle['sol_header'])
            puzzle_text += "Solution docstring (*not* usually provided)\n\n"
            puzzle_text += py(puzzle['sol_docstring'])
            if num_ai_sols:
                if sol_entry['longest_sol']:
                    puzzle_text += f"Shortest Codex solution:\n"
                    puzzle_text += py(sol_entry['sol'])
                    puzzle_text += f"Longest Codex solution:\n"
                    puzzle_text += py(sol_entry['longest_sol'])
                else:
                    puzzle_text += f"Codex solution:\n"
                    puzzle_text += py(sol_entry['sol'])
            if n_sols:
                for body in sol_bodies:
                    puzzle_text += "Hand-written solution:\n"
                    puzzle_text += py(body)

            puzzle_text += "</details>\n\n"

            section += indent(puzzle_text, 4)[4:]

        content += section

    table += f"\nTotal ({tot_puzzles:,} problems, {tot_instances:,} instances)\n"

    content = TOP.format(table) + content
    if ai_sols:
        content = content.replace(
            "Summary of Puzzles",
            f"Summary of Puzzles and Codex solutions",
            1
        ).replace(
            "----",
            f"----\n\nThe puzzles in each module are sorted by percent of Codex correct solutions\n\n",
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

    gens_by_module = utils.inv_dict({g: g.__module__.replace("generators.", "") + ".py" for g in gens})

    utils.info(f"Python version {sys.version}")
    utils.info(f"Generating from templates: {list(gens_by_module)}")
    puzzles = []
    summaries = {}

    for module_name, gens in gens_by_module.items():  # order determined by generators/__init__.py
        module_multiplier = 0.2 if "trivial" in module_name else 1.0
        readme_examples = []
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
                    "weight": i.multiplier * module_multiplier
                }
                for i in gen.instances]
            puzzles.extend(instances)
            readme_examples.append({** instances[0], "name": gen.name, "n_instances": len(instances)})

        summaries[module_name] = {
            "docstring": inspect.getmodule(gens[0]).__doc__,
            "examples": readme_examples
        }

    utils.save_json(puzzles, args.json, make_dirs_if_necessary=True, indent=2)
    save_readme(summaries, args.readme, args.solutions)
    utils.info(f"Elapsed time: {(time.perf_counter() - start_time) / 60:.2f} minutes")
    utils.info(f"Saved {len(puzzles):,} to {args.json} and {args.readme}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
