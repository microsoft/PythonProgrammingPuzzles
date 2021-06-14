import argparse
import fnmatch
import time

import problems
import utils
import templates  # This loads all the problem templates
import inspect

TARGET_NUM_PER_PROBLEM = 1000


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--target_num_per_problem',
                    '-n',
                    type=int,
                    default=TARGET_NUM_PER_PROBLEM,
                    help='Target number of variants to generate per problem.')

parser.add_argument('--templates',
                    '-t',
                    type=str,
                    default='*',
                    help='Glob phrase to select the template files to generate from.')


def main(args):
    start_time = time.perf_counter()
    if problems.Problem.Debug.subclass_descendents(): # if there are debug problems, don't make the dataset
        problems.Problem.debug_problems()
        print("Didn't make dataset because there are `Problem.Debug` problems, remove the `.Debug` to proceed.")
        return

    all_probs = problems.Problem.subclass_descendents()

    probs_by_template = utils.inv_dict({p: p.__module__.split(".")[-1] for p in all_probs})

    used_templates = fnmatch.filter(probs_by_template, args.templates)
    utils.info(f"Generating from templates: {used_templates}")
    problem_sets = []
    for name in used_templates:  # order determined by import order in templates/__init__.py
        probs = probs_by_template[name]
        ps = problems.ProblemSet(name, inspect.getmodule(probs[0]).__doc__)
        for cls in probs:
            problem = cls()
            problem.build(args.target_num_per_problem, ps.get_already_tested())
            ps.add(problem)
        ps.save()
        problem_sets.append(ps)

    problems.save_readme(problem_sets)
    utils.info(f"Elapsed time: {(time.perf_counter()-start_time)/60:.2f} minutes")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
