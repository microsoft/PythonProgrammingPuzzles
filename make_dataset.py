import argparse
import fnmatch
import time

import problems
import utils
import templates  # This loads all the problem templates

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
    templates = fnmatch.filter(problems.problem_registry.keys(), args.templates)
    utils.info(f"Generating from templates: {templates}")
    problem_sets = []
    for name in templates:  # order determined by import order in templates/__init__.py
        entry = problems.problem_registry[name]
        ps = problems.ProblemSet(entry["name"], entry["summary"])
        for cls in entry["classes"]:
            problem = cls()
            problem.build(args.target_num_per_problem)
            ps.add(problem)
        ps.save()
        problem_sets.append(ps)

    problems.save_readme(problem_sets)
    utils.info(f"Elapsed time: {(time.perf_counter()-start_time)/60:.2f} minutes")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
