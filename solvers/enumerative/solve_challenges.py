from typing import List, Set, Dict, Callable, Tuple
from random import randint, randrange
from math import cos, sin, pi, log, exp, log2
from copy import deepcopy
from tqdm import tqdm
import random
import json
import argparse
import numpy as np
import time
import multiprocessing as mp
import functools
import logging
import glob
import os
from datetime import datetime

from models import MODEL_REGISTRY
from challenges import Challenge, Solution, SolverSolution, verify_solutions
import tython
import top_down
import utils


logger = logging.getLogger(__name__)

DEFAULT_CHALLENGES_PATH = "data/all.json"

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def boolean_string(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True' or s == 'true'

parser.add_argument('--challenges_path',
                    '-p',
                    default=DEFAULT_CHALLENGES_PATH,
                    help='Path pattern for challenges.')

parser.add_argument('--all_instances',
                    default=False,
                    action='store_true',
                    help='Use all instances from the json file (Otherwise takes only names ending with _0).')

parser.add_argument('--eval_challenges_path',
                    default='',
                    help='Path pattern for eval challenges.')

# Model params.
parser.add_argument('--model_name', '-m',
                    default='unigram',
                    type=str,
                    help="Which model to use: %s" % list(MODEL_REGISTRY.keys()))

parser.add_argument('--ml_model', '-ml',
                    default='rf',
                    type=str,
                    help="ml model (use with --model that starts with ml_).")

parser.add_argument('--copy_prob',
                    default=0.5,
                    type=float,
                    help="Probability to copy constants from question.")
# Logging params.
parser.add_argument('--logging_dir',
                    default='',
                    type=str,
                    help="Where to store log (default: don't store)")

parser.add_argument('--logging_level',
                    default='INFO',
                    type=str,
                    help="Logging level [INFO, DEBUG]")

parser.add_argument('--out_file',
                    default='out.json',
                    type=str,
                    help="store solutions of uniform")

parser.add_argument('--logging_name',
                    default='',
                    type=str,
                    help="model name for logging to xl file")

# Solver params.
parser.add_argument('--verify_sols',
                    default=False,
                    action='store_true',
                    help='Verify the solutions in the input file.')

parser.add_argument('--learn_from_train_sols',
                    default=False,
                    action='store_true',
                    help='Learn from the provided solutions.')

parser.add_argument('--solve_uniform',
                    default=False,
                    action='store_true',
                    help='Try to solve the challenges first with uniform model.')

parser.add_argument('--resolve',
                    default=False,
                    action='store_true',
                    help='Try to solve the challenges again using the solved solutions from the first round.')

parser.add_argument('--timeout_secs',
                    default=None,
                    type=int,
                    help='Max seconds to wait for solver.')

parser.add_argument('--max_n_progs',
                    default=None,
                    type=int,
                    help='Max programs to try.')

parser.add_argument('--seed',
                    default=42,
                    type=int,
                    help='random seed.')
# Other params.
parser.add_argument('--threads', default=10, type=int, help='Threads.')


def solve_challenge(params, timeout_secs=20, max_ticks=10000):
    name, f_str, sol_kind = params
    prog = tython.Program(f_str)
    st_time = time.time()
    ans, count = top_down.solve(
        prog,
        sol_kind,
        model.get_candidates(prog),
        timeout_secs=timeout_secs,
        n_progs=max_n_progs,
        max_ticks=max_ticks,
    )
    if ans is not None:
        a_py = ans.src(safe=False, simplify=False)
    else:
        a_py = None
    return name, a_py, time.time() - st_time, count


def _solve_challenge_init_fn(_model, _max_n_progs):
    global model
    global max_n_progs
    model = deepcopy(_model)
    max_n_progs = _max_n_progs


def solve_challenges(challenges,
                     model,
                     n_progs=None,
                     timeout_secs=20,
                     out_with_gold=True,
                     threads=0):
    '''
    Tries to solve the challenges.
    '''
    solved = []
    unsolved = []
    params = [(name, ch.f_str, ch.sol_kind)
              for name, ch in challenges.items()]
    if threads == 0:
        for ch in tqdm(challenges.values(), total=len(challenges.keys()), desc='Solving'):
            logger.debug(f"Trying to solve '{ch.f_str}'...")
            st_time = time.time()
            ans, count = top_down.solve(ch.prog,
                                 ch.sol_kind,
                                 model.get_candidates(ch.prog),
                                 timeout_secs=timeout_secs,
                                 n_progs=n_progs)
            duration = time.time() - st_time
            if ans is None:
                logger.debug(f"Failed to solve, generated {count:,} programs in {duration:.2f}s.")
                unsolved.append(ch)
            else:
                logger.debug("Got answer in {:.2f}s and {:,} programs: {}".format(duration, count, ans))
                solved.append(ch)
            ch.solver_solutions.append(
                SolverSolution(string="?" if ans is None else str(ans),
                               prog=ans,
                               time=duration,
                               count=count))

    else:
        workers = mp.Pool(processes=threads, initializer=_solve_challenge_init_fn, initargs=(model, args.max_n_progs,)) # zzz
        map_fn = workers.imap_unordered
        worker_fn = functools.partial(solve_challenge,
                                      timeout_secs=timeout_secs)
        with tqdm(total=len(challenges.keys()), desc="Solving") as pbar:
            for name, ans, duration, count in map_fn(worker_fn, params):
                ch = challenges[name]
                ch.solver_solutions.append(
                    SolverSolution(string="" if ans is None else str(ans),
                                   prog=ans,
                                   time=duration,
                                   count=count))
                if ans is not None:
                    logger.debug(f"Got answer for '{ch.f_str}' in {duration:.2f}s and {count:,} programs: '{ans}'")
                    solved.append(ch)
                else:
                    logger.debug(f"Failed to solve '{ch.f_str}', generated {count:,} programs in {duration:.2f}s.")
                    unsolved.append(ch)
                pbar.update()

    for (success, li) in [("Solved", solved), ("Unsolved", unsolved)]:
        logger.info(f"--- {success}: (name | challenge | solution | time | count)")
        for ch in li:
            logger.info("'{}' | '{}' | '{}' | ({:.2f}s, {})".format(
                ch.name, ch.f_str, ch.solver_solutions[-1].string,
                ch.solver_solutions[-1].time, ch.solver_solutions[-1].count))

    solved_configs = []
    for name, f_str, sol_kind in params:
        if out_with_gold:
            out = dict(name=name, sat=f_str,
                       sols=[x.string for x in challenges[name].gold_solutions] + [x.string for x in challenges[name].solver_solutions],
                       sol_time=[x.time for x in challenges[name].gold_solutions] + [x.time for x in challenges[name].solver_solutions],
                       sol_tries=[x.count for x in challenges[name].gold_solutions] + [x.count for x in challenges[name].solver_solutions])
        else:
            out = dict(name=name, sat=f_str,
                       sols=[x.string for x in challenges[name].solver_solutions],
                       sol_time=[x.time for x in challenges[name].solver_solutions],
                       sol_tries=[x.count for x in challenges[name].solver_solutions])
        solved_configs.append(out)

    log_solved_percent(challenges)
    return solved_configs


def log_solved_percent(challenges):
    sols = [ch.solver_solutions[-1] for ch in challenges.values()]
    success_rate = np.mean([s.prog is not None for s in sols])
    avg_duration = np.mean([s.time for s in sols])
    logger.info(f"Solved {success_rate:.1%} of challenges using {avg_duration:.1f}secs on avg.")


if __name__ == "__main__":
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    assert not (args.solve_uniform and args.learn_from_train_sols)
    if args.resolve:
        assert args.solve_uniform or args.learn_from_train_sols

    assert args.timeout_secs or args.max_n_progs


    if args.logging_dir:
        os.makedirs(args.logging_dir, exist_ok=True)
        # make sure filename doesn't have * in it:
        fname = os.path.splitext(os.path.split(args.challenges_path)[-1])[0].replace("*", "STAR")
        fname += datetime.now().strftime("_%d_%m_%Y_%H_%M_%S.log")
        fh = logging.FileHandler(os.path.join(args.logging_dir, fname))
        logger.addHandler(fh)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(args.logging_level.upper())

    for k, v in args.__dict__.items():
        logger.info('{}: {}'.format(k, v))

    challenges_files = glob.glob(args.challenges_path)
    logger.info(f"Loading challenges from {challenges_files}")
    challenge_configs = []
    seen_challenges = set()
    for f_name in challenges_files:
        chs = json.load(open(f_name, 'r'))
        for ch in chs:
            if not args.all_instances and not ch['name'].endswith('_0'):
                continue
            if ch['name'] not in seen_challenges:
                challenge_configs.append(ch)
                seen_challenges.add(ch['name'])
            seen_challenges.add(ch['name'])

    # Parse challenges.
    challenges = {}
    for config in challenge_configs:
        ch = Challenge(config)
        if ch.prog is not None:
            challenges[ch.name] = ch

    logger.info("Successfully parsed {} ({:.2f}%) out of {} challenges.".format(
        len(challenges.keys()),
        100 * len(challenges.keys()) / len(challenge_configs),
        len(challenge_configs)))

    if args.model_name not in MODEL_REGISTRY:
        logger.error("%s not in model options: %s", args.model_name, list(MODEL_REGISTRY.keys()))
    
    if args.solve_uniform:
        model = MODEL_REGISTRY['uniform'](copy_prob=args.copy_prob)
        out_configs = solve_challenges(challenges,
                             model=model,
                             timeout_secs=args.timeout_secs,
                             n_progs=args.max_n_progs,
                             out_with_gold=False,
                             threads=args.threads)

        with open(args.out_file, 'w') as fw:
            json.dump(out_configs, fw, indent=4)

    #model_kwargs = dict(copy_prob=args.copy_prob)
    model_kwargs = {}
    if args.model_name.startswith("ml_"):
        model_kwargs["ml_model"] = args.ml_model

    model = MODEL_REGISTRY[args.model_name](**model_kwargs)

    if args.learn_from_train_sols:
        verify_solutions(challenges.values())
        #sol_list = [(ch.prog, s.prog) for ch in challenges.values()
        #            for s in ch.gold_solutions if s.prog is not None]

        # only last solutions:
        sol_list = []
        for ch in challenges.values():
            added_one_sol = False
            if len(ch.gold_solutions) == 0:
                continue
            for s in reversed(ch.gold_solutions):
                if not added_one_sol and s.prog is not None:
                    sol_list.append((ch.prog, s.prog))
                    added_one_sol = True

        #sol_list = [(ch.prog, s.prog) for ch in challenges.values() if len(ch.gold_solutions)>0
        #            for s in [ch.gold_solutions[-1]] if s.prog is not None]

        logger.info("Learning from provided {} solutions".format(len(sol_list)))
        model.learn(sol_list)

        #out_configs = solve_challenges(challenges,
        #                 model=model,
        #                 timeout_secs=args.timeout_secs,
        #                 n_progs=args.max_n_progs,
        #                 threads=args.threads)

        #with open(args.out_file, 'w') as fw:
        #    json.dump(out_configs, fw, indent=4)

    if args.resolve:
        sol_list = [(ch.prog, a.prog) for ch in challenges.values()
                    for a in ch.solver_solutions if a.prog is not None]
        logger.info("Learning from solved {} solutions".format(len(sol_list)))
        model.learn(sol_list)
        solve_challenges(challenges,
                         model=model,
                         timeout_secs=args.timeout_secs,
                         n_progs=args.max_n_progs,
                         threads=args.threads)

    if args.eval_challenges_path:
        eval_challenges_files = glob.glob(args.eval_challenges_path)
        logger.info(f"Loading evaluation challenges from {eval_challenges_files}")
        challenge_configs = []
        seen_challenges = set()
        for f_name in eval_challenges_files:
            chs = json.load(open(f_name, 'r'))
            for ch in chs:
                if not args.all_instances and not ch['name'].endswith('_0'):
                    continue
                if ch['name'] not in seen_challenges:
                    challenge_configs.append(ch)
                    seen_challenges.add(ch['name'])
                seen_challenges.add(ch['name'])

        # Parse challenges.
        eval_challenges = {}
        for config in challenge_configs:
            ch = Challenge(config, max_ticks=10000)
            if ch.prog is not None:
                eval_challenges[ch.name] = ch

        logger.info("Successfully parsed {} ({:.2f}%) out of {} eval_challenges.".format(
            len(eval_challenges.keys()),
            100 * len(eval_challenges.keys()) / len(challenge_configs),
            len(challenge_configs)))

        out_configs = solve_challenges(eval_challenges,
                         model=model,
                         timeout_secs=args.timeout_secs,
                         n_progs=args.max_n_progs,
                         threads=args.threads)

        with open(args.out_file, 'w') as fw:
            json.dump(out_configs, fw, indent=4)

    if args.verify_sols:
        verify_solutions(challenges.values())
        if hasattr(model, 'get_likelihood'):
            sol_list = [(ch, sol) for ch in challenges.values()
                        for sol in ch.gold_solutions if sol.prog is not None]
            logger.info("--- Likelihoods of gold solutions: (name | challenge | solution | probability)")
            for ch, sol in sol_list:
                prob = model.get_likelihood(ch.prog, sol.prog)
                sol.Likelihoods = prob
                logger.info("{} | {} | {} | {:.4f})".format(
                    ch.name, ch.f_str, sol.string, prob))

    if args.logging_name:
        from tools import spreadsheet
        results = {}
        for ch in challenges.values():
            if len(ch.solver_counts) > 0:
                results[ch.name] = ch.solver_counts[-1]
            else:
                results[ch.name] = -1

        results['model'] = args.logging_name
        spreadsheet.log_results(results)

