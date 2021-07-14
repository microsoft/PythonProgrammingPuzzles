import itertools
import time
import numpy as np
from typing import List, Dict, Callable, Tuple, Generator, Set, Sequence
import logging
from utils import prod
import random

from tython import Program, TastNode, _RULES_BY_KIND, RULES, Rule, str2name
from tython.rules import DEF_RULE

logger = logging.getLogger(__name__)


# logger.setLevel("DEBUG")

# %%
########################################################################################################################
# Generation
########################################################################################################################


def compute_uppers(candidates, max_depth):
    time0 = time.time()
    ans = {rule: [0.0] * len(candidates[rule]) for rule in candidates}
    ans.update({rule: [] for c in candidates.values() for _p_nodes, p_rules in c for _p, rule in p_rules
                if not rule.kids})
    prev_sum = None
    for depth in range(max_depth):
        by_rule = {rule: prod(ans[rule]) for rule in ans}
        ans = {rule: [max([0.0] + [p * by_rule.get(r, 0.0) for p, r in p_rules] + [p for p, _node in p_nodes])
                      for p_nodes, p_rules in candidates.get(rule, [])]
               for rule in ans}
        s = sum(x for v in ans.values() for x in v)
        if prev_sum == s:
            break
        prev_sum = s
    logging.debug(f"Computed uppers in {time.time() - time0:.2f}s")
    return ans


def has_free_var(node: TastNode, required_var) -> bool:
    goal = "var:" + required_var
    def helper(n):
        if not isinstance(n, TastNode):
            return False
        return n.rule.name == goal or any(helper(k) for k in n.children)

    return helper(node)


def check_free_vars(node: TastNode, active_vars=set()) -> bool:
    active_vars = active_vars.copy()

    def helper(n):
        nonlocal active_vars
        name = n.rule.name
        if name[:4] == "def:":
            return False
        if name[:4] == "var:":
            return name[4:] in active_vars
        if name != "Comp":
            return all([helper(k) for k in n.children if k.rule.name != "literal"])

        comp_vars = set()
        try:
            for for_in_if in n.children[1:]:
                # zzz
                #assert for_in_if.rule.name == "for_in_if", f"rule.name: {for_in_if.rule.name}"
                if for_in_if.rule.name != "for_in_if":
                    return False
                target, iter, *ifs = for_in_if.children
                d, v_name = target.rule.name[:4], target.rule.name[4:]
                if not helper(iter) or d != "def:" or v_name in active_vars:
                    return False
                active_vars.add(v_name)
                comp_vars.add(v_name)
                if not all(map(helper, ifs)):
                    return False
            return helper(n.children[0])
        finally:
            active_vars -= comp_vars

    return helper(node)


def check_gen_progs(p_nodes: List[Tuple[float, TastNode]]) -> List[Tuple[float, Program]]:
    #return [(p, Program(node)) for p, node in p_nodes] # zzz
    """Check that comprehension variables are correctly used and sort programs"""
    n = len(p_nodes)
    #p_nodes = [p_node for p_node in p_nodes if check_free_vars(p_node[1])]
    if n == 0:
        logger.debug("zero nodes in this round")
    else:
        logger.debug(f"filtered {n-len(p_nodes)}/{n} = {1-len(p_nodes)/n:.1%} of the nodes, leaving {len(p_nodes):,}")
    p_nodes.sort(key=lambda p_node: -p_node[0])
    # print(f"Generated {len(ans):,} programs_up_to_threshold(min_prob={min_prob:.2}) of depth<{max_depth} in {(time.time() - time0):.1f} seconds")
    return [(p, Program(node)) for p, node in p_nodes]


def programs_up_to_threshold(min_prob, candidates, uppers, stop_time, max_depth):
    def helper(min_prob, parent_rule, child_num, max_depth, comp: bool):
        if max_depth < 0 or uppers[parent_rule][child_num] < min_prob:
            return []
        if stop_time and time.time() > stop_time:
            raise TimeoutError
        p_nodes, p_rules = candidates[parent_rule][child_num]
        ans = [z for z in p_nodes if z[0] >= min_prob]
        for p, r in p_rules:
            if r.name == "literal" or (not comp and r.name.startswith("var:")):
                continue
            max_ps = uppers[r]
            best = p * prod(max_ps)
            if best < min_prob:
                continue
            for x in itertools.product(*[helper(min_prob / best * max_ps[i],
                                                r, i, max_depth - 1, comp or r.name == "Comp")
                                         for i in range(len(r.kids))]):
                new_prob = p * prod([p2 for p2, n in x])
                if new_prob >= min_prob:
                    ans.append((new_prob, TastNode(r, [n for p, n in x])))
                if stop_time and time.time() > stop_time:
                    raise TimeoutError
        return ans

    return check_gen_progs(helper(min_prob, DEF_RULE, 2, max_depth, False))


# simple implementation:
def programs_up_to_threshold_wo_uppers(min_prob, candidates, stop_time=None, max_depth=10):
    def helper(min_prob, parent_rule, child_num, max_depth, comp: bool):
        if max_depth <= 0:
            return []
        if stop_time and time.time() > stop_time:
            raise TimeoutError
        p_trees, p_rules = [[x for x in c if x[0] >= min_prob] for c in candidates[parent_rule][child_num]]
        # old version:
        #    p_x_r = [(p * prod([p2 for p2, n in x]), x, r) for p, r in p_rules for x in
        #             itertools.product(*[helper(min_prob / p, r, i, max_depth-1) for i in range(len(r.children))])]
        #    return [(p, TastNode(r, [n for p, n in x])) for p, x, r in p_x_r if p >= min_prob] + p_trees
        ans = p_trees
        for p, r in p_rules:
            if r.name == "literal" or (not comp and r.name.startswith("var:")):
                continue
            min_prob2 = min_prob / p
            children_candidates = []
            for i in range(len(r.kids)):
                h = helper(min_prob2, r, i, max_depth - 1, comp or r.name == "Comp")
                if not h:
                    break
                min_prob2 /= max([pp for pp, _ in h])
                children_candidates.append(h)
            if len(children_candidates) == len(r.kids):
                for x in itertools.product(*children_candidates):
                    p2 = p * prod([pp for pp, _ in x])
                    if p2 >= min_prob:
                        ans.append((p2, TastNode(r, [n for p, n in x])))
        return ans

    return check_gen_progs(helper(min_prob, DEF_RULE, 2, max_depth, False))  # DEF_RULE child-2 generates a sol body


def rand_node(candidates, free_var=None, max_depth=10, top_p=0.95, rand_func=random.random, max_attempts=10 ** 4):
    """

    :param candidates:
    :param free_var: a variable from tython.VAR_KINDS that must be included as free variables
    :param max_depth:
    :param top_p: top_p=1 is uniform sampling; otherwise, for each node consider most likley choices with tot prob top_p
    :param rand_func: a function that selects a random number between 0 and 1
    :param max_attempts:
    :return: a random node
    """

    if top_p is not None and top_p >= 1.0:
        top_p = None

    class Fail(Exception):
        pass

    def helper(parent_rule, child_num, max_depth, comp: bool):
        if max_depth <= 0:
            logger.debug("max depth <= 0")
            raise Fail
        p_trees, p_rules = [[x for x in c if x[0] > 0.0] for c in candidates[parent_rule][child_num]]
        if not comp:
            p_rules = [(p, rule) for (p, rule) in p_rules
                       if not (rule.name.startswith("var:")
                               or rule.name == 'var'
                               or rule.name == 'COPY'
                               or rule.name == 'const'  # should use type specific consts.
                              )
                          or rule.name[4:] == free_var]

        p_node_or_rules = p_trees + p_rules
        tot = sum([p for p, n in p_node_or_rules])
        if tot <= 0.0:  # should never be negative
            logger.debug("tot <= 0")
            raise Fail
        r = rand_func() * tot
        if top_p is not None:
            r *= top_p
            p_node_or_rules.sort(key=lambda x: -x[0])
        for p, nr in p_node_or_rules:
            r -= p
            if r < 0:
                break
        # nr is selection
        if isinstance(nr, TastNode):
            return nr
        assert isinstance(nr, Rule)
        if nr.name == "literal":
            logger.debug("nr.name == literal")
            raise Fail
        comp = comp or nr.name == "Comp"

        return TastNode(nr, [helper(nr, i, max_depth - 1, comp) for i in range(len(nr.kids))])

    for _ in range(max_attempts):
        try:
            node = helper(None, 0, max_depth, False)
            if free_var:
                if check_free_vars(node, {free_var}) and has_free_var(node, free_var):
                    return node
            else:
                if check_free_vars(node):
                    return node
        except Fail:
            pass

    assert False, f"Failed to generate a program in {max_attempts:,} attempts"


# same as above but with less list comprehensions.
# def programs_up_to_threshold_wo_uppers(min_prob, candidates, stop_time=None, max_depth=10):
#     '''
#     Get all program trees with probability greater than min_prob.
#     '''
#     if None not in candidates: # cannot generate this kind
#         return []
#
#     def helper(min_prob, parent_rule, child_num, max_depth):
#         if max_depth < 0:
#             return []
#         if stop_time and time.time()>stop_time:
#             raise TimeoutError
#
#         p_consts, p_rules = [], []
#         # Early pruning.
#         for i, cand_list in [(0, p_consts), (1, p_rules)]:
#             if parent_rule is not None and parent_rule.name == 'literal':
#                 continue
#             for cand in candidates[parent_rule][child_num][i]:
#                 prob = cand[0]
#                 if prob >= min_prob:
#                     cand_list.append(cand)
#
#         p_t_r = []
#         for prob, rule in p_rules:
#             # Cartesian product (all possible sub-trees)
#             for tree in itertools.product(*[
#                     helper(min_prob / prob, rule, i, max_depth - 1)
#                     for i in range(len(rule.children))
#             ]):
#                 joint_prob = prob * prod([p for p, _ in tree])
#                 # Keep only trees above threshold.
#                 if joint_prob >= min_prob:
#                     p_t_r.append((joint_prob, TastNode(rule, [child for _, child in tree])))
#
#         out_list = p_t_r + p_consts
#         return out_list
#
#     trees_above_prob = helper(min_prob, parent_rule=None, child_num=0, max_depth=max_depth)
#     return [(prob, Program(node))
#             for prob, node in sorted(trees_above_prob, key=lambda z: -z[0])]


# def most_likely_n_progs(candidates, num_progs, stop_time=None, uppers_thresh=100):
#     """
#     Generates a list of num_prog (prob, node)'s in decreasing order based on candidates
#
#     :param candidates: function that maps (kind, parent_rule) to a pair of:
#                         1) list of (prob, TastNode) e.g. for constant nodes
#                         2) list of (prob, rule)
#     :param num_progs: how many programs do you want? (careful what you wishfor)
#     :param stop_time: time to stop if haven't stoped earlier.
#     :return: list of num_prog (prob, node)'s in decreasing probabilities
#     """
#     min_prob = 0.1 / num_progs
#     if num_progs >= 100:
#         # The upper bound computation takes some time so if num_progs is small
#         # it's better without...
#         uppers = compute_uppers(candidates, max_depth)
#     ans = []
#     while True:
#         try:
#             if num_progs >= 100:
#                 ans = programs_up_to_threshold(min_prob, candidates, uppers, stop_time, max_depth)
#             else:
#                 ans = programs_up_to_threshold_wo_uppers(min_prob, candidates, stop_time, max_depth)
#         except TimeoutError:
#             return ans
#         if ans and len(ans) >= num_progs:
#             return ans[:num_progs]
#         estimate = len(ans) / num_progs
#         if estimate < 0.05:
#             min_prob *= 0.1
#         else:
#             min_prob *= estimate * 2 / 3


def generate_progs(candidates, stop_time=None, max_depth=10, min_prob=1e-3, uppers_thresh=20):
    """
    A generator for programs of a certain kind in decreasing likelihood

    :param candidates: function that maps (kind, parent_rule) to a pair of:
                        1) list of (prob, TastNode) e.g. for constant nodes
                        2) list of (prob, rule)
    :return: it's a generator so it's like an infinite list. If you waant a specific n use most_most_likely_num_progs
    """
    n = 0
    uppers = None
    while min_prob > 0:
        try:
            tick = time.time()
            if uppers is None:
                batch = programs_up_to_threshold_wo_uppers(min_prob, candidates, stop_time, max_depth)
            else:
                batch = programs_up_to_threshold(min_prob, candidates, uppers, stop_time, max_depth)
        except TimeoutError:
            return
        assert len(batch) >= n
        for ans in batch[n:]:
            if stop_time and time.time() > stop_time:
                return
            yield ans
        n = len(batch)
        if uppers is None and n > uppers_thresh:
            uppers = compute_uppers(candidates, max_depth)
        min_prob /= np.e  # optimal ratio computed using math and expected values :-)
    # print(f"*** exhausted all {n:,} programs of depth<{max_depth}")


def map_candidates_rules(candidates: Dict) -> Dict:
    '''
    Given a candidate dict with indices of rules, convert the indices to rule objects
    and return the new dict.
    '''
    out = {}
    for rule_ind in candidates.keys():
        rule = RULES[rule_ind] if rule_ind < len(RULES) else None
        out[rule] = []
        for child in candidates[rule_ind]:
            cops = child[0]
            rule_probs = [(prob, RULES[ind]) for prob, ind in child[1]]
            out[rule].append((cops, rule_probs))

    return out


def test_sol(q_src, a_src):
    env = dict(List=List, Dict=Dict, Set=Set)
    exec(a_src + "\n" + "answer = sol()", env)

    answer = env["answer"]
    # assert answer == decode(encode(env["answer"])), "encode/decode round trip failed"

    env2 = dict(answer=answer, List=List, Dict=Dict, Set=Set)  # in case they mucked with env
    exec(q_src + "\n" + "assert sat(answer)", env2, description=self.name)
    self.sol_srcs.append(sol_src)


def solve(q, a_kind, candidates, n_progs=None, timeout_secs=10, verbose=False, max_ticks=2000, rule_inds=False):
    if rule_inds:
        candidates = map_candidates_rules(candidates)
    q_py = q.src(safe=False, simplify=False)

    exceptions1 = {}
    exceptions2 = {}
    f = q.run(max_ticks=max_ticks)['sat']
    time0 = time.time()
    stop_time = (time0 + timeout_secs) if timeout_secs else None
    if verbose:
        print(f'*** Question: {q_py}')
    count = 0
    a = None
    args_node = q.tree.children[0].children[1].children[-1]

    for p, a_body in generate_progs(candidates, stop_time):
        count += 1
        if count == n_progs:
            break
        a = Program(TastNode(DEF_RULE, [str2name("sol"), args_node, a_body.tree]))

        a_safe = a.src(simplify=False)
        a_py = a.src(safe=False, simplify=False)
        try:
            x = a.run(max_ticks=max_ticks / 2)["sol"]()
            q.reset_clock()
            try:
                v = f(x)
                assert isinstance(v, bool)
                if v == True:
                    return a, count
            except Exception as e:
                if verbose and str(e) not in exceptions2:
                    print(f"exception2 {e}: '{a_py}'")
                    exceptions2.setdefault(str(e), []).append((a_py, e, q_py))
        except Exception as e:
            if verbose and str(e) not in exceptions1:
                print(f"exception1 {e}: '{a_py}'")
                exceptions1.setdefault(str(e), []).append((a_py, e, q_py))
    if verbose:
        print(
            f"{(time.time() - time0) / 60:.1f} mins for {count} programs on '{q_py}', ans='{a.src(safe=False) if a else None}'",
            flush=True)
    return None, count


#
#
# failures, successes = [], []
# for q, k in Qkinds:
#     a = solve(q, k, model=UniformModel().candidates(q))
#     if a:
#         successes.append((q, a))
#     else:
#         failures.append(q)
#
# print(f'{len(failures) / len(Qkinds):.1%} failures')
# print(f'{np.mean([bool(s) for q, s in successes]):.1%} successes')


def test():
    from models import uniform
    u = uniform.UniformModel(copy_prob=0.01)
    candidates = u.get_candidates_by_nodes(bool, {})
    for _ in range(1000):
        print("random program", Program(rand_node(candidates, "x")))

    tests = []
    tests.append(('x["cat"]=="dog"', (Dict, str, str)))

    tests += [(f"len(x)=={i} and x[0]=='{c}'", str) for i in [1, 2, 9] for c in "abcz"]
    tests += [(f"len(x)=={i} and x[0]=={j}", (List, int)) for i in [1, 2, 9] for j in [0, 4, 17]]
    test_progs = [Program(q_py, {"x": k}) for q_py, k in tests]
    count = 0
    n = 100
    print(f"Most likely {n:,} integer programs:")
    print([g.src(safe=False, simplify=True) for p, g in
           most_likely_n_progs(UniformModel().candidates(test_progs[0]), n)])


if __name__ == "__main__":
    test()
