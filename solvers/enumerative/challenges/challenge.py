import utils
import tython
import logging
from typing import List, Dict, Callable, Tuple, Generator, Set, Sequence
from tython import Program, nt

logger = logging.getLogger(__name__)


def extract_constants(prog) -> Dict:
    '''
    Extract all constants from program. Does not (yet) allow copying of comprehensions, e.g., '[i*i for i in range(10)]'
    '''

    from collections import defaultdict
    consts = defaultdict(list)

    def handle_args(args_node):

        if args_node.rule.name == 'cast:ARGS':
            handle_args(args_node.children[0])
        else:
            if len(args_node.children) >= 3 and args_node.children[1].nt == nt.TYPE:
                annotation_node = args_node.children[1]
                t = nt.type2nt(eval(annotation_node.src()))
                consts[t].append(args_node.children[0])
            if args_node.children and args_node.children[-1].nt in {nt.ARGS, nt.DEFAULT_ARGS}:
                handle_args(args_node.children[-1])

    def helper(node):
        if node.rule.name == 'def':  # it's a function
            name_node, args_node, body_node = node.children
            if name_node.src() == 'sat':
                handle_args(args_node.children[-1])  # skip first arg for `def sat`
            else:
                handle_args(args_node)
            helper(body_node)
            return False
        elif node.nt in {nt.NAME}:
            return False
        elif node.nt in {nt.STMT}:
            for c in node.children:
                helper(c)
            return False
        if node.rule.name not in {"int-const", "str-const"} and not all([helper(c) for c in node.children]):
            return False
        if node.nt.isa(nt.LIST, nt.SET, nt.DICT, nt.TUPLE, nt.RANGE,
                       nt.INT, nt.FLOAT, nt.BOOL, nt.STR):
            consts[node.nt].append(node)
        return True

    if prog is not None:
        helper(prog.tree)

    return dict(consts)

#
# q = Program("""
# def sat(i: List[str], a=5):
#     return i==['5']
# """)
#
# extract_constants(q)
#
#
# %%
class Solution():
    def __init__(self, string=None, prog=None, likelihood=None, time=None, count=None):
        self.string = string
        self.prog = prog
        self.likelihood = likelihood
        self.time = time
        self.count = count


class SolverSolution(Solution):
    def __init__(self, string=None, prog=None, likelihood=None, time=None, count=None):
        super().__init__(string=string, prog=prog, likelihood=likelihood)
        self.time = time
        self.count = count


def get_arg_type_str(sat_str):
    assert sat_str.startswith("def sat(") and ":" in sat_str
    depth = 0
    for i, c in enumerate(sat_str):
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
        elif c in ")," and depth == 0:
            return sat_str[sat_str.index(":") + 1:i].lstrip()
    assert False


class Challenge():
    def __init__(self, challenge_config, max_ticks=100000000):
        self.name = challenge_config["name"]
        self.f_str = challenge_config["sat"]
        self.type_str = get_arg_type_str(challenge_config["sat"])
        self.type = eval(self.type_str)
        self.gold_solutions = []
        self.solver_solutions = []
        for sol in challenge_config["sols"]:
            self.gold_solutions.append(Solution(string=sol))
        if "sol_tries" in challenge_config:
            for i, x in enumerate(challenge_config["sol_tries"]):
                self.gold_solutions[i].count = x

        if "sol_time" in challenge_config:
            for i, x in enumerate(challenge_config["sol_time"]):
                self.gold_solutions[i].time = x

        self.solution_strs = challenge_config["sols"]
        self.max_ticks = max_ticks

        self._parse_challenge()

    def _parse_challenge(self):
        '''
        Converts the challenge string to a tython program.
        '''
        self.sol_kind = tython.nt.type2nt(self.type)
        self.prog = None
        self.f = None
        try:
            self.prog = tython.Program(
                self.f_str)
            self.f = self.prog.run(max_ticks=self.max_ticks)
        except Program.EvalException as e:
            logger.warning(f"Exception evaluating {self.name} '{self.f_str}': {e}")
        except Exception as e:
            logger.warning(f"Exception parsing {self.name} '{self.f_str}': {e}")
