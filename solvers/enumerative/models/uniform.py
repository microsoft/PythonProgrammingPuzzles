from typing import List, Dict, Callable, Tuple, Generator, Set, Sequence
import functools
import operator
from collections import defaultdict
import random

from tython import Program, TastNode, _RULES_BY_KIND, Rule, nt
from models.model import CandidateGenerator, reachable_rules_by_kind
from models import RegisterModel
from challenges import extract_constants


def prod(iterable):  # like sum but product
    return functools.reduce(operator.mul, iterable, 1)


@RegisterModel("uniform")
class UniformModel(CandidateGenerator):
    '''
    Uniformly samples from all rules.
    '''

    def __init__(self, copy_prob=0.5) -> None:
        super().__init__()
        self.copy_prob = copy_prob
        # self.intended_depth = intended_depth
        #self._random_cache = defaultdict(lambda: defaultdict(dict))
        self._random_cache = {} # defaultdict is not pickable (for multiproc)

    def random(self, kind, max_depth=5, nodes_by_kind=None):
        if nodes_by_kind is None:
            nodes_by_kind = {}
        key = sum(hash(n) for n in nodes_by_kind)
        if key not in self._random_cache:
            self._random_cache[key] = {}
        cache = self._random_cache[key]

        # cache[kind][depth] is a list of available rules

        def available_rules(kind, depth):
            if depth <= 0:
                return []
            if kind in cache and depth in cache[kind]:
                return cache[kind][depth]
            rules = [r for r in _RULES_BY_KIND[kind] if
                     all([nodes_by_kind.get(k) or available_rules(k, depth - 1) for k in r.kids])]
            if kind not in cache:
                cache[kind] = {}
            cache[kind][depth] = rules
            return rules

        def helper(kind, depth):
            assert depth >= 0
            rules = available_rules(kind, depth)
            assert rules or nodes_by_kind.get(kind), f"Cannot generate random {kind} of depth <= {depth}"

            if nodes_by_kind.get(kind) and (not rules or random.random() < self.copy_prob):
                return random.choice(nodes_by_kind[kind])

            rule = random.choice(rules)

            return TastNode(rule, [helper(k, depth - 1) for k in rule.kids])

        return Program(helper(kind, max_depth))

    def get_candidates_by_nodes(self, kind, nodes_by_kind):
        rules_by_kind = _RULES_BY_KIND # zzz reachable_rules_by_kind(kind, nodes_by_kind)

        if kind not in rules_by_kind:
            return {}

        # p_kind_rules = {k: (1 - self.copy_prob if nodes_by_kind.get(k) else 1) / max(1, len(rules_by_kind[k]))
        #                 for k in rules_by_kind}

        by_kind = {}

        for parent_kind in rules_by_kind:
            rules = rules_by_kind[parent_kind]
            has_copy = sum(r.name == "COPY" for r in rules)
            if has_copy:
                assert has_copy == 1
                by_kind[parent_kind] = [], [
                    (self.copy_prob if r.name == "COPY" else (1 - self.copy_prob) / len(rules), r)
                    for r in rules]
            else:
                by_kind[parent_kind] = [], [(1 / len(rules), r) for r in rules]

        ans = {r: [by_kind[k] for k in r.kids]
               for rules in rules_by_kind.values() for r in rules if r.name != "COPY"}
        ans.update({r: [([(1. / len(nodes_by_kind[r.nt]), n) for n in nodes_by_kind.get(r.nt, [])], [])]
                    for rules in rules_by_kind.values() for r in rules if r.name == "COPY"})
        ans[None] = [by_kind[kind]]

        return ans

    def get_candidates(self, q: Program) -> Dict[Rule,
                                                 List[Tuple[List[Tuple[float, TastNode]], List[Tuple[float, Rule]]]]]:

        consts = extract_constants(q)

        sol_type_annotation = q.tree.children[0].children[1].children[1]
        sol_kind = nt.type2nt(eval(sol_type_annotation.src()))

        return self.get_candidates_by_nodes(sol_kind, consts)


if __name__ == "__main__":
    u = UniformModel()
    random.seed(0)
    for _ in range(100):
        p = u.random((List, int), 5)
        print(p.src(safe=False))
        try:
            p.val(max_ticks=1000)
        except Program.EvalException:
            pass
