from typing import List, Dict, Callable, Tuple, Generator, Set, Sequence

from tython import Program, TastNode, Rule, _RULES_BY_KIND

def reachable_rules_by_kind(sol_kind, terminal_kinds) -> Set:
    '''
    Finds all kinds that could be in a tree with sol_kind as the root
    and terminal kinds as possible kinds for constants.
    '''
    # some kinds are unusuable because there is no way to generate a complete tree
    # starting with them.
    completable = set(terminal_kinds)  # force copy
    signatures = {k: {r.kids for r in _RULES_BY_KIND[k]} for k in _RULES_BY_KIND}
    uncompletable = set(signatures) - completable
    while True:
        completable.update({k for k in uncompletable
                            if any(all(k in completable for k in s) for s in signatures[k])})
        if not completable.intersection(uncompletable):
            break
        uncompletable -= completable
    for k in signatures:
        signatures[k] = {s for s in signatures[k] if all(k2 in completable for k2 in s)}
    signatures = {k: s for k, s in signatures.items() if s}
    if sol_kind not in completable:
        return {}

    good_kinds = set()
    kind_queue = {sol_kind}

    while kind_queue:
        parent_kind = kind_queue.pop()
        good_kinds.add(parent_kind)
        kind_queue.update({k for sig in signatures[parent_kind]
                           for k in sig if k not in good_kinds})

    good_rules_by_kind = {k: [r for r in _RULES_BY_KIND[k] if all(k2 in good_kinds for k2 in r.kids)]
                          for k in _RULES_BY_KIND if k in good_kinds}

    return {k: v for k, v in good_rules_by_kind.items() if v}

class CandidateGenerator:
    '''
    A model that generate candidate solutions for challenges.
    '''

    def __init__(self) -> None:
        pass

    def learn(self, QAs):
        return

    def get_candidates(self, q: TastNode) -> Dict[Rule,
                                                  List[Tuple[List[Tuple[float, TastNode]], List[Tuple[float, Rule]]]]]:
        '''
        Get solution candidates for a question q.
        TODO: program instead of TastNode?
        '''
        pass

