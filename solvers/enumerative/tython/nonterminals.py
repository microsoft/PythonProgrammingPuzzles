"""
Nonterminals

In our grammar, a nonterminal can be simple like INT, STR or have nested parameters, like LIST[TUPLE[INT, STR]]

The `stock` variable contains the set of nonterminals that we can use (though it is a dict for stability)
like INT, LIST[INT], etc.


Notes:
    Nonterminals are hashable and (LIST[INT] is LIST[INT]) is necessarily True
"""

import astor
from typing import List, Dict, Callable, Tuple, Generator, Set, Sequence
from collections.abc import Iterable

########################################################################################################################
# Class definitions
########################################################################################################################

class Nonterminal:  # root is None for nts like INT/BOOL/etc. or LIST/SET or if it's an instance of one of those
    registry = {}

    def __init__(self, name, root=None, kids=None):
        self.name = name
        self.root = root
        self.kids = kids
        assert name not in self.registry, "two nts with same name"
        if not self.root:
            self.registry[name] = self

    @property
    def kid(self):
        assert self.kids and len(self.kids) == 1
        return self.kids[0]

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def isa(self, *nts): # can be an nt or a list of nts, etc.
        if len(nts)==1 and isinstance(nts[0], Iterable):
            nts = nts[0]
        return any(self is nt or self.root is nt for nt in nts)


class ParametrizedNonterminal(Nonterminal):
    """A nt with kids like LIST so that we can call LIST[INT], LIST[LIST[INT]]

    Note that LIST[LIST[INT]] will always return the same object
    """
    registry = {}

    def __init__(self, name, num_kids=-1):
        self.name = name
        self.instances = {}
        self.root = self.kids = None
        self.num_kids = num_kids
        assert name not in self.registry, "two ParametrizedNonterminals with same name"
        self.registry[name] = self

    def __getitem__(self, key):
        if key in self.instances:
            return self.instances[key]
        kids = key if isinstance(key, tuple) else (key,)

        if self.num_kids != -1:
            assert len(kids) == self.num_kids
        new_name = f'{self.name}[{", ".join(str(k) for k in kids)}]'
        ans = self.instances[key] = Nonterminal(new_name, self, kids)
        return ans


########################################################################################################################
# The "stock" of Nonterminals we actually use in tython, like INT, LIST[INT], TUPLE[STR, INT]
########################################################################################################################


assert len(Nonterminal.registry) == 0  # No nonterminals yet

ARGS = Nonterminal('ARGS')
DEFAULT_ARGS = Nonterminal('DEFAULT_ARGS')
BOOL = Nonterminal('BOOL')
INT = Nonterminal('INT')
FLOAT = Nonterminal('FLOAT')
STR = Nonterminal('STR')
RANGE = Nonterminal('RANGE')
SLICE = Nonterminal('SLICE')
NONE = Nonterminal('NONE')
STMT = Nonterminal('STMT')
EMPTY = Nonterminal('EMPTY')  # empty slice li[2:(empty_slice)]
DIGIT = Nonterminal('DIGIT')  # [0-9]
DIGITS = Nonterminal('DIGITS')
CHARS = Nonterminal('CHARS')
ALPHA_CHAR = Nonterminal('ALPHA_CHAR')  # [a-zA-Z_]
OTHER_CHAR = Nonterminal('OTHER_CHAR')  # other characters including ['",. ]
NAME = Nonterminal('NAME')
FOR_IN_IF = Nonterminal('FOR_IN_IF')
TYPE = Nonterminal('TYPE')
FSTR = Nonterminal('FSTR')  # the inside of the f-string
FORMATTED_VALUE = Nonterminal('FORMATTED_VALUE')  # a {x:y} or just {x} inside of an f-string
Z = Nonterminal('Z')  # catch-all expressions, python types
NUMERIC = Nonterminal('NUMERIC')
BODY = Nonterminal('BODY')  # sequence of statements

LIST = ParametrizedNonterminal('LIST', 1)
SET = ParametrizedNonterminal('SET', 1)
DICT = ParametrizedNonterminal('DICT', 2)
GEN = ParametrizedNonterminal('GEN', 1)
ELTS = ParametrizedNonterminal('ELTS', -1)
COMP = ParametrizedNonterminal('COMP', 1)
ITER = ParametrizedNonterminal('ITER', 1)  # any nt of iterable: a LIST, SET, DICT, or GENERATOR
# PROB = ParametrizedNonterminal('PROB', 1)
# SOL = ParametrizedNonterminal('SOL', 1)
VAR = ParametrizedNonterminal('VAR', 1)

FUNCTION = ParametrizedNonterminal('FUNC', -1)  # any number of kids
TUPLE = ParametrizedNonterminal('TUPLE', -1)  # any number of kids


def _make_stock(MAX_KIND_COMPLEXITY=5):  # TODO: add functions with no argument nts...
    # hashable_types = _BASE + \
    #                  [tupleK(t) for t in _BASE] + \
    #                  [tupleK(t1, t2) for t1 in _BASE for t2 in _BASE]  # + \
    # # [f"Tuple[{t1}, {t2}, {t3}]" for t1 in base for t2 in base for t3 in base]
    # sets = [setK(t) for t in hashable_types]
    # ans = hashable_types + sets
    # ans += [listK(t) for t in ans]
    # ans += [dictK(t1, t2) for t1 in hashable_types for t2 in ans]
    # ans += [listK(t) for t in ans] + [generatorK(t) for t in ans]
    # # ans += [f"Callable[[], {r}]" for r in ans] +
    # ans = dedup([t for t in ans if len(list(flatten((t,)))) < MAX_KIND_COMPLEXITY])
    # ans.append(noneK)
    #
    # ans += [functionK(*p) for r in range(2, 4) for p in itertools.product(_BASE, repeat=r)]
    # ans += [functionK(t, boolK) for t in ans]
    #

    _base_nts = [INT, BOOL, FLOAT, STR, RANGE, Z]

    # these are argument names for the problem function
    _var_nts = [INT, STR, FLOAT, LIST[INT], LIST[STR], LIST[LIST[INT]], Z]

    ans = list(Nonterminal.registry.values())

    ans += [g[k] for g in (ELTS, COMP, LIST, SET, GEN, ITER) for k in (INT, STR, BOOL, FLOAT)]

    ans += [ITER[l[k]] for k in (INT, STR, BOOL, FLOAT) for l in [LIST, ITER]]
    ans.remove(ITER[LIST[STR]])
    ans.remove(ITER[LIST[BOOL]])
    ans += [g[TUPLE[t1, t2]] for t1 in _base_nts for t2 in _base_nts for g in (VAR, GEN, ITER, LIST)]

    ans += [LIST[Z],
            COMP[LIST[Z]],
            COMP[LIST[INT]],
            COMP[Z],
            LIST[LIST[Z]],
            LIST[TUPLE[INT, INT]],
            ELTS[TUPLE[INT, INT]],
            VAR[TUPLE[LIST[FLOAT], LIST[FLOAT]]],
            ARGS,
            DEFAULT_ARGS,
            LIST[TUPLE[Z, Z]],
            TUPLE[Z, Z],
            TUPLE[Z, Z, Z],
            TUPLE[Z, Z, Z, Z],
            TUPLE[STR, STR, STR],
            TUPLE[INT, INT],
            TUPLE[INT, STR],
            LIST[TUPLE[INT, STR]],
            TYPE,
            EMPTY,
            ELTS[Z],
            ELTS[NAME],
            ITER[Z],
            ELTS[TUPLE[INT, STR]],
            SET[Z],
            GEN[Z],
            GEN[TUPLE[STR, STR, STR]],
            GEN[TUPLE[LIST[INT], LIST[INT]]],
            GEN[TUPLE[ITER[INT], ITER[INT]]],
            GEN[TUPLE[ITER[ITER[INT]], ITER[ITER[INT]]]],
            GEN[TUPLE[LIST[LIST[INT]], LIST[LIST[INT]]]],
            GEN[TUPLE[INT, INT]],
            LIST[TUPLE[STR, STR, STR]],
            LIST[LIST[INT]], ITER[ITER[INT]], VAR[ITER[INT]],
            LIST[LIST[FLOAT]],
            VAR[LIST[LIST[FLOAT]]],
            VAR[LIST[FLOAT]],
            VAR[TUPLE[INT, INT, INT]],
            DICT[Z, Z],
            DICT[INT, INT],
            DICT[INT, STR],
            FUNCTION[Z, Z],
            FUNCTION[Z, Z, Z]
            ]  # for specific puzzles..., TODO: expand
    # ans += [FUNCTION[b1, b2] for b1 in (INT, BOOL, FLOAT, STR, LIST[INT], LIST[STR], SET[INT],
    #                                     TUPLE[STR, STR, STR]) for b2 in (INT, BOOL, FLOAT, STR)]

    ans += [ELTS[k.kids] for k in ans if k.isa(DICT)] # Add ELTs for dict's. Dict ELTs have two children.

    ans += [VAR[k] for k in _var_nts]


    return {a: True for a in ans} # make a dictionary (which is unordered so deterministic)


stock = _make_stock()


########################################################################################################################
# Misc functions
########################################################################################################################




def type2nt(type_):
    """
    Convert a python type from typing, like List[int], to one of our "nts"
    :param type_: standard python type like List[int]
    :return: a nt
    """
    base = {int: INT, bool: BOOL, str: STR, range: RANGE, float: FLOAT, type(None): NONE, type: TYPE, slice: SLICE}
    if type_ in base:
        return base[type_]
    name = type_.__origin__.__name__.lower()  # works on python 3.8 and 3.6
    kids = map(type2nt, type_.__args__)
    if name == 'list':
        [k] = kids
        return LIST[k]
    if name == 'tuple':
        return TUPLE[kids]
    if name == 'set':
        [k] = kids
        return SET[k]
    if name == 'callable':
        return FUNCTION[kids]
    assert False, f'Unable to convert `{type_}` to nt'


def annotation2nt(node):
    return node and type2nt(eval(astor.to_source(node)))


def nt2type_str(nt):
    return str(nt).lower().replace("list", "List").replace("dict", "Dict").replace("tuple", "Tuple").replace("set", "Set")

def comparable(nt):
    if nt in {FLOAT, INT, BOOL, STR, Z}:
        return True
    return nt.isa(TUPLE) and all(comparable(k) for k in nt.kids)


def hashable(nt):
    return comparable(nt) or nt == RANGE


def iterable(nt):
    return nt in (RANGE, STR) or any(nt.isa(k) for k in [LIST, DICT, GEN, SET, ITER])


def iterand(nt):
    if nt == RANGE:
        return INT
    if nt == STR:
        return STR
    if nt.isa(TUPLE):
        return nt.kids[0] if len(set(nt.kids)) == 1 else Z
    elif nt == Z:
        return Z
    else:
        assert nt.isa(LIST) or nt.isa(SET) or nt.isa(DICT) or nt.isa(GEN) or nt.isa(ITER)
    return nt.kid


def typable(nt):
    return any(nt.isa(g) for g in [INT, BOOL, FLOAT, STR, RANGE, Z, TUPLE, DICT, LIST, SET, GEN, TYPE])


typables = [k for k in stock if typable(k)]

nequalables = [[INT, FLOAT], [FLOAT, INT]] + [[k, k] for k in typables
                                              if not k.isa(FUNCTION) and not k.isa(GEN)]

equalables = nequalables + [[k, k, k] for k in typables
                            if not k.isa(FUNCTION) and not k.isa(GEN)]

comparables = [ks for ks in equalables if comparable(ks[0])]


# def pretty_type(type_):
#     return str(type2nt(type_))
