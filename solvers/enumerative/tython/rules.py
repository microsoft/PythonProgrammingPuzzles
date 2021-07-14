"""
tython -- timed and gently typed python

Usage:
> Program("sum([2*i for i in range(1000)])").val()
999000

> Program("sum([2*i for i in range(1000)])").val(max_ticks=100)
TimeoutError

> print(Program("sum(range(1000))").tree
(sum: int, (range: range, (const: int, (literal: LIT, 1000))))
our own internal format with a set of rules

> p = Program("[i for i in range(27)]")
> p.val()
[0, 1, ..., 26]
> print(p.clock) # tell how much time was used
27.0

# How much time was used is tricky if the return value is a function.
> p = Program("lambda n: sum([i for i in range(n)])", lambda_nts={"n": int})
> f = p.val()
> f(3)
6
> print(p.clock)
6.0
> f(3)
> print(p.clock)
12.0
> f(3)
6
> print(p.clock)
18.0
> p.reset_clock()
> f(3)
6
> print(p.clock)
6.0
"""

import os
import re
import itertools
import json
import math
from random import Random
from math import cos, sin, pi, log, exp, log2, inf
import astor
import ast
from typing import List, Dict, Callable, Tuple, Generator, Set, Sequence
import time
import operator
import inspect
import sys
from orderedset import OrderedSet
from collections import defaultdict
import glob

import utils

from .nonterminals import *  # so we can just write LIST[TUPLE[INT, STR]] and stock

print("Running rules.py")

sys.setrecursionlimit(3000)

start_time = time.time()


########################################################################################################################
# Helper funcs
########################################################################################################################

# def _find_first(key, dicts, expected_nt):
#     for d in dicts:
#         if key in d:
#             return dicts[k]
#     raise KeyError(f"variable '{k}' not defined in environment")


def make_const_func(const):
    def f(*args, **kw_args):
        return const

    return f


########################################################################################################################


########################################################################################################################
# Rule defs
########################################################################################################################
DEF_RULE = None
rules = []
_rules_by_nt = {}
_rule_lookup = {}
_dup_rules = set()
_env = None  # current evaluation environment


class Rule:
    def __init__(self, index, name, t, children_nts, to_python_funcs):
        self.index = index
        self.name = name
        self.nt = t
        self.kids = tuple(children_nts)
        if callable(to_python_funcs):
            self.to_python = self.to_python_safe = to_python_funcs
        else:
            self.to_python, self.to_python_safe = to_python_funcs

    @property
    def kid(self):
        assert self.kids and len(self.kids) == 1
        return self.kids[0]

    def __repr__(self):
        try:
            desc = self.to_python(*["_" for _ in self.kids]) + "' '" + self.name
        except:
            desc = self.name
        return f"'{desc}' {self.nt} :: ({', '.join(map(str, self.kids))}) {self.index}"

    def var_repr(self):
        var_list = ['x', 'y', 'z', 'a', 'b', 'i', 'j', 'k']

        try:
            desc = self.to_python(*[var_list[i] for i in range(len(self.kids))]) + "' '" + self.name
        except:
            desc = self.name

        var_desc = [f"{v}: {d}" for v, d in zip(var_list, self.kids)]
        return f"{desc} -> {self.nt} :: ({', '.join(var_desc)})"


# _all_names = []


def add_rule(name, to_py_funcs, t, children_nts):
    children_nts = tuple(children_nts)
    for k in [t, *children_nts]:
        if k not in stock:
            # print(f"* Unknown nonterminal {k} in rule `{name}`")
            return
    rule = Rule(len(rules), name, t, children_nts, to_py_funcs)
    idx = (name, *children_nts)
    if idx in _rule_lookup:
        print(f"Duplicate rule:\n    ignoring {rule}\n keeping  {_rule_lookup[idx]}")
        return
    # _all_names.append(name)
    rules.append(rule)
    _rules_by_nt.setdefault(t, []).append(rule)
    _rule_lookup[idx] = rule
    if name=="def":
        global DEF_RULE
        assert DEF_RULE is None
        DEF_RULE = rule


def add_rules(name, to_py_funcs, nt_or_nt_lists, kids_lists=None):
    """Adds multiple rules with the same name but different nt signatures. See usage examples below.

    :param name: The name of the rule
    :param to_py_funcs: func generating source code from children source OR two funcs (regular_fn, safe_fn)
    :param nt_or_nt_lists: (A) the return nt OR (B) a list of lists, each list being (return nt, *children_nts)
    :param kids_lists: (A) the list of children_nt lists or (B) None
    :return: None
    """
    if kids_lists:
        nt = nt_or_nt_lists
        for kids in kids_lists:
            add_rule(name, to_py_funcs, nt, kids)
    else:
        nt_lists = nt_or_nt_lists
        assert nt_lists
        for nt, *kids in nt_lists:
            add_rule(name, to_py_funcs, nt, kids)


_cast_costs = {}


def add_cast(parent_nt, other_nt, cost=1.0):
    if parent_nt == other_nt:
        return
    if (f'cast:{parent_nt}', other_nt) in _rule_lookup:
        return  # ignore extra cast
    add_rule(f'cast:{parent_nt}', identity, parent_nt, [other_nt])
    _cast_costs.setdefault(other_nt, []).append((cost, parent_nt))


def add_casts(parent_nt, other_nts, cost=1.0):
    for k in other_nts:
        add_cast(parent_nt, k, cost)


def update_cast_costs():
    for k in stock:
        _cast_costs.setdefault(k, []).append((0.0, k))
    for k in _cast_costs:
        _cast_costs[k].sort(key=lambda z: z[0])


########################################################################################################################
# The actual rules
########################################################################################################################

def identity(x):
    return x


add_rule(
    "empty",
    lambda: "",
    EMPTY,
    []  # no children
)

_num_binary_nts = {
    FLOAT: [[FLOAT, FLOAT], [INT, FLOAT], [FLOAT, INT], (FLOAT, FLOAT, BOOL), (FLOAT, BOOL, FLOAT)],
    INT: [(BOOL, BOOL), (INT, BOOL), (BOOL, INT), [INT, INT]],
    Z: [(Z, Z)]
}

_tuple_nts = [k for k in stock if k.isa(TUPLE)]
_dict_nts = [k for k in stock if k.isa(DICT)]
_set_nts = [k for k in stock if k.isa(SET)]


def infix(op_str):
    return lambda *args: "(" + (")" + op_str + "(").join(args) + ")"


def prefix(func_name):
    return lambda *args: f"{func_name}({', '.join(args)})"


def suffix(func_name):
    return lambda a, *args: f"({a}).{func_name}({', '.join(args)})"


def safe_prefix(func_name):
    return prefix(func_name), prefix('safe.' + func_name)


def safe_suffix(func_name):
    return suffix(func_name), prefix('safe.dot_' + func_name)


add_casts(NUMERIC, [FLOAT, INT], 0.5)

add_casts(TUPLE[Z, Z], [TUPLE[k, Z] for k in typables])

add_rules(
    "not",
    lambda a: f"not ({a})",
    BOOL,
    [[k] for k in typables]
)

# TODO: make safe for INT*INT, INT+INT, Z*Z
for k, kids in _num_binary_nts.items():
    for op in ["*", "+", "-", "//"]:
        add_rules(
            op,
            infix(op),
            k,
            kids
        )
        add_rules(
            op + "=",
            infix(op + "="),
            STMT,
            kids
        )

add_rules(
    "*=",  # not exactly equivalent...
    (lambda var, val: f"{var} *= ({val})", lambda var, val: f"{var} = safe.mul(var, val)"),
    STMT,
    [(VAR[STR], INT), (VAR[Z], Z), (Z, INT)]
)

add_rules(  # not exactly equivalent...
    "+=",
    (lambda var, val: f"{var} += ({val})", lambda var, val: f"{var} = safe.add(var, val)"),
    STMT,
    [(VAR[STR], VAR[STR]), (VAR[LIST[INT]], VAR[LIST[INT]]), (VAR[Z], Z)]
)

for k in stock:
    if k.isa(LIST) or k == STR:
        add_rules(  # "cat" * 3 or 3 * "cat" or [1,2]*2
            "*",
            (infix("*"), lambda a, b: f"safe.mul({a}, {b})"),
            k,
            [[k, INT], [INT, k]]
        )
        add_rules(  # "cat" + "dog" or [1,2,3] + [3,4,5]
            "+",
            (infix("+"), lambda a, b: f"safe.add({a}, {b})"),
            k,
            [[k, k]]
        )

add_rules(
    "%",
    (infix("%"), prefix('safe.mod')), # safety for string %
    [[INT, INT, INT], [Z, Z, Z]]
)

for op in ["&", "|", "^"]:
    add_rules(
        op,
        infix(op),  # TODO: add safety (and types) for set operations
        INT,
        [(INT, INT), (Z, Z)]
    )

add_rules(
    "/",
    lambda a, b: f"({a})/({b})",
    FLOAT,
    [(NUMERIC, NUMERIC), (Z, Z)]
)

add_rules(
    "<<",
    (infix("<<"), prefix('safe.lshift')),
    INT,
    [(INT, INT), (Z, Z)]
)

add_rules(
    "**",
    (infix("**"), prefix("safe.pow")),
    [(INT, INT, INT), (FLOAT, FLOAT, NUMERIC), (Z, Z, Z)]
)

add_rules(
    "==",
    infix("=="),  # TODO: make safe with timing
    BOOL,
    equalables
)

add_rules(
    "!=",
    infix("!="),  # TODO: make safe with timing
    BOOL,
    nequalables
)

for r in ("is", "is not"):
    add_rules(
        r,
        infix(r),
        [[BOOL, n, n] for n in typables]
    )

for op in ["<=", "<", ">=", ">"]:
    add_rules(
        op,
        infix(op),  # TODO: make safe with timing
        BOOL,
        comparables
    )

_listable_nts = [k for k in stock if LIST[k] in stock]
_iter_nts = [k for k in stock if k.isa(ITER)]

for op in ['or', 'and']:
    add_rule(
        op,
        infix(op),  # TODO: make safe with timing
        BOOL,
        [BOOL, BOOL]
    )

for c, k in [("None", NONE), ("True", BOOL), ("False", BOOL)]:
    add_rules(c, make_const_func(c), [[k]])  # no children

# TODO: convert to functions so that map(abs, ...) will work
_builtin_funcs = {
    # the first item in each tuple is the t and the rest are the children stock
    #     "abs": [(INT, INT), (FLOAT, FLOAT), (NUMERIC, Z)],
    #     "len": [(INT, ITER[Z])],
    #     "sum": [(INT, ITER[INT]), (FLOAT, ITER[FLOAT]), (NUMERIC, Z)],
    #     "range": [(RANGE, INT), (RANGE, INT, INT), (RANGE, INT, INT, INT)]
    # }
    "abs", "len", "range", "sum", "str", "any", "all", "min", "max", "sorted", "chr", "ord",
    "list", "set", "tuple", "zip",
    "cos", "sin", "log", "exp", "round", "float", "int", "bool", "reversed", "type",
    "COPY"}  # removed "random", "randint", "repr",

_builtin_constants = {"pi", "inf"}

_fixed_dot_funcs = ["join", "split", "count", "endswith", "startswith", "index", "replace",
                    "append", "union", "issuperset", "issubset"]

_unary_numeric_nts = {INT: [(INT,), (BOOL,)], FLOAT: [(FLOAT,)], Z: [(Z,)]}

add_rules(
    "join",
    safe_suffix("join"),
    STR,
    [[STR, LIST[STR]], [STR, GEN[STR]], [STR, LIST[Z]], [STR, Z], [Z, Z]]
)  # todo: add dictionaries with string keys or a nonterminal with orderedStrIterable or something

add_rules(
    "split",
    safe_suffix("split"),
    LIST[STR],
    [[STR, STR], [STR]]
)

for r in ["count", "index"]:
    add_rules(
        r,
        safe_suffix(r),
        INT,
        [(LIST[k], k) for k in _listable_nts] + [(STR, STR), (Z, Z)]
    )

add_rules(
    "append",
    suffix("append"),
    STMT,
    [(t, t.kid) for t in stock if t.isa(LIST)]
    + [(t, Z) for t in stock if t.isa(LIST) and t.kid != Z]
    + [(Z, Z)]
)

add_rules(
    "replace",
    safe_suffix("replace"),
    STR,
    [[STR, STR, STR], [Z, Z, Z]]
)

for r in ["startswith", "endswith"]:
    add_rule(
        r,
        suffix(r),
        BOOL,
        [STR, STR]
    )

for r in ["issubset", "issuperset"]:
    add_rules(
        r,
        safe_suffix(r),
        BOOL,
        [[SET[iterand(k)], k] for k in _iter_nts]
    )

for k in _iter_nts:
    sk = SET[iterand(k)]
    add_rule(
        "union",
        safe_suffix("union"),
        sk,
        [sk, k]
    )

for name, op in [("-unary", "-"), ("+unary", "+"), ("abs", "abs")]:
    for k, kids in _unary_numeric_nts.items():
        add_rules(
            name,
            prefix(op),
            k,
            kids
        )

add_rules(
    "round",
    prefix("round"),
    INT,
    [[FLOAT], [Z], [FLOAT, INT], [Z, Z]]
)

_GENERATOR_KINDS = [k for k in stock if GEN[k] in stock]

# add_rules(
#     "reversed",
#     prefix("reversed"),
#     [(GENERATOR[k], LIST[k]) for k in stock if GENERATOR[k] in stock] + [(GENERATOR[STR])]
# )

add_rules(
    "reversed",
    prefix("reversed"),
    [(GEN[k], LIST[k]) for k in _GENERATOR_KINDS]
    + [(GEN[Z], Z)]
    + [(GEN[STR], STR)]
    + [(GEN[INT], RANGE)]
)

add_rules("chr", prefix("chr"), [(STR, INT), (STR, Z)])

add_rules("ord", prefix("ord"), [(INT, STR), (INT, Z)])

for f_name in ["cos", "sin", "log", "exp"]:
    add_rules(f_name, prefix(f_name), [(FLOAT, FLOAT), (FLOAT, Z)])
# add_rules("pi", lambda: "pi", [(floatK,)])


add_rules(
    "len",
    prefix("len"),
    [(INT, k) for k in stock if any(k.isa(g) for g in [RANGE, STR, LIST, SET, DICT, Z])]  # can't len(generator)
)

add_rules("range", prefix("range"), [(RANGE,) + (k,) * i for i in range(1, 4) for k in (INT, Z)])

for k in _iter_nts:
    k2 = iterand(k)
    if k2 in {FLOAT, INT, BOOL, Z}:
        add_rules(
            "sum",
            safe_prefix('sum'),  # todo: deal with sums of tuples
            {BOOL: INT, INT: INT, FLOAT: FLOAT, Z: Z}[k2],
            [[k]]
        )


def stringable_nt(k):  # TODO: make more general for stock
    if k in {STR, FLOAT, INT, BOOL, NONE, RANGE}:
        return True
    if k.isa(LIST) or k.isa(TUPLE) or k.isa(DICT):
        return all(stringable_nt(k2) for k2 in k.kids)
    return False


_STR_KINDS = [k for k in stock if stringable_nt(k)]

# add_rules("randint", prefix("randint"), intK, [[intK, intK]])
# add_rules("random", prefix("random"), [(floatK,)])
add_rules(
    "str",
    safe_prefix("str"),
    STR,
    [[k] for k in _STR_KINDS] + [[Z]]
)  # enforce determinism by excluding sets and funcs
# TODO: remove this requirement by enforcing determinism by setting env var PYTHONHASHSEED=0

add_rules(
    "type",
    prefix("type"),
    TYPE,
    [[k] for k in typables]
)

add_rules(
    "bool",
    prefix("bool"),
    BOOL,
    [[k] for k in typables]
)

add_rules(
    "float",
    prefix("float"),
    FLOAT,
    [[STR], [INT], [FLOAT], [BOOL], [Z]]
)

add_rules(
    "int",
    prefix("int"),
    INT,
    [[STR], [INT], [FLOAT], [BOOL], [Z]]
)

# add_rules("repr", safe_prefix("repr"), [(strK, k) for k in _TYPE_KINDS])

for r in ["any", "all"]:
    add_rules(
        r,
        safe_prefix(r),
        BOOL,
        [[k] for k in _iter_nts] + [[Z]]
    )

# CONSTS = {
# }
#
# for k in CONSTS:
#     for c in CONSTS[k]:
#         add_rules(stringify(c), make_const_func(stringify(c)), [(k,)])

for k in typables:
    add_rule(
        "COPY",
        prefix("COPY"),
        k,
        [k]
    )

add_rule(
    "list",
    safe_prefix('list'),
    LIST[Z],
    [Z]
)

for k in _iter_nts:
    add_rule(
        "list",
        safe_prefix('list'),
        LIST[iterand(k)],
        [k]
    )
    if comparable(iterand(k)) or iterand(k) == Z:
        for r in ['min', 'max']:
            add_rule(
                r,
                safe_prefix(r),
                iterand(k),
                [k]
            )

sorted_sigs = [(LIST[iterand(k)], k) for k in _iter_nts if comparable(iterand(k))] + [(LIST[Z], Z)]

add_rules("sorted", safe_prefix('sorted'), sorted_sigs)
add_rules("revsorted",
          (lambda a: f"sorted({a}, reverse=True)", lambda a: f"safe.sorted({a}, reverse=True)"),
          sorted_sigs)

add_rule(  # overly specific rule :-(
    "tuple",
    prefix('tuple'),
    TUPLE[STR, STR, STR],
    [LIST[STR]]
)

add_rules("zip", prefix('zip'),
          [(GEN[TUPLE[iterand(t1), iterand(t2)]], t1, t2) for t1 in _iter_nts for t2 in _iter_nts])

add_rule(
    "zip",
    prefix('zip'),
    GEN[TUPLE[STR, STR, STR]],
    [STR, STR, STR]
)

for k in _iter_nts:
    if hashable(iterand(k)):
        add_rule(
            "set",
            safe_prefix('set'),
            SET[iterand(k)],
            [k]
        )

# should add this back once there are callable kinds
# _CALLABLE_KINDS = [k for k in stock if k.isa(FUNCTION)]
#
# for k in _CALLABLE_KINDS:
#     *args, ret = k.kids
#     add_rule(
#         "call",
#         lambda f, *args: f"({f})({', '.join(args)})",
#         ret,
#         [k, *args]
#     )

# add_rule(
#     "lambda",
#     lambda *args: f"lambda {', '.join(args[:-1])}: ({args[-1]})",
#     k,
#     ([defK] * len(args)) + [ret]
# )

for k in _tuple_nts:
    add_rule(
        "(tuple)",
        lambda *args: f"({args[0] + ',' if len(args) == 1 else ', '.join(args)})",
        k,
        k.kids  # todo: make more general for stock
    )


def py_dict(*args):
    assert len(args) in {2, 3}
    return ("" if len(args) == 2 else args[0] + ", ") + args[-2] + ": " + args[-1]


# TODO: fix {a:b, c:d, ...}
# for k in _dict_nts:
#     add_rule(
#         "{dict}",
#         lambda elts: "{" + elts + "}",
#         k,
#         [ELTS[k.kids]]
#     )
#     add_rule(  # this may be buggy!
#         "Elts",
#         py_dict,
#         ELTS[k.kids],
#         [ELTS[k.kids], *k.kids]
#     )


for ek in filter(lambda ek: ek.isa(ELTS) and len(ek.kids) == 1, stock):  # not a dictionary
    k = ek.kid
    add_rule(
        'Elts',
        lambda *args: ", ".join(args),
        ELTS[k],
        [k, ELTS[k]]
    )
    add_rule(
        f'Elts0:{k}',  # since Nonterminal cannot be determined by kids
        lambda: "",
        ELTS[k],
        [],
    )
    if LIST[k] in stock:  # this check isn't necessary if the rule adder ensures nonterminals are in stock
        add_rule(
            "[list]",
            lambda elts: f"[{elts}]",
            LIST[k],
            [ELTS[k]]
        )
    # else:
    #     print("** ooh", k)

for k in _set_nts:
    add_rule(
        "{set}",
        (lambda elts: "{" + elts + "}" if elts else "set()", lambda elts: f"safe.set([{elts}])"),
        k,
        [ELTS[k.kid]]
    )

add_rules(
    '{dict}',
    lambda elts: "{" + elts + "}",
    [(t, ELTS[t.kids]) for t in _dict_nts]
)

add_rules(
    'Elts',
    lambda k, v, rest: f"{k}: {v}" + (rest and f', {rest}'),
    [(ELTS[t.kids], t.kids[0], t.kids[1], ELTS[t.kids]) for t in _dict_nts]
)

for t in _dict_nts:
    k = t.kids
    add_rule(  # since empty list of ELTS Nonterminal cannot be determined by kids
        f'Elts0:{k}',
        lambda: "",
        ELTS[k],
        []
    )

add_rules(
    "Digits",
    lambda *args: "".join(args),
    DIGITS,
    [[DIGIT, DIGITS], [DIGIT]]
)

_digits = "0123456789"
for i in _digits:
    add_rule(
        i,
        make_const_func(i),
        DIGIT,
        []  # no children
    )

_alpha_chars = "abcdefghijklmnopqrstuvwxyz_"
for i in _alpha_chars:
    add_rule(
        f"'{i}'",
        make_const_func(i),
        ALPHA_CHAR,
        []
    )

add_rule(
    "upper_alpha_char",
    lambda x: x.upper(),
    ALPHA_CHAR,
    [ALPHA_CHAR]
)

add_rules(
    'name',
    lambda *args: ''.join(args),
    NAME,
    [[ALPHA_CHAR], [NAME, ALPHA_CHAR], [NAME, DIGIT]]
)

add_rules(
    'Chars',
    lambda *args: ''.join(args),
    CHARS,
    [[ALPHA_CHAR, CHARS], [DIGIT, CHARS], [OTHER_CHAR, CHARS], []]
)

_other_chars = '''!@#$%^&*'":;,. '''
for c in _other_chars:
    add_rule(
        c,
        make_const_func(c),
        OTHER_CHAR,
        []
    )

add_rule(
    "other_char",
    lambda num_str: chr(min(int(num_str), 1000)),
    OTHER_CHAR,
    [DIGITS]
)

add_rule(
    "str-const",
    utils.stringify,
    STR,
    [CHARS]
)

add_rule(
    "int-const",
    lambda x: x.lstrip("0") or "0",
    INT,
    [DIGITS]
)

add_rule(
    "float-const",  # 2.4
    lambda a, b: a + "." + b,
    FLOAT,
    [DIGITS, DIGITS]
)

add_rule(
    "float-const-large",  # like 2.1e100
    lambda a, b, e: a + "." + b + "e" + e,
    FLOAT,
    [DIGITS, DIGITS, DIGITS]
)

add_rule(
    "float-const-tiny",  # like 2.1e-100
    lambda a, b, e: a + "." + b + "e-" + e,
    FLOAT,
    [DIGITS, DIGITS, DIGITS]
)

add_rule(
    "inf",
    lambda: "inf",
    FLOAT,
    []  # no children
)

_SUBSCRIPT_KINDS = ([(Z, Z, SLICE), (Z, Z, Z)] +
                    [(INT, RANGE, INT), (RANGE, RANGE, SLICE), (STR, STR, INT), (STR, STR, SLICE)] +
                    [((LIST[k], LIST[k], SLICE)) for k in _listable_nts] +
                    [((k, LIST[k], INT)) for k in _listable_nts] +
                    [(k.kids[1], k, k.kids[0]) for k in stock if k.isa(DICT)])

for (k, *kids) in _SUBSCRIPT_KINDS:
    add_rule(
        "[i]",
        lambda a, i: f"({a})[{i}]",
        k,
        kids
    )


# Subscripts for tuples must be constants since otherwise we cannot determine types
def _tup_at(i):
    return lambda a: f"({a})[{i}]"


_max_tuple_len = max(len(k.kids) for k in _tuple_nts) if _tuple_nts else 0
for i in range(-_max_tuple_len, _max_tuple_len):
    for k in _tuple_nts:
        if -len(k.kids) <= i < len(k.kids):
            add_rule(
                f"[{i}]",
                _tup_at(i),
                k.kids[i],
                [k]
            )

add_rules(
    ":slice",
    lambda a, b, c: f"{a}:{b}:{c}",
    SLICE,
    list(itertools.product([INT, EMPTY, Z], repeat=3))
)

# TODO if we want
# add_rule(
#     "import",
#     lambda *args: f"import " + ', '.join(args),
#     STMT,
#     [[NAME]*i for i in range(1, 4)]
# )
#
#
# add_rule(
#     f"import_from",
#     lambda m, *args: f"from {m} import " + ', '.join(args),
#     STMT,
#     [[NAME]*i for i in range(2, 5)]
# )
#
# add_rule(
#     f"import_from.",
#     lambda m, *args: f"from .{m} import " + ', '.join(args),
#     STMT,
#     [[NAME]*i for i in range(2, 5)]
# )
#
# add_rule(
#     f"import_from..",
#     lambda m, *args: f"from ..{m} import " + ', '.join(args),
#     STMT,
#     [[NAME]*i for i in range(2, 5)]
# )


_var_nts = [INT, FLOAT, STR, LIST[STR], LIST[INT], LIST[LIST[INT]], SET[INT], TUPLE[STR, STR, STR], Z]

for v_nt in [k for k in stock if k.isa(VAR)]:
    t = v_nt.kid
    add_rules(
        f'var:{t}',
        identity,
        v_nt,
        [[NAME], [t]]  # t.kid catches things like a[17] = ...
    )
    if t.isa(TUPLE) and len(t.kids) > 1:
        add_rule(
            f'var:{t}',
            lambda *vars: f"({', '.join(vars)})",
            v_nt,
            [NAME] * len(t.kids)  # shortcut for x,y,z=1,2,3
        )
    # add_cast(VAR[Z], v_nt)

    add_rule(
        f'access_var:{t}',
        identity,
        t,
        [NAME]
    )

for k in _tuple_nts:
    add_rule(
        "def_ANY_tuple",
        lambda *args: f"({','.join(args)})",
        VAR[k],
        [k]
    )


# zzz add LIST[LIST[k]] casts to ITER[ITER[k]]

def get_iterables(k):
    ans = [LIST[k], SET[k], GEN[k], TUPLE[k, k], TUPLE[k, k, k]]
    if k == INT:
        ans.append(RANGE)
    elif k == STR:
        ans.append(STR)
    return ans


for k in _iter_nts:
    k2 = iterand(k)
    if k2.isa(ITER):
        k3 = iterand(k2)
        add_casts(k, [g2 for g in get_iterables(k3) for g2 in get_iterables(g)], 0.1)
        add_casts(VAR[k], [VAR[g2] for g in get_iterables(k3) for g2 in get_iterables(g)], 0.1)
    else:
        add_casts(k, get_iterables(k2), 0.1)
        add_casts(VAR[k], [VAR[g] for g in get_iterables(k2)], 0.1)

# add_rules(
#     "iterable_pair",
#     identity,
#     ITERABLE_PAIR,
#     [[k] for k in _iterable_nts
#      if tuple_nt[k.kid] and len[k.kid] == 3 or k.isa(LIST)]
# )

add_rule(
    "f_string",
    lambda fs: f'f"{fs}"' if "'" in fs else f"f'{fs}'",
    STR,
    [FSTR]
)

for nt in typables:
    add_rule(
        f'type:{nt}',
        make_const_func(nt2type_str(nt)),
        TYPE,
        []
    )


def formatted_string_inside(*args):
    if len(args) == 0:
        return ''
    a, b = args
    return a + (b if b.startswith("{") else b.replace('"', '\\"').replace("'", "\\'"))


add_rules(
    "f_string_inside",
    formatted_string_inside,
    FSTR,
    [[], [FSTR, CHARS], [FSTR, FORMATTED_VALUE]]
)

add_rules(
    "formatted_value",
    lambda *args: "{" + ':'.join(args) + "}",
    FORMATTED_VALUE,
    [[Z, FORMATTED_VALUE], [Z]]
)


def indent(src):
    return "\n".join(["    " + line for line in src.split("\n")])


def nt2type(t):
    d = {
        RANGE: 'range',
        STR: 'str',
        FLOAT: 'float',
        BOOL: 'bool',
        NONE: 'type(None)',
        INT: 'int'
    }
    if t in d:
        return d[t]
    if t.isa(FUNCTION):
        *args, ret = [nt2type(k) for k in t.kids]
        return f"Callable[[{', '.join(args)}],{ret}]"
    m = {
        LIST: "List",
        SET: "Set",
        GEN: "Generator",
        TUPLE: "Tuple",
        DICT: "Dict"
    }
    assert t.root in m, f'Cannot convert t `{t}` to type'
    return m[t.root] + "[" + ', '.join(nt2type(k) for k in t.kids) + "]"


def make_ps_funcs(t):
    try:
        ty = nt2type(t)
        p_annotation = ": " + ty
        s_annotation = "-> " + ty
    except AssertionError:
        p_annotation = s_annotation = ""
    return [lambda v, body: f"def problem({v}{p_annotation}):\n{indent(body)}",
            lambda body: f"def solution(){s_annotation}:\n{indent(body)}"]


# for t in [t2.kid for t2 in stock if t2.isa(PROB)]:
#     p_func, s_func = make_ps_funcs(t)
#     add_rules(f'def_problem(: {t})', p_func, [(PROB[t], NAME, BODY)])
#     add_rules(f'def_solution->{t}', s_func, [(SOL[t], BODY)])


def args2py(name, *args):
    *type_holder, default, rest = args
    type_str = f": {type_holder[0]}" if type_holder else ""
    return f'name{type_str}={default}' + (f", {rest}" if rest else '')


def comma_if_neccessary(st):
    return ", " + st if st else ''


# ARGS is a list of arguments without default values followed by a list of arguments with default values.
# For example, in (lambda a, b: int, c=17: a+b+c) a and b are args and c is a default_args

add_rules(
    'arg',
    lambda name, typ, rest: f"{name}: {typ}{comma_if_neccessary(rest)}",
    [(ARGS, NAME, TYPE, ARGS)]
)

add_rules(
    'arg',
    lambda name, rest: f"{name}{comma_if_neccessary(rest)}",
    [(ARGS, NAME, ARGS)]
)

add_cast(ARGS, DEFAULT_ARGS, cost=0.0)

add_rules(
    'default_arg',
    lambda name, typ, default, rest: f"{name}: {typ}={default}{comma_if_neccessary(rest)}",
    [(DEFAULT_ARGS, NAME, TYPE, Z, DEFAULT_ARGS)]  # DEFAULT_ARGS can only be followed by other DEFAULT_ARGS
)

add_rules(
    'default_arg',
    lambda name, default, rest: f"{name}={default}{comma_if_neccessary(rest)}",
    [(DEFAULT_ARGS, NAME, Z, DEFAULT_ARGS)]  # DEFAULT_ARGS can only be followed by other DEFAULT_ARGS
)

add_rules(  # DEFAULT_ARGS ends with optional *var_args and **kwargs
    "*args",
    lambda *args: ', '.join(pre + a for pre, a in zip(['*', '**'], args) if a),
    DEFAULT_ARGS,
    [[],  # end of argument list, no children
     [NAME],  # with *var_args
     [NAME, NAME],  # with *var_args and **kwargs
     [EMPTY, NAME]]  # with **kwargs only
)

# function definition.
add_rules(
    "def",
    lambda name, args, body: f'def {name}({args}):\n{indent(body)}',
    [(STMT, NAME, ARGS, BODY)]
)

add_rules(
    'return',
    lambda a: f"return ({a})",
    STMT,
    [[BOOL], [STR], [INT], [FLOAT], [LIST[INT]], [LIST[LIST[INT]]], [Z]]
)

add_rules(
    'body',
    lambda *args: "\n".join(args).rstrip(),
    [(BODY, STMT), (BODY, STMT, BODY)]
)

add_rules(
    'pass',
    lambda: 'pass',
    [[STMT]]
)

add_rules(
    "assert",
    lambda assertion, msg=None: f"assert {assertion}" + ("" if msg is None else f", {msg}"),
    STMT,
    [[Z], [Z, STR]]
)

add_casts(STMT, [Z])

add_casts(Z, typables)
add_casts(Z, [nt for nt in stock if nt.isa(VAR)])
# for k in typables:
#     add_cast(k, Z, 3.0)
#     add_cast(ITER[Z], k, 3.0)

# add_cast(VAR[Z], Z)

# TODO: should time for list assignment x[1:3]= y
add_rules(
    "=",
    infix("="),
    STMT,
    [(VAR[k], k) for k in typables] + [(VAR[TUPLE[k, k]], LIST[k]) for k in typables] + [(Z, Z)]
)

for r in ["in", "not in"]:
    if r == "in":
        py_funcs = (lambda a, b: f"({a}) in ({b})", lambda a, b: f"safe.in_({a}, {b})")
    else:
        py_funcs = (lambda a, b: f"({a}) not in ({b})", lambda a, b: f"safe.not_in({a}, {b})")
    add_rules(r, py_funcs, [(BOOL, iterand(k), k) for k in _iter_nts] + [(BOOL, Z, Z)])


def if_py(test, body, orelse=None):
    ans = f"if {test}:\n{indent(body)}"
    if orelse:
        if orelse.startswith("if "):
            ans += f"\nel{orelse}"
        else:
            ans += f"\nelse:\n{indent(orelse)}"
    return ans


add_rules(
    "if",
    if_py,
    STMT,
    [[Z, BODY], [Z, BODY, BODY]]
)

add_rules(
    "ifExp",
    lambda test, body, orelse: f"({body}) if ({test}) else ({orelse})",
    [(k, Z, k, k) for k in typables]  # a test can be any python obj like `li[0] if li else 17`
)

# A for_in_if is for the [(blah) for _ in _ if _] in an comprehension. Notes:
# * The if part can be any python object. A value of True means that it should be omitted.
# * A comprehension can have multiple for_in_ifs
add_rules(
    "for_in_if",  # always has an ifs but ifs=True is omitted
    (lambda vars, it, ifs: f"for {vars} in ({it})" + (f" if {ifs}" if ifs != "True" else ""),
     lambda vars, it, ifs: f"for {vars} in ({it}) if safe.tick(1" + (f", {ifs})" if ifs != "True" else ")")),
    FOR_IN_IF,
    [(VAR[k], ITER[k], Z) for k in stock if VAR[k] in stock and ITER[k] in stock] + [(Z, Z, Z)]
)

# add_rules(
#     "for_in_if",
#     (
#         lambda target1, target2, it, ifs=None: f"for {target1}, {target2} in ({it})" + (
#             " if " + ifs if ifs else ""),
#         lambda target1, target2, it, ifs=None: f"for {target1}, {target2} in ({it}) if safe.tick(1" + (
#             f", {ifs})" if ifs else ")")
#     ),
#     FOR_IN_IF,
#     [[VAR, VAR, ITER[Z]], [VAR, VAR, ITER[Z], BOOL]]
# )

for k in _listable_nts:
    add_rule(
        "[ListComp]",
        lambda comp: f"[{comp}]",
        LIST[k],
        [COMP[k]]
    )

for k in _set_nts:
    c = [COMP[k.kid]]
    add_rule(
        "{SetComp}",
        lambda comp: "{" + comp + "}",
        k,
        c,
    )
    add_rule(
        "(GeneratorComp)",
        lambda comp: f"({comp})",
        GEN[k.kid],
        c
    )

for k in _listable_nts:
    add_rules(
        "Comp",  # comprehension
        lambda elt, *for_in_ifs: f"{elt} {' '.join(for_in_ifs)}",
        COMP[k],
        [[k, FOR_IN_IF], [k, FOR_IN_IF, FOR_IN_IF]]
    )

add_rules(
    "for",  # for i in ___:
    (lambda v, it, b: f"for ({v}) in ({it}):\n{indent(b)}",
     lambda v, it, b: f"for ({v}) in ({it}):\n{indent('safe.tick(')}{len(b.splitlines())})\n{indent(b)}"),
    STMT,
    [(VAR[iterand(k)], k, BODY) for k in _iter_nts]
    + [(t, Z, BODY) for t in stock if t.isa(VAR, Z)]
)

add_rule(
    "for",
    lambda v1, v2, it, b: f"for ({v1}, {v2}) in ({it}):\n{indent(b)}",
    STMT,
    [VAR[Z], VAR[Z], ITER[Z], BODY]
)


# add_rules("iterable",
#            identity,
#            [(("iterable", intK), range)] +
#            [(("iterable", k), (g, k)) for g in setK(Generator) for k in _SET_KINDS] +
#            [(("iterable", k), listK[k]) for  in setK(Generator)])


def viz_py(py):
    print(astor.dump_tree(ast.parse(py)))


update_cast_costs()


def check_dead_end_rules():
    live_rules_by_nt = _rules_by_nt
    n = sum([len(v) for v in live_rules_by_nt.values()])
    assert n == len(rules)
    while True:
        live_rules_by_nt = {t: [r for r in rules2 if all(live_rules_by_nt.get(k) for k in r.kids)]
                            for t, rules2 in live_rules_by_nt.items()}
        m = sum([len(v) for v in _rules_by_nt.values()])
        if m == n:
            break
    dead_ends = set(rules) - {r for v in live_rules_by_nt.values() for r in v}
    if dead_ends:
        print("Dead end rules", dead_ends)
    else:
        print("No dead end rules")


check_dead_end_rules()

# from collections import Counter
#
# print(Counter(_all_names).most_common(10))

# TODO:
# Functionality to add:
#     [i+j for i, j in [(1,2), (3,4)]]
#     reversed
#     dictionary iterators like {i: i*i for i in range(10)}
#     other builtin functions and numpy functions
#
# rename safe something more intuitive
# Compute minimum rule probabilities
# -- very low priority:
# Make and and or take more than 2 by either having multiple and rules or by having an "inside_and" type
