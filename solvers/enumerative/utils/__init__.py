from utils.str_utils import *
from utils.time_utils import *

import functools
import operator


def prod(iterable):  # like sum but product
    return functools.reduce(operator.mul, iterable, 1)


def flatten(it):
    return (e for a in it for e in (flatten(a) if isinstance(a, (tuple, list)) else (a,)))


def load_json(filename):
    import json
    with open(filename, "r") as f:
        return json.load(f)


def viz_py(py):
    import astor, ast
    print(astor.dump_tree(ast.parse(py)))


def dedup(li):
    seen = set()
    return [x for x in li if x not in seen and not seen.add(x)]
