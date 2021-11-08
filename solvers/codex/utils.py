import functools
import operator
import os
import re
import json
import time
import logging

def get_lambda_arg_name(lam):
    assert lam.startswith("lambda ")
    return lam[len("lambda "):lam.index(":")].strip()


def stringify(const):
    if type(const) is str:
        return json.dumps(const)
    return str(const)


def color_str(obj, code="\033[0;36m"):
    return code + str(obj) + '\033[0m'


def prod(iterable):  # like sum but product
    return functools.reduce(operator.mul, iterable, 1)


def flatten(it):
    return (e for a in it for e in (flatten(a) if isinstance(a, (tuple, list)) else (a,)))

def save_json(obj, filename, make_dirs_if_necessary=False, indent=2, **kwargs):
    """Saves compressed file if filename ends with '.gz'"""
    import json
    if make_dirs_if_necessary:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if filename.endswith(".gz"):
        import gzip
        with gzip.open(filename, "wt") as f:
            return json.dump(obj, f, indent=indent, **kwargs)
    with open(filename, "w", encoding="utf8") as f:
        return json.dump(obj, f, indent=indent, **kwargs)

def load_json(filename):
    """Loads compressed file if filename ends with '.gz'"""
    import json
    if filename.endswith(".gz"):
        import gzip
        with gzip.open(filename, "rt") as f:
            return json.load(f)
    with open(filename, "r", encoding="utf8") as f:
        return json.load(f)


def viz_py(py):
    import astor, ast
    print(astor.dump_tree(ast.parse(py)))


def dedup(li):
    seen = set()
    return [x for x in li if x not in seen and not seen.add(x)]


def test_puzzle(f, x):
    """Checks if x is of the correct type and makes f return True (literally True, not an integer or whatever)

    :param f: Puzzle
    :param x: candidate answer
    :return:
    """
    answer_type = list(f.__annotations__.values())[0]
    if not type_check(x, answer_type):
        raise TypeError
    return f(x) is True



def type_check(obj, typ):
    """
    check if obj is of type `typ` where `typ` is a `typing` module type annotation, eg List[int]
    The way we do this to be compatible across versions is we first convert the type to a string.
    """

    type_str = str(typ).replace("typing.", "")
    if type_str.startswith("<class '"):
        type_str = type_str[8:-2]


    def helper(obj, type_st: str):
        """test if obj is of type type_st"""
        t = {"str": str, "int": int, "float": float, "bool": bool}.get(type_st)
        if t is not None:
            return type(obj) == t
        assert type_st.endswith("]"), f"Strange type `{type_st}`"
        inside = type_st[type_st.index("[")+1:-1].split(", ")
        if type_st.startswith("List["):
            [i] = inside
            return isinstance(obj, list) and all(type_check(elem, i) for elem in obj)
        if type_st.startswith("Set"):
            [i] = inside
            return isinstance(obj, set) and all(type_check(elem, i) for elem in obj)
        print(f"type not handled: {typ}")
        return True

    return helper(obj, type_str)



def rename_src_var(orig, new, src, count=0):
    def helper(s):
        return re.sub(r'\b' + orig + r'\b(?!["\'])', new, s, count)
    if src.count('"""') >= 2:
        a = src.index('"""')
        b = src.index('"""', a+1) + 3
        if count == 1:
            h = helper(src[:a])
            if h != src[:a]:
                return h + src[a:]
        return helper(src[:a]) + src[a:b] + helper(src[b:])

    return helper(src)

logger = None

def timeit(method):
    global logger
    if logger is None:
        logger = logging.getLogger(__name__)
    def timed(*args, **kw):
        tick = time.time()
        result = method(*args, **kw)
        tock = time.time()
        logger.debug(f'{method.__name__}: {tock - tick:.3f}s')

        return result
    return timed

