import astor
import ast
import inspect
import random
import sys
from math import cos, sin, pi, log, exp, log2, log10, inf
from typing import List, Dict, Callable, Tuple, Generator, Set, Sequence
from orderedset import OrderedSet

import utils

from . import parse
from . import rules

sys.setrecursionlimit(10**6)


class Program:
    class EvalException(Exception):
        def __init__(self, base_exception=None, *args):
            self.base_exception = base_exception
            super().__init__(*args)

    class TimeoutEvalException(EvalException):
        pass

    def __init__(self, src_or_root, var_nts=None):
        if isinstance(src_or_root, TastNode):
            self.tree = src_or_root
        else:
            if isinstance(src_or_root, ast.AST):
                astree = src_or_root
            elif isinstance(src_or_root, str):
                astree = ast.parse(src_or_root if isinstance(src_or_root, str) else inspect.getsource(src_or_root))
            else:
                assert False, "Unknown type for Program(...)"
            self.tree = parse.from_ast(astree, var_nts)
        self._safe = None

    def pretty_print(self):
        self.tree.pretty_print()

    def src(self, safe=True, simplify=True):
        def helper(node):
            r = rules.rules[node.rule_num]
            if safe:
                return r.to_python_safe(*(helper(n) for n in node.children))
            else:
                return r.to_python(*(helper(n) for n in node.children))

        ans = helper(self.tree)
        if simplify:
            ans = astor.to_source(
                ast.parse(ans),
                pretty_source=lambda source: ''.join(astor.source_repr.split_lines(source, maxline=120))
            ).strip()  # re.sub(r" *\n *", " ", astor.to_source(ast.parse(ans)).strip())
        return ans

    def size(self):
        def helper(node):
            if hasattr(node, "children"):
                return 1 + sum(helper(k) for k in node.children)
            return 1

        return helper(self.tree)

    def run(self, max_ticks=None, env=None):
        py = self.src(simplify=False)
        self._safe = Safe(max_ticks)
        globals_ = self._safe.globals
        if env:
            globals_.update(env)
        try:
            exec(py, globals_, None)
        except TimeoutError:
            raise Program.TimeoutEvalException
        except Exception as e:
            self._exception = e
            raise Program.EvalException(e)

        return globals_

    def val(self, max_ticks=None, env=None):
        py = self.src(simplify=False)
        assert not py.startswith("def "), "Call Program.run() instead of Program.val()"
        self._safe = Safe(max_ticks)
        globals_ = self._safe.globals
        if env:
            globals_.update(env)
        try:
            self._val = eval(py, globals_, None)
        except TimeoutError:
            raise Program.TimeoutEvalException
        except Exception as e:
            self._val = e
            raise Program.EvalException(e)

        return self._val

    @property
    def clock(self):  # tells you how long it took to run val
        return self._safe.clock

    @property
    def reset_clock(self):
        return self._safe.reset

    def __str__(self):
        return self.src(safe=False, simplify=True)

    def __repr__(self):
        return self.src(safe=False, simplify=True)


class TastNode:
    def __init__(self, rule: rules.Rule, children):  # , derivation=None):
        children = tuple(children) if isinstance(children, (tuple, list)) else (children,)
        # self.derivation = derivation
        assert isinstance(rule, rules.Rule)
        self.rule_num = rule.index
        self.children = children

    @property
    def nt(self):
        return rules.rules[self.rule_num].nt

    @property
    def rule(self):
        return rules.rules[self.rule_num]

    def copy(self):
        return TastNode(self.rule,
                        [(c.copy() if isinstance(c, TastNode) else c) for c in self.children],
                        derivation="copy")

    def __repr__(self):
        # return f"({', '.join(map(repr, [self.rule_num, *self.children]))})"
        return f"({self.rule.name}: {', '.join(map(str, [self.nt, *self.children]))})"

    def __str__(self):
        def helper(node):
            r = rules.rules[node.rule_num]
            if r.name in ["Digits", 'Chars', 'name']:
                return Program(node).src(simplify=False, safe=False)
            ans = r.to_python(*(helper(n) for n in node.children))
            return ans

        return helper(self)

    def pretty_print(self):
        def helper(node):
            r = rules.rules[node.rule_num]
            if r.name in ["Digits", 'Chars', 'name']:
                return Program(node).src(simplify=False, safe=False)
            ans = utils.color_str(f"{r.nt}:")
            ans += r.to_python(*(helper(n) for n in node.children))
            return ans

        print(helper(self))

    def __lt__(self, other):  # implementing < just so they can be part of a sorted structure
        return True

    def __eq__(self, other):
        return (self.nt, self.rule, *self.children) == (other.nt, other.rule, *other.children)

    def src(self, safe=False, simplify=False):
        return Program(self).src(safe=safe, simplify=simplify)


class Safe:
    def __init__(self, max_ticks, int_base=64, random_seed=0):
        self.max_ticks = max_ticks
        self.clock = 0.0
        self.int_base = int_base
        self.random_seed = random_seed
        self._random = random.Random(random_seed)
        self.globals = {"cos": cos, "sin": sin, "pi": pi, "log": log, "exp": exp, "inf": inf,
                        # "randint": self._random.randint, "random": self._random.random,
                        "safe": self, "COPY": lambda x: x, "List": List, "Dict": Dict, "Tuple": Tuple,
                        "Generator": Generator, "Set": Set}

    def reset(self):
        self.clock = 0.0
        self._random.seed(self.random_seed)

    def tick(self, ticks, ret_val=True):
        self.clock += max(0, ticks)
        if self.max_ticks and self.clock > self.max_ticks:
            raise TimeoutError(f"Reached {self.clock} ticks (max is {self.max_ticks})")
        return ret_val

    def int_size(self, a):
        ans = 0
        a = abs(a)
        while a:
            ans += 1
            a >>= self.int_base
        return ans

    def add(self, a, b):
        if hasattr(a, "__len__"):
            self.tick(len(a) + len(b))
        return a + b

    def all(self, gen):
        return all(self.tick(1, i) for i in gen)

    def any(self, gen):
        return any(self.tick(1, i) for i in gen)

    def dot_count(self, a, b):  # TODO: deal better with size of b
        self.tick(len(a))
        if hasattr(b, "__len__"):
            self.tick(len(b))
        return a.count(b)

    def dot_index(self, a, b):  # TODO: deal with size
        ans = a.index(b)
        self.tick(ans)
        return ans

    def dot_issubset(self, a, b):  # TODO: deal with hashing costs?
        assert type(a) == OrderedSet
        self.tick(len(a))
        return a.issubset(b) if isinstance(b, OrderedSet) else a.issubset(self.tick(1, i) for i in b)

    def dot_issuperset(self, a, b):  # TODO: deal with hashing costs?
        assert type(a) == OrderedSet
        if hasattr(b, "__len__"):
            self.tick(len(b))
            return a.issuperset(b)
        else:
            return a.issuperset(i for i in a if self.tick(1))

    def dot_join(self, a, b):
        return a.join([i for i in b if self.tick(len(i))])

    def dot_replace(self, a, b, c):
        self.tick(len(a) + len(a).count(b) * len(c))
        return a.replace(b, c)

    def dot_split(self, *args):
        self.tick(sum(len(x) for x in args))
        return args[0].split(*args[1:])

    def dot_union(self, a, b):  # TODO: deal with hashing costs?
        assert type(a) == type(b) == OrderedSet
        self.tick(len(a) + len(b))
        return a.union(b)

    def in_(self, a, b):
        if isinstance(b, (range, dict, OrderedSet)):
            return a in b
        if isinstance(b, str):
            self.tick(len(b))
            return a in b
        return a in (i for i in b if self.tick(1))

    def list(self, x):
        return [i for i in x if self.tick(1)]
        # return self.tick(len(x), list(x))

    def lshift(self, m, n):
        self.tick(n + self.int_size(m))
        return m << n

    def max(self, x):  # todo: account for comparison time/elt size?
        self.tick(len(x))
        return max(x)

    def min(self, x):  # todo: account for comparison time/elt size?
        self.tick(len(x))
        return min(x)

    def mod(self, a, b):
        ans = a % b
        if isinstance(ans, str):
            self.tick(len(ans))
        return ans

    def mul(self, a, b):
        if isinstance(a, Sequence):
            self.tick(len(a) * b)
        elif isinstance(b, Sequence):
            self.tick(a * len(b))
        elif isinstance(a, int) and isinstance(b, int):
            ans = a * b
            self.tick(self.int_size(ans))
            return ans
        return a * b

    def not_in(self, a, b):
        return not self.in_(a, b)

    def pow(self, m, n):
        if isinstance(m, int) and isinstance(n, int):
            self.tick(self.int_size(m) * (n**2))
        return m ** n

    def set(self, x):  # TODO: deal with hashing costs?
        if hasattr(x, "__len__"):
            self.tick(len(x))
            return OrderedSet(x)
        else:
            return OrderedSet(i for i in x if self.tick(1))

    def sorted(self, x, reverse=False):  # todo: account for comparison time/elt size?
        self.tick(len(x) * log10(10 + len(x)))
        return sorted(x, reverse=reverse)

    def str(self, x):
        ans = str(x)
        self.tick(len(ans))
        assert "object at " not in ans  # avoid nondeterministic strings
        return ans

    def sum(self, gen):
        return sum(self.tick(1, i) for i in gen)

    # def repr(self, x):
    #     ans = repr(x)
    #     self.tick(len(ans))
    #     return ans
