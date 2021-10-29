"""
Each PuzzleGenerator has one or more Instances corresponding a to a different input. A "simple" problem like
(lambda x: x + "world" == "Hello world") that has no inputs has just one instance.
"""

import inspect
import json
from typing import List, Callable, Dict, Set
import random
import re
import sys
import traceback
import time

import utils

# The seed used for randomness is important because if a solver has access to this seed it can cheat and
# reverse-engineer the solutions to some puzzles. Don't share the seed with AI puzzle solvers :-)
_AI_SEED = 12389484322359235125123212243523534510980967133563
DEFAULT_TIMEOUT = 1.0  # seconds


def type_check(typ, obj):
    """
    Checks the object is the correct type. Supports only bool, int, float, str, and (possibly nested) lists of these
    """
    type_s = type_str(typ)  # convert to string if necessary

    nest_depth = type_s.count("List")
    assert type_s.count("[") == nest_depth, "type_check only supports List for now, no Sets, Dicts, Tuples, ..."

    assert type_s.startswith("List[" * nest_depth) and type_s.endswith("]" * nest_depth)
    base_type = {"bool": bool, "int": int, "float": float, "str": str}[type_s[5 * nest_depth:len(type_s) - nest_depth]]

    def helper(depth, o):
        if depth == 0:
            return type(o) is base_type
        else:
            return type(o) is list and all(helper(depth - 1, i) for i in o)

    return helper(nest_depth, obj)


def test_puzzle(f: callable, x, ans_type: str):
    """Checks if x is of the correct type and makes f return True (literally True, not an integer or whatever)

    :param f: Puzzle
    :param x: candidate answer
    :param ans_tye:
    :return:
    """
    if not type_check(x, ans_type):
        raise TypeError
    return f(x) is True


class InterpreterError(Exception): pass


def my_exec(cmd, globals=None, locals=None, description='source string'):
    """
    https://stackoverflow.com/questions/28836078/how-to-get-the-line-number-of-an-error-from-exec-or-execfile-in-python
    """
    try:
        exec(cmd, globals, locals)
    except SyntaxError as err:
        error_class = err.__class__.__name__
        detail = err.args[0] if err.args else ""
        line_number = err.lineno
    except Exception as err:
        error_class = err.__class__.__name__
        detail = err.args[0] if err.args else ""
        cl, exc, tb = sys.exc_info()
        line_number = traceback.extract_tb(tb)[-1][1]
    else:
        return

    cmd_str = "\n".join([f"{i + 1}: {x}" for i, x in enumerate(cmd.split("\n"))])
    raise InterpreterError("%s at line %d of %s: %s\n%s" % (error_class, line_number, description, detail, cmd_str))


def type_str(ty: type) -> str:
    """
    Convert type ty to string.

    :param ty: str, typing.List[int] , typing.List[typing.List[bool]], etc.
    :return: string form of type, "str", "List[int]" , "List[List[bool]]", etc.
    """
    type_str = str(ty).replace("typing.", "")
    return type_str[8:-2] if type_str.startswith("<class '") else type_str


def gen_dump_code(var_name: str, ty: type) -> str:
    """
    create code to output an object of type ty as a string

    :param var_name: The variable name, like "x"
    :param ty: str, typing.List[int] , typing.List[typing.List[bool]], etc.
    :return: code that writes the variable to standard out as a json object
    """

    tys = type_str(ty)
    if tys.startswith("Set["):
        return "print(json.dumps({k : 1 for k in " + var_name + "})) # write sets as dictionaries\n"
    return f"print(json.dumps({var_name}))\n"


def gen_load_code(var_name: str, ty: type) -> str:
    """
    create code to load an object of type ty as a string

    :param var_name: The variable name, like "x"
    :param ty: str, typing.List[int] , typing.List[typing.List[bool]], etc.
    :return: code that reads the variable from stdin as a json object
    """

    tys = type_str(ty)

    if tys.startswith("Set["):
        assert tys.endswith("]")
        inside = tys[4:-1]
        ans = f"{var_name} = set(json.load(sys.stdin))) # convert set (stored as json dictionary)"
        assertions = [f"all(isinstance(x, {inside}) for x in {var_name})"]
    else:
        ans = f"{var_name} = json.load(sys.stdin)"
        num_lists = tys.count("List[")
        assert tys.startswith("List[" * num_lists) and tys.endswith("]" * num_lists)
        inside = tys[5 * num_lists: len(tys) - num_lists]
        if num_lists == 0:
            assertions = [f"isinstance({var_name}, {inside})"]
        else:
            assertions = [f"isinstance({var_name}, list)"]
            if num_lists == 1:
                assertions.append(f"all(isinstance(x, {inside}) for x in {var_name})")
            else:
                assertions.append(f"all(isinstance(x, list) for x in {var_name})")
                if num_lists == 2:
                    assertions.append(f"all(isinstance(y, {inside}) for x in {var_name} for y in x)")
                elif num_lists == 3:
                    assertions += [f"all(isinstance(y, list) for x in {var_name} for y in x)",
                                   f"all(isinstance(z, {inside}) for x in {var_name} for y in x for z in y)"]
                else:
                    assert False, f'Unknown type {tys}'

    assert inside in ["int", "float", "bool", "str"], f'Unknown type {tys}'
    return ans + "\n\n" + "\n".join(f"assert {a}, 'Type error: expecting `{tys}`'" for a in assertions)


def add_preamble(src):
    preamble = []
    types = []
    if "List[" in src:
        types.append("List")
    if "Set[" in src:
        types.append("Set")
    if types:
        preamble.append(f"from typing import {','.join(types)}")
    if "json." in src:
        preamble.append("import json")
    if "sys." in src:
        preamble.append("import sys")

    return "\n".join(preamble) + "\n" * 3 + src if preamble else src


def gen_prob_code(var_name: str, var_type: type, prob_src: str, inputs: str):
    s = f"""{prob_src}

{gen_load_code(var_name, var_type)}

inputs = {inputs}

assert problem({var_name}, **inputs)

print("Success!")
"""
    # import inspect
    # print(inspect.getsource(problem))

    return add_preamble(s)


def gen_sol_code(var_name: str, var_type: type, sol_src: str, inputs: str):
    s = f"""{sol_src}

inputs = {inputs}

{var_name} = solution(**inputs)

{gen_dump_code(var_name, var_type)}
"""

    return add_preamble(s)


class BuilderRandom(random.Random):
    """Adds extra random functions useful for building instances."""

    def __init__(self, seed=None):
        self._init_seed = seed
        super().__init__(seed)

    def reseed(self):
        self.seed(self._init_seed)

    def pseudo_word(self, min_len=1, max_len=20):
        w = "".join(self.choice(["text", "th", "ch", "qu", *"bcdfghjklmnprstvwxz"]) + self.choice("aeiyou")
                    for _ in range(1 + max_len // 2))
        return w[:self.randrange(min_len, max_len + 1)]

    def heavy_tail_float(self, lower=-1000.0, upper=1000.0, median_dev=1.0):  # heavy tailed distribution
        mean = (lower + upper) / 2.0
        trunc = (upper - lower) / 2.0
        while True:
            r = (self.random() ** (-2) - 1) / 3
            if self.randrange(2):
                r = -r
            x = mean - median_dev * r
            if abs(x - mean) <= trunc:
                return x

    def char(self,
             chars="0123456789abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ,.:|/;?[]<>-=()+*&^%$#@!"):
        return self.choice(chars)

    def string(self, min_len=1, max_len=20):
        length = self.randrange(min_len, max_len + 1)
        return "".join(self.char() for _ in range(length))


def get_problems(globs: dict):
    seen = {PuzzleGenerator}  # don't add abstract class PuzzleGenerator
    ans = []
    for v in globs.values():
        try:
            if v in seen:
                continue
            else:
                seen.add(v)
        except TypeError:
            continue
        try:
            is_prob = isinstance(v, PuzzleGenerator)
        except TypeError:
            is_prob = False
        if is_prob:
            ans.append(v)
        else:
            try:
                is_prob_class = issubclass(v, PuzzleGenerator)
            except TypeError:
                is_prob_class = False
            if is_prob_class:
                ans.append(v())
    return ans


def deep_copy(obj):
    t = type(obj)
    if t in {tuple, list, set}:
        return t(deep_copy(x) for x in obj)
    if t == dict:
        return {k: deep_copy(v) for k, v in obj.items()}
    return obj


def same_types(obj1, obj2):
    """
    Recursively check that obj1 and obj2 are of the same types.
    Better than type(obj1) == type(obj2) because it recursively checks inside lists, sets, dicts, and tuples
    """
    t = type(obj1)
    if t is not type(obj2):
        return False
    if t in {list, set, dict}:
        for iterables in ([(obj1, obj2), (obj1.values(), obj2.values())] if t is dict else [(obj1, obj2)]):
            lst = [i for o in iterables for i in o]
            if not all(same_types(lst[0], o) for o in lst[1:]):
                return False
    if t is tuple:
        return len(obj1) == len(obj2) and all(same_types(o1, o2) for o1, o2 in zip(obj1, obj2))
    return True


# def test_same_types():
#     assert same_types(1, 2)
#     assert same_types({1:[]}, {})
#     assert same_types({1:[2,3]}, {4:[5,6]})
#     assert not same_types(True, 1)
#     assert not same_types(1, 2.0)
#     assert not same_types(1, 2.0)
#     assert not same_types({1:[2,3]}, {4:[5.,6.]})
#     assert not same_types({1:[2,3], 3:[5.]}, {})
#
# test_same_types()

def homogeneous_type(obj):
    """
    Checks that the type is "homogeneous" in that all lists are of objects of the same type, etc.
    """
    return same_types(obj, obj)


# def test_homogeneous_types():
#     assert homogeneous_type(1)
#     assert homogeneous_type([1, 2, 3])
#     assert homogeneous_type([[[]], [[]]], [[3], [4]])
#     assert homogeneous_type({})
#     assert not homogeneous_type([1, 2, 3.3])
#     assert homogeneous_type([[[]], [[], [4.]]], [[3], []])
#
# test_homogeneous_types()


def decode(st: str):  # small modifications to make json roundtrip work
    def helper(obj):
        if type(obj) in [int, str, float, bool]:
            return obj
        if type(obj) == list:
            if len(obj) == 2 and obj[0] == "__SET__:":
                return set(helper(obj[1]))
            return [helper(i) for i in obj]
        if type(obj) == dict:
            return {json.loads(k): helper(v) for k, v in obj.items()}
        assert False, f"Unexpected type {type(obj)}"

    return helper(json.loads(st))


def encode(obj):  # small modifications to make json roundtrip work
    def helper(x):  # encodes sets in a json-friendly fashion
        if type(x) in [int, str, float, bool]:
            return x
        if type(x) == list:
            return [helper(i) for i in x]
        if type(x) == set:
            return ["__SET__:", helper({i: 0 for i in x})]
        if type(x) == dict:
            return {json.dumps(k): helper(v) for k, v in x.items()}
        assert False, f"Unexpected type {type(x)}"

    return json.dumps(helper(obj))


class Instance:
    def __init__(self, name: str, src: str, sol_header: str, sol_bodies: List[str], multiplier: float):
        self.name = name  # instance name
        self.src = src
        self.sol_header = sol_header
        self.sol_bodies = sol_bodies
        self.multiplier = multiplier


def unindent(docstr):
    lines = [line for line in docstr.strip().split("\n")]
    de_indent = None
    for i in range(1, len(lines)):
        line = lines[i]
        if de_indent is None and line.strip():
            de_indent = len(line) - len(line.lstrip(" "))
        if de_indent and len(line) > de_indent:
            assert not line[:de_indent].strip(), f"Weird indentation in docstring:\n{docstr}"
            lines[i] = line[de_indent:]
    return "\n".join(lines)


def get_body(function_src):
    match = re.search(r"\)\s*:(.*)\n", function_src)
    assert match and (match.group(1).replace(" ", "") == ""), \
        f"Bad solution header for, maybe move to next line:\n\n{match.group(1)}\n\nin:\n\n{function_src}"
    return function_src[match.end():]


class PuzzleGenerator:
    '''PuzzleGenerator is an abstract class for a puzzle generator which builds 1 or more instances.
    Each problem MUST OVERRIDE sat. Examples from templates/hello.py:

    class HelloWorld(PuzzleGenerator):
        """Trivial example, no solutions provided"""

        @staticmethod
        def sat(s: str):
            return s + 'world' == 'Hello world'

    class BackWorlds(PuzzleGenerator):
        """Two solutions, no inputs"""

        @staticmethod
        def sat(s: str):
            return s[::-1] + 'world' == 'Hello world'

        @staticmethod
        def sol():
            return 'olleH '

        @staticmethod
        def sol2():
            # solution methods must begin with 'sol'
            return 'Hello '[::-1]


    # With other inputs, the default values of the input are used to generate the first instance.
    # You can run Uncat.get_example() to get the inputs, so you can then run
    # assert Uncat.sat(Uncat.sol(**Uncat.get_example()))
    class Uncat(PuzzleGenerator):
        """Simple example with inputs."""

        @staticmethod
        def sat(st: str, a='world', b='Hello world'):
            return st + a == b

        @staticmethod
        def sol(a, b):
            return b[:len(b)-len(a)]

        def gen_random(self):
            b = self.random.pseudo_word()
            a = b[self.random.randrange(len(b)+1):]
            self.add({"a": a, "b": b})
        '''


    DEBUG = False  # DEBUG = True while making a puzzle makes it run before any other problems
    skip_example = False # skip the example in the default arguments to sat, so it's not the first instance

    @staticmethod
    def sat(ans, *other_inputs):  # must override
        raise NotImplementedError

    @classmethod
    def get_example(cls):
        if not hasattr(cls, "_example"):
            p_spec = inspect.getfullargspec(cls.sat)
            if p_spec.defaults:
                cls._example = dict(zip(p_spec.args[-len(p_spec.defaults):], p_spec.defaults))
            else:
                cls._example = {}
            cls._example_copy = deep_copy(cls._example)
        return cls._example

    @classmethod
    def subclass_descendents(cls):  # finds all problems
        def descendents(cls):
            ans = []
            for c in cls.__subclasses__():
                ans.append(c)
                ans.extend(descendents(c))
            return ans

        ans = utils.dedup(descendents(cls))
        # ans = [cls for cls in ans if cls.sat is not PuzzleGenerator.sat]
        names = set()
        for problem in ans:
            name = problem.__name__
            assert name not in names, f"Duplicate problems named `{name}`"
            names.add(name)

        return ans

    @classmethod
    def debug_problems(cls, target_num_instances=None):
        defaults = {"target_num_instances": target_num_instances} if target_num_instances else {}
        all_gens = PuzzleGenerator.subclass_descendents()
        debug_problems = [cls for cls in all_gens if cls.DEBUG]
        if debug_problems:
            for P in debug_problems:
                P().debug(**defaults)
            print(f"PuzzleGenerator.DEBUG=True problem(s) succeeded: {[p.__name__ for p in debug_problems]}")
            print("Next, remove `DEBUG=True` from these classes")
        else:
            print("Suggestion for debugging: set DEBUG=True on PuzzleGenerator classes to test a single class.")
            print(f"No DEBUG=True PuzzleGenerator classes found, so testing {len(all_gens):,} classes:")
            for P in all_gens:
                P().test(**defaults)
            print(f"Success on all {len(all_gens):,} problem(s).")

        print("To make the dataset, run make_dataset.py")
        print("See https://github.com/microsoft/PythonProgrammingPuzzles/wiki/How-to-add-a-puzzle for more info.")

    def __init__(self):
        self.name = self.__class__.__name__
        assert self.sat is not PuzzleGenerator.sat, f"Must override {self.name}.sat"
        self.sat_src, sat_spec = get_src_spec(self.sat)
        self.docstring = utils.get_docstring(self.sat_src)
        self.sat_src = utils.remove_docstring(self.sat_src)
        assert len(sat_spec.args) > 0, f"{self.name}.sat() takes no arguments!"
        self.ans_name, *self.arg_names = sat_spec.args
        assert self.ans_name in sat_spec.annotations, f"Missing type hint for {self.name}.sat({self.ans_name}: ???"
        self.ans_type = type_str(sat_spec.annotations[self.ans_name])
        assert self.ans_type.replace("List[", "").replace("]", "") in "bool float int str".split(), \
            f"Answer type for {self.name} must be bool/int/float/str or Lists (or Lists of Lists etc.) of those"
        if not self.__doc__ or self.__doc__ == PuzzleGenerator.__doc__:
            self.desc = ""
        else:
            self.desc = unindent(self.__doc__)

        self.random = BuilderRandom(seed=self.name)

        # these are created at Build time
        self._seen_inputs = None
        self._inputs = None  # inputs to test during build
        self.instances = None

        sol_names = [k for k in dir(self) if k.startswith("sol")]
        self.sols = [getattr(self, k) for k in sol_names]
        self.sol_bodies = []
        for sol in self.sols:  # check solution headers and extract bodies
            sol_src, sol_spec = get_src_spec(sol)
            assert self.arg_names == sol_spec.args, f"mismatched problem/solution arguments for {self.name}"
            assert not sol_spec.defaults, f"Don't set default parameter values for {self.name}.sol -- we'll do it"
            self.sol_bodies.append(get_body(sol_src))

        assert set(self.arg_names) == set(self.get_example()), f"Bad {self.name} example"
        for v, val in self.get_example().items():
            if not homogeneous_type(val):
                utils.warn(f"Non-homogeneous type for example var {v} in {self.name}")

        # check that sat and sol's are @staticmethod's
        mro_dict = {}
        for mro in inspect.getmro(self.__class__)[::-1]:
            mro_dict.update(mro.__dict__)
        assert all(isinstance(mro_dict[k], staticmethod) for k in ["sat"] + sol_names), \
            f"{self.name} `sat` and `sol` must be defined with @staticmethod"

    def test_input(self, name, inp, test: bool, multiplier: float, already_tested={}):
        """Check if the input has been tested already. If not, assert that the solution(s) satisfy the given
        inputs. Do a round-trip json encoding/decoding to mimic the actual test.
        Ideally this could be done by running a protected process (like in evaluating programming
        contest submissions) but that is much slower. Since this is a test we authored presumably it has
        no evil code.

        Returns the new instance and number of solutions actually tested (that were not in cache)"""
        num_tested = 0
        new_sat_src = create_sat(self.sat_src, self.ans_name, self.ans_type, self.arg_names, inp)
        sol_header = create_sol_header(inp)

        instance = Instance(
            name,
            new_sat_src,
            sol_header,
            self.sol_bodies if test else [],
            multiplier
        )

        for sol_body, sol_func in zip(instance.sol_bodies, self.sols):
            if new_sat_src in already_tested and sol_body in already_tested[new_sat_src]:
                continue  # skip
            num_tested += 1
            time0 = time.perf_counter()
            env = dict(List=List)
            if self.DEBUG:  # In debug mode just run the darn tests
                answer = sol_func(**inp)
            else:
                try:
                    my_exec(
                        instance.sol_header + " \n" + sol_body + "\n" + "answer = sol()",
                        env,
                        description=instance.name
                    )
                except Exception:
                    sol_func(**inp)
                    utils.error("Strange, failed test in exec but passed without exec")
                    raise

                answer = env["answer"]

            assert answer is not None, "sol returned None"
            assert type_check(self.ans_type, answer), f"Solution returned wrong type for {self.name}"

            if self.DEBUG:
                assert self.sat(answer, **inp) is True, f"Puzzle {self.name} didn't return True on `{inp}`"
            else:
                assert answer == decode(encode(answer))
                try:
                    env2 = dict(answer=answer, List=List)  # in case we screwed up env
                    my_exec(instance.src + "\n" + "assert sat(answer) is True", env2, description=self.name)
                except Exception:
                    assert self.sat(answer, **inp) is True, \
                        f"Puzzle {instance.name} didn't return True on `{inp}`"
                    utils.error("Strange, failed test in exec but passed without exec")
                    raise

            dur = time.perf_counter() - time0
            if dur > DEFAULT_TIMEOUT * multiplier:
                utils.warn(f"Took {dur}s to test {instance.name} (multiplier={multiplier})")

        return instance, num_tested

    def num_generated_so_far(self):
        """
        Call this function during gen/gen_random to see how many unique puzzle instances have been generated so far.
        """
        return len(self._inputs)

    def build(self, target_num_instances, already_tested={}, max_random_attempts=100, force_trivial_test=False):
        self.check_for_trivial_solutions(force_trivial_test, already_tested)
        self._seen_inputs = set()
        self._inputs = []  # for recording the inputs to test
        self.random.reseed()
        start_time = time.perf_counter()
        if not self.skip_example:
            self.add(self.get_example())

        if target_num_instances > len(self._inputs):
            self.gen(target_num_instances - len(self._inputs))

        while len(self._inputs) < target_num_instances:
            n = len(self._inputs)
            for _ in range(max_random_attempts):
                self.gen_random()
                if n != len(self._inputs):  # added a problem
                    assert len(self._inputs) == n + 1, f"{self.name}.gen_random() generated more than one instance"
                    break

            if len(self._inputs) == n:  # failed max_random_attempts, give up
                break

        self._inputs = self._inputs[:target_num_instances]

        num_tested = 0
        self.instances = []

        for inp, test, multiplier in self._inputs:
            instance, n = self.test_input(f"{self.name}:{len(self.instances)}", inp, test, multiplier, already_tested)
            self.instances.append(instance)
            num_tested += n
        build_time = time.perf_counter() - start_time

        assert self._example_copy == self._example, f"Puzzle {self.name} changed inputs"

        if num_tested:
            utils.info(f"Actually tested {num_tested:,}/{len(self.instances):,} "
                       f"instances of {self.name} in {build_time:.1f}s")

        self._seen_inputs = None
        self._inputs = None  # for recording the inputs to test

    def check_for_trivial_solutions(self, force, already_tested):  # check for trivial solutions
        example = self.get_example()
        src = create_sat(self.sat_src, self.ans_name, self.ans_type, self.arg_names, example)
        if (not force and src in already_tested) or not hasattr(self, "sol"):
            return
        utils.info(f"Checking for trivial solutions to {self.name}")
        time0 = time.perf_counter()
        ans = self.sol(**example)
        if type(ans) == int:
            if ans in range(-1000, 1000):
                tests = [ans]
            else:
                tests = []
        elif type(ans) == str:
            if len(ans) <= 1:
                tests = [ans]
            else:
                tests = ["cat", "dog", "aa", "ab", "foo", "bar", "baz"]
        elif type(ans) == float:
            tests = [-100.0, -10.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]
        elif type(ans) == bool:
            tests = [True, False]
        elif type(ans) == list:
            if len(ans) == 0:
                tests = [ans]
            else:
                el = list(ans)[0]
                if type(el) == int:
                    base = list(range(-3, 4))
                elif type(el) == str:
                    base = ["a", "b", "foo", "bar", "baz"]
                elif type(el) == bool:
                    base = [True, False]
                elif type(el) == float:
                    base = [-1.0, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0]
                else:
                    # print(f"Can't create trivial instances fitting pattern `{ans}`"[:1000])
                    base = []
                from itertools import product
                tests = []
                for r in range(6):
                    tests.extend(list(p) for p in product(base, repeat=r))
        else:
            print(f"Can't check for types, unexpected type `{type(ans)}`")
            tests = []
        for t in tests:
            try:
                assert self.sat(t, **example)
            except:
                continue
            utils.warn(f"`{self.name}` in file `{self.__module__.split('.')[-1]}` "
                       f"has trivial solution `{t}`")
            break
        dur = time.perf_counter() - time0
        if dur > 1.0:  # warn if above one second
            utils.warn(f"Took {dur:.1f}s to test for trivial solutions to `{self.name}`")

    def gen(self, target_num_instances):
        pass

    def gen_random(self):
        pass

    def add(self, inp: dict, test=True, multiplier=1.0):
        s = str(inp)
        if s in self._seen_inputs:
            return  # duplicate problem
        else:
            self._seen_inputs.add(s)

        assert set(inp) == set(self.arg_names), f"Instance #{self.num_generated_so_far()} keys mismatch in {self.name}"
        example = self.get_example()
        for k in inp:
            v1, v2 = example[k], inp[k]
            if not same_types(v1, v2):
                utils.warn(f"Instance #{self.num_generated_so_far()} variable `{k}` type mismatch in {self.name}")

        self._inputs.append((inp, test, multiplier))


    def debug(self, target_num_instances=10000):
        print(f"Debugging {self.name}")
        old_debug = self.DEBUG
        self.DEBUG = True
        self.build(target_num_instances, force_trivial_test=True)
        self.DEBUG = old_debug



def get_src_spec(f: Callable):
    try:
        src = inspect.getsource(f)
        spec = inspect.getfullargspec(f)
    except OSError:
        utils.error("Cannot use inspect, happens in some interpreters... Try running in ipython.")
        raise

    de_indent = min([len(line) - len(line.lstrip(" ")) for line in src.splitlines() if line.strip()])
    src = "\n".join([line[de_indent:] for line in src.splitlines()]).strip()

    if src.startswith("@staticmethod"):
        src = src[len("@staticmethod"):].strip()
    assert src.startswith("def ")
    return src, spec


def create_sol_header(defaults, function_name="sol"):
    # could add types here if needed
    ans = f"def {function_name}("
    ans += ", ".join(f'{var}={utils.stringify(default)}' for var, default in defaults.items())
    ans += "):"
    return ans


def create_sat(src, ans_name, ans_type, args, defaults, function_name="sat"):
    assert set(defaults) == set(args), f"Add error: defaults don't match args {args} in {src}"
    ans = f"def {function_name}({ans_name}: {ans_type}"
    if args:
        ans += ", " + ", ".join(f"{v_name}={utils.stringify(defaults[v_name])}" for v_name in args)
    ans += "):\n"
    ans += get_body(src)
    return ans


def get_func_name(src):
    assert src.startswith("def ")
    return src[4:src.index("(")]
