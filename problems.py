"""
A ProblemSet contains one or more (related) Problems. Each Problem has one or more Instances corresponding
a to a different input. A "simple" problem like (lambda x: x + "world" == "Hello world") that has no inputs has
just one instance,

It is important that different ProblemSets do not overlap, in terms of duplicates or answers should not be given
away, like if one had a weighted shortest path problem in one problem set and an unweighted shortest path in another,
they should be combined into a single ProblemSet. This us useful for tests which involve giving away some solutions
but not others.

"""

import inspect
import json
from typing import List, Dict, Callable, Set, Tuple
import random
import os
import sys
import traceback
import time

import utils

# The seed used for randomness is important because if a solver has access to this seed it can cheat and
# reverse-engineer the solutions to some puzzles. Don't share the seed with AI puzzle solvers :-)
_AI_SEED = 12389484322359235125123212243523534510980967133563
DEFAULT_TIMEOUT = 1.0  # seconds



PATH = os.path.join(utils.my_path, "problems/")


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


def gen_type_assertion(var_name: str, ty: type) -> str:
    """
    create code to assert type of var_name is ty

    :param var_name: The variable name, like "x"
    :param ty: str, List[int] , List[List[bool]], etc.
    :return: code that asserts that var_name is of type ty
    """

    tys = type_str(ty)
    vars = [c for c in 'abcdefghijklmnop' if c != var_name][::-1]

    def helper(var_name, tys):
        tys = tys.strip()
        pre_bracket = tys.split("[")[0].lower()  # part before [ (or the entire string if no bracket
        ans = f"type({var_name}) is {pre_bracket}"
        if "[" in tys:
            inside = tys[tys.index("[") + 1:-1]
            new_var = vars.pop()
            if pre_bracket == "list" or pre_bracket == "set":
                inside_check = helper(new_var, inside)
                # if " and " in inside_check:
                #     inside_check = "(" + inside_check + ")"
                ans += f" and all({inside_check} for {new_var} in {var_name})"
            elif pre_bracket == "dict":
                depth = 0
                for i, c in enumerate(inside):
                    if c == "[":
                        depth += 1
                    elif c == "]":
                        depth -= 1
                    elif c == "," and depth == 0:
                        break
                assert depth == 0 and c == ",", "Dict[(expecting comma inside)]"
                key_var = vars.pop()
                key_check = helper(key_var, tys[:i])
                val_check = helper(new_var, tys[i + 1:])
                ans += f" and all({key_check} and {val_check} for {key_var}, {new_var} in {var_name}.items())"
            else:
                assert False, f"Unknown type `{tys}`"
        return ans

    return f"assert {helper(var_name, tys)}, '{var_name} must be of type {tys}'"


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

    def char(self, chars="0123456789abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ!"):
        return self.choice(chars)


class ProblemSet:
    def __init__(self, name, summary=None, path=PATH):
        self.problems = []  # Problem's
        self.summary = summary
        self.name = name
        self.path = PATH
        self._already_tested = None
        # self.np_random = np.random.default_rng([ord(c) for c in seed])

    def add(self, problem):
        self.problems.append(problem)

    def get_filename(self):
        return self.path and (os.path.join(self.path, self.name.split(".")[-1]) + ".json")

    def get_already_tested(self):
        if self._already_tested is None:
            try:
                js = utils.load_json(self.get_filename())
            except:
                js = []
            self._already_tested = {j["sat"]: {sol for sol in j["sols"]} for j in js}
        return self._already_tested

    def save(self):
        obj = []
        for p in self.problems:
            for i in p.instances:
                z = {"name": i.name, "sat": i.src, "sols": i.sol_srcs}
                if p.timeout is not None and p.timeout != 1:
                    z["timeout"] = p.timeout
                obj.append(z)
        filename = self.get_filename()
        if not filename:
            json.dumps(obj)  # for debugging, just to make sure that it can be converted to json
            utils.warning(f"No path, not saving. {[len(p.instances) for p in self.problems]}")
        else:
            try:
                os.makedirs(self.path, exist_ok=True)
                with open(filename, "w") as f:
                    json.dump(obj, f, indent=2)
                solved = sum((1 if i.sol_srcs else 0) for p in self.problems for i in p.instances)
                dur = sum(p.build_time for p in self.problems)
                utils.info(f"Solved {solved:5,}/{sum([len(p.instances) for p in self.problems]):5,} instances of "
                           f"{len(self.problems):3,} probs in {dur:.2f}s => {filename}")
            except FileNotFoundError:
                utils.error(f"***Could not save {filename}, perhaps a path problem?")
                return

        # for e in self.problems[0].instances[:10]:
        #     utils.debug(str(e)[:100])


def get_problems(globs: dict):
    seen = {Problem}  # don't add abstract class Problem
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
            is_prob = isinstance(v, Problem)
        except TypeError:
            is_prob = False
        if is_prob:
            ans.append(v)
        else:
            try:
                is_prob_class = issubclass(v, Problem)
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


def get_type(obj, ignore_errors=False):  # better than type(x) because it can do things like List[int], etc.
    try:
        t = type(obj)
        if t in {int, float, bool, complex, range, str}:
            return t
        assert t in {tuple, list, set, dict}, f"Unacceptable input type '{t}'"
        iterand_types = tuple(get_type(i) for i in obj)
        if t == tuple:
            return Tuple[iterand_types]
        assert len(iterand_types) > 0, "Cannot get type of empty list/set/dict"
        assert len(set(iterand_types)) == 1, "Lists/sets/dicts must be a single type"
        if t == list:
            return List[iterand_types[0]]
        if t == set:
            return Set[iterand_types[0]]
        if t == dict:
            val_types = [get_type(i) for i in obj.values()]
            assert len(set(val_types)) == 1, "Dict values must be a single type"
            return Dict[iterand_types[0], val_types[0]]
    except AssertionError:
        if ignore_errors:
            return None
        raise


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
    def __init__(self, src, name: str, timeout=None):
        self.src = src
        self.name = name  # which Problem template did it come from?
        self.timeout = timeout
        self.sol_srcs = []

    def add_test(self, sol_src, run_test=True):
        """Assert that the solution satisfies the given instance and add the solution to the instance.
        Do a round-trip json encoding/decoding to mimic the actual test and deter strange attacks.
        Ideally this could be done by running a protected process (like in evaluating programming
        contest submissions) but that is much slower so we will only add that later if the AI
        starts misbehaving."""

        if sol_src in self.sol_srcs:  # already added this solution
            return
        if run_test:
            env = dict(List=List, Dict=Dict, Set=Set)
            time0 = time.perf_counter()
            my_exec(sol_src + "\n" + "answer = sol()", env, description=self.name)

            assert env["answer"] is not None, "sol returned None"

            answer = decode(encode(env["answer"]))
            assert answer == env["answer"], "encode/decode round trip failed"

            env2 = dict(answer=answer, List=List, Dict=Dict, Set=Set)  # in case they mucked with env
            my_exec(self.src + "\n" + "assert sat(answer)", env2, description=self.name)
            dur = time.perf_counter() - time0
            if dur > (DEFAULT_TIMEOUT if self.timeout is None else self.timeout * DEFAULT_TIMEOUT):
                utils.warn(f"Took {dur}s to test {self.name} (self.timeout={self.timeout})")
        self.sol_srcs.append(sol_src)


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


class Problem:
    '''Problem is an abstract class for a problem template which builds 1 or more instances.
    Each problem MUST OVERRIDE sat. Examples from templates/hello.py:

    class HelloWorld(Problem):
        """Trivial example, no solutions provided"""

        @staticmethod
        def sat(s: str):
            return s + 'world' == 'Hello world'

    class BackWorlds(Problem):
        """Two solutions, no inputs"""

        @staticmethod
        def sat(s: str):
            return s[::-1] + 'world' == 'Hello world'

        @staticmethod
        def sol():
            return 'olleH '

        @staticmethod
        def sol2():  # solution methods must begin with 'sol'
            return 'Hello '[::-1]


    # With other inputs, the default values of the input are used to generate the first instance.
    # You can run Uncat.get_example() to get the inputs, so you can then run
    # assert Uncat.sat(Uncat.sol(**Uncat.get_example()))
    class Uncat(Problem):
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

    timeout = None  # how long to run in seconds, default is 1.0 seconds if this is None

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
                if c is not DebugProblem:
                    ans.append(c)
                ans.extend(descendents(c))
            return ans

        ans = utils.dedup(descendents(cls))
        names = set()
        for problem in ans:
            name = problem.__name__
            assert name not in names, f"Duplicate problems named `{name}`"
            names.add(name)
        return ans

    @classmethod
    def debug_problems(cls, target_num_instances=None):
        defaults = {"target_num_instances": target_num_instances} if target_num_instances else {}
        debug_problems = DebugProblem.subclass_descendents()
        if debug_problems:
            for P in debug_problems:
                P().debug(**defaults)
            print(f"Problem.Debug problem(s) succeeded: {[p.__name__ for p in debug_problems]}")
            print("Next, remove the `Debug` from these `Problem.Debug` classes")
        else:
            all_problems = Problem.subclass_descendents()
            print("Suggestion for debugging: subclass Problem.Debug to test a single problem.")
            print(f"No Problem.Debug problems found, so testing {len(all_problems):,} problems:")
            for P in all_problems:
                P().test(**defaults)
            print(f"Success on all {len(all_problems):,} problem(s).")

        print("To make the dataset, run make_dataset.py")
        print("See https://github.com/microsoft/PythonProgrammingPuzzles/wiki/How-to-add-a-puzzle for more info.")

    def __init__(self, seed=_AI_SEED):
        self.name = self.__class__.__name__
        assert self.sat is not Problem.sat, f"Must override {self.name}.sat"
        self.sat_src_spec = get_src_spec(self.sat)
        if not self.__doc__ or self.__doc__ == Problem.__doc__:
            self.desc = "<No description/docstring>"
        else:
            self.desc = unindent(self.__doc__)
        self.random = BuilderRandom(f"{seed} | {self.name}")
        self.instances = []
        self._seen_problems = set()
        self._built_target = 0
        self.build_time = None
        self._already_tested = None

        sol_names = [k for k in dir(self) if k.startswith("sol")]
        self.sols = [getattr(self, k) for k in sol_names]
        self.sol_src_specs = [get_src_spec(s) for s in self.sols]

        mro_dict = {}
        for mro in inspect.getmro(self.__class__)[::-1]:
            mro_dict.update(mro.__dict__)
        assert all(isinstance(mro_dict[k], staticmethod) for k in ["sat"] + sol_names), \
            f"{self.name} `sat` and `sol` must be defined with @staticmethod"

        p_spec = self.sat_src_spec[1]

        self.arg_names = p_spec.args
        assert len(self.arg_names) > 0, f"{self.name}.problem() takes no arguments!"
        self.types = p_spec.annotations

        if self.sols:
            s_spec = self.sol_src_specs[0][1]
            assert self.arg_names[1:] == s_spec.args, \
                f"mismatched problem/solution arguments for {self.name}"
            self.types.update(s_spec.annotations)

        assert set(self.arg_names[1:]) == set(self.get_example()), f"Bad {self.name} example"
        self.types.update({v: get_type(x) for v, x in self.get_example().items() if get_type(x, True)})

        for v in self.arg_names:
            assert v in self.types, f"Cannot determine type of `{v}` in {self.name} -- no annotation/_example"

    def build(self, target_num_instances, already_tested={}, max_random_attempts=100, force_trivial_test=False):
        if self._built_target == target_num_instances:
            return

        self.check_for_trivial_solutions(force_trivial_test, already_tested)
        self._already_tested = already_tested
        self._seen_problems = set()
        self._built_target = target_num_instances
        self.random.reseed()
        self._tested = 0
        self.instances = []
        start_time = time.perf_counter()
        self.add(self.get_example())
        if target_num_instances > 1:
            self.gen(target_num_instances)
        while len(self.instances) < target_num_instances:
            n = len(self.instances)
            for _ in range(max_random_attempts):
                self.gen_random()
                if n != len(self.instances):  # added a problem
                    break
            if len(self.instances) == n:  # failed max_random_attempts, give up
                break

        if not self.instances:
            utils.error(f"{self.name} did not generate any problem instances")

        self.build_time = time.perf_counter() - start_time
        self._already_tested = None
        assert self._example_copy == self._example, f"Puzzle {self.name} changed inputs"
        if self._tested:
            utils.info(f"Tested {self._tested} instances of {self.name}")

    def check_for_trivial_solutions(self, force, already_tested):  # check for trivial solutions
        example = self.get_example()
        src = inject_into_src(*self.sat_src_spec, "sat", example, self.types, add_type_assertion=True)
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
        elif type(ans) == list or type(ans) == set:
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
                    tests.extend(type(ans)(p) for p in product(base, repeat=r))
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
        if dur > (self.timeout or 1.0):
            utils.warn(f"Took {dur:.1f}s to test for trivial solutions to `{self.name}`")

    def gen(self, target_num_instances):
        pass

    def gen_random(self):
        pass

    def check_seen_input(self, inp):
        """
        Returns True if the input is a duplicate of a previous puzzle, and also makes sure that the types match
        """
        s = str(inp)
        if s in self._seen_problems:
            return True  # duplicate problem

        self._seen_problems.add(s)

        assert set(inp) == set(self.arg_names[1:]), \
            f"Instance #{len(self.instances)} keys mismatch in {self.name}"
        for v in inp:
            assert get_type(inp[v], ignore_errors=True) in (None, self.types[v]), \
                f"Instance #{len(self.instances)} variable `{v}` type mismatch in {self.name}"

        return False

    def add(self, inp: dict, test=True):

        if self.check_seen_input(inp):
            return  # don't add duplicate problems

        instance = Instance(
            inject_into_src(*self.sat_src_spec, "sat", inp, self.types, add_type_assertion=True),
            f"{self.name}_{len(self.instances)}",
            timeout=self.timeout
        )

        if test:
            for s, (sol_src, sol_spec) in zip(self.sols, self.sol_src_specs):
                sol_src = inject_into_src(sol_src, sol_spec, "sol", inp)
                if instance.src in self._already_tested and sol_src in self._already_tested[instance.src]:
                    instance.add_test(sol_src, run_test=False)
                else:
                    try:
                        instance.add_test(sol_src, run_test=True)
                        self._tested += 1
                    except Exception:  # failed to pass test, rerun test without for debugging with normal exception
                        assert self.sat(s(**inp), **inp) is True, f"Puzzle {self.name} didn't return True on `{inp}`"
                        utils.error("Strange, failed test in exec but passed without exec")
                        raise

        self.instances.append(instance)

    def test(self, target_num_instances=100):
        self.build(target_num_instances, force_trivial_test=True)


class DebugProblem(Problem):
    """
    A useful class for creating a debugging problems. Any DebugProblem will be run before any other problems.
    The program will exit if it successful.
    """

    # def __init__(self):
    #     print("ho")
        # super().__init__()


    def add(self, inp: dict, test=True):
        if self.check_seen_input(inp):
            return  # don't add duplicate problems

        if test:
            var_name = self.sat_src_spec[1].args[0]
            type_assertion_str = gen_type_assertion(var_name, self.types[var_name])
            for s in self.sols:
                answer = s(**inp)
                exec(type_assertion_str, {var_name: answer})
                assert self.sat(answer, **inp) is True, f"Puzzle {self.name} didn't return True on `{inp}`"

        self.instances.append(("DEBUG TEST", bool(test and self.sols)))  # for counting purposes

    def debug(self, target_num_instances=10000):
        print(f"Debugging {self.name}")
        self.build(target_num_instances, force_trivial_test=True)
        solved = sum(i[1] for i in self.instances)
        dur = self.build_time
        utils.info(f"Tested {solved:,}/{len(self.instances):,} instances of "
                   f"({self.name}: DebugProblem) in {dur:0.2f}s")


Problem.Debug = DebugProblem


def remove_type_assertion(src: str):
    lines = src.split("\n")
    assert lines[1].strip().startswith("assert type")
    lines.pop(1)
    return "\n".join(lines)


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


def inject_into_src(src, spec, new_function_name=None, defaults={}, types={}, add_type_assertion=False):
    if spec.defaults:  # combine defaults, with defaults over-riding spec.defaults
        defaults = {**dict(zip(spec.args[-len(spec.defaults):], spec.defaults)), **defaults}
    assert all(var in spec.args for var in defaults), f"Defaults {defaults} not all in spec.args"

    for v, t in spec.annotations.items():
        assert v not in types or types[v] == t, f"Annotation mismatch in {src}"

    types = {**spec.annotations, **types}

    func_name = (new_function_name or src[4:src.index('(')])

    arg_st = ", ".join([var +
                        (f": {type_str(types[var])}" if var in types else "") +
                        (f"={utils.stringify(defaults[var])}" if var in defaults else "")
                        for var in spec.args])

    ans = f'def {func_name}({arg_st}):'

    if add_type_assertion:
        assert func_name == "sat"
        indent = min([len(line) - len(line.lstrip(" "))
                      for line in src.splitlines() if line.strip() and line.startswith(" ")])
        ans += "\n" + " " * indent + gen_type_assertion(spec.args[0], types[spec.args[0]])

    ans += src[src.index("):") + 2:]

    return ans


def get_func_name(src):
    assert src.startswith("def ")
    return src[4:src.index("(")]


def save_readme(problem_sets, filename=os.path.join(PATH, "README.md")):
    top = """# Python Programming Puzzles: dataset summary
This document summarizes the dataset stored in .json files.
Each .json file contains a number of related problems with one or more puzzles each. (Each puzzle in the json 
files contains an assert statement on its first line ensuring that the input is of the correct type---these
assertions have been removed below for readability.) The only import required for puzzles is: 
```from typing import List, Set, Dict```    



## Files:

{}

----

"""



    table = ""
    content = ""
    tot_probs = 0
    tot_instances = 0
    for ps in problem_sets:
        section = ""
        sec_name = ps.name.split(".")[-1]
        section += f"## {sec_name}\n\n"
        section += f"{ps.summary}\n\n"
        n = len(ps.problems)
        link = f"[{sec_name}](#{sec_name.lower().replace(' ', '-')})"
        section += "[^ Top](#files)\n\n"
        n_instances = sum(len(p.instances) for p in ps.problems)
        tot_probs += len(ps.problems)
        tot_instances += n_instances
        table += f"- [{sec_name} ({len(ps.problems):,} problems, {n_instances:,} instances)](#{sec_name.lower().replace(' ', '-')})\n"
        for i, problem in enumerate(ps.problems):
            section += f"### {problem.name}\n({link} {i + 1:,}/{n:,})\n\n"
            section += f"**Description:**\n{problem.desc}\n\n"
            section += f"**Problem:**\n\n```python\n{remove_type_assertion(problem.instances[0].src)}\n```\n"
            if len(problem.instances[0].sol_srcs) > 0:
                section += "<details><summary><strong>Reveal solution(s):</strong></summary>\n\n"
                for sol in problem.instances[0].sol_srcs:
                    section += f"```python\n{sol}\n```\n\n"
                section += "</details>\n\n"

            # section += f"[^ Back](#{sec_name.lower().replace(' ', '-')})\n\n"
            # Replaced with link in the header

        section += "[^^ Top](#files)\n"
        content += section

    table += f"\nTotal ({tot_probs:,} problems, {tot_instances:,} instances)\n"


    content = top.format(table) + content

    with open(filename, "w", encoding='utf8') as f:
        f.write(content)
