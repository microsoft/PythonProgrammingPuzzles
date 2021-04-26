"""
A ProblemSet contains one or more (related) Problems. Each Problem has one or more Instances corresponding
a to a different input. A "simple" problem like (lambda x: x + "world" == "Hello world") that has no inputs has
just one instance,

It is important that different ProblemSets do not overlap, in terms of duplicates or answers should not be given
away, like if one had a weighted shortest path problem in one problem set and an unweighted shortest path in another,
they should be combined into a single ProblemSet. This us useful for tests which involve giving away some solutions
but not others.

"""

import abc
import inspect
import json
from typing import List, Dict, Callable, Set, Tuple
import random
import os
import sys
import traceback
import time

import utils

# if os.environ['PYTHONHASHSEED'] != '0':
#     utils.warn("Environment variable PYTHONHASHSEED should be set to 0 to make executions deterministic")


problem_registry = {}
_seen_names = set()


def register(problem_class):
    global problem_registry

    module = inspect.getmodule(problem_class)
    name = module.__name__.split('templates.')[-1]

    if name not in problem_registry:
        summary = module.__doc__ or name
        problem_registry[name] = {"classes": [], "name": name, "summary": summary}

    assert problem_class not in problem_registry[name]["classes"], "double registry"
    problem_registry[name]["classes"].append(problem_class)

    p_name = problem_class.__name__
    assert p_name not in _seen_names, f"Duplicate class name `{p_name}`"
    _seen_names.add(p_name)

    return problem_class


PATH = os.path.join(utils.my_path, "problems/")

_secret_seed = None


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


def get_seed_str(filename, summary):
    global _secret_seed
    if _secret_seed is None:
        secret_seed_path = os.path.join(utils.my_path, "_secret_seed.txt")
        try:
            with open(secret_seed_path, "r") as f:
                _secret_seed = f.read()
        except FileNotFoundError:
            utils.warning(f"Couldn't find `{secret_seed_path}`")
            _secret_seed = "92354683922359"
    return f"{filename} | {_secret_seed} | {summary}"


class ProblemSet:
    def __init__(self, name, summary=None):
        self.problems = []  # Problem's
        self.summary = summary
        self.name = name
        # self.np_random = np.random.default_rng([ord(c) for c in seed])

    def add(self, problem):
        self.problems.append(problem)

    def test(self, target_per_problem=100):
        for c in self.problems:
            c().test(target_per_problem)

    def save(self, path=PATH):
        obj = []
        for p in self.problems:
            for i in p.instances:
                z = {"name": i.name, "sat": i.src, "sols": i.sol_srcs}
                if p.timeout is not None and p.timeout != 1:
                    z["timeout"] = p.timeout
                obj.append(z)
        if not path:
            json.dumps(obj)  # for debugging, just to make sure that it can be converted to json
            utils.warning(f"No path, not saving. {[len(p.instances) for p in self.problems]}")
        else:
            try:
                os.makedirs(path, exist_ok=True)
                filename = os.path.join(path, self.name.split(".")[-1]) + ".json"
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
    def __init__(self, src, name=None):
        self.src = src
        self.name = name  # which Problem template did it come from?
        self.sol_srcs = []

    def test(self, sol_src):
        """Assert that the solution satisfies the given instance and add the solution to the instance.
        Do a round-trip json encoding/decoding to mimic the actual test and deter strange attacks.
        Ideally this could be done by running a protected process (like in evaluating programming
        contest submissions) but that is much slower so we will only add that later if the AI
        starts misbehaving."""

        if sol_src in self.sol_srcs:  # already tested/added this solution
            return
        env = dict(List=List, Dict=Dict, Set=Set)
        my_exec(sol_src + "\n" + "answer = sol()", env, description=self.name)

        assert env["answer"] is not None, "sol returned None"

        answer = decode(encode(env["answer"]))
        assert answer == env["answer"], "encode/decode round trip failed"

        env2 = dict(answer=answer, List=List, Dict=Dict, Set=Set)  # in case they mucked with env
        my_exec(self.src + "\n" + "assert sat(answer)", env2, description=self.name)
        self.sol_srcs.append(sol_src)


def unindent(docstr):
    lines = [line for line in docstr.strip().split("\n")]
    de_indent = None
    for i in range(1, len(lines)):
        line = lines[i]
        if de_indent is None and line.strip():
            de_indent = len(line) - len(line.lstrip(" "))
        if de_indent and len(line) > de_indent:
            assert not line[:de_indent].strip(), "Weird indentation in docstring"
            lines[i] = line[de_indent:]
    return "\n".join(lines)


class Problem(abc.ABC):
    '''Problem is an abstract class for a problem template which builds 1 or more instances.
    Each problem MUST OVERRIDE sat. Examples from templates/hello.py:

    @register
    class HelloWorld(Problem):
        """Trivial example, no solutions provided"""

        @staticmethod
        def sat(s: str):
            return s + 'world' == 'Hello world'

    @register
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
    @register
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

    @staticmethod
    @abc.abstractmethod
    def sat(ans, *other_inputs):  # must override
        raise NotImplementedError

    @classmethod
    def get_example(cls):
        p_spec = inspect.getfullargspec(cls.sat)
        if p_spec.defaults:
            return dict(zip(p_spec.args[-len(p_spec.defaults):], p_spec.defaults))
        return {}

    MISMATCH = "'TYPE MISMATCH'"

    timeout = None  # how much longer than usual can sat run?

    def __init__(self, seed=None):
        self.name = self.__class__.__name__
        if not self.__doc__ or self.__doc__ == Problem.__doc__:
            self.desc = "<No description/docstring>"
        else:
            self.desc = unindent(self.__doc__)
        self.random = BuilderRandom(f"{seed} | {self.name} | {self.desc}")
        self.instances = []
        self._seen_problems = set()
        self._built_target = 0
        self.build_time = None

        self.sols = [getattr(self, k) for k in dir(self) if k.startswith("sol")]

        try:
            p_spec = inspect.getfullargspec(self.sat)
        except OSError:
            utils.error("Cannot use inspect, happens in some interpreters... Try running in ipython.")
            raise

        self.arg_names = p_spec.args
        assert len(self.arg_names) > 0, f"{self.name}.problem() takes no arguments!"
        self.types = p_spec.annotations

        self._example = self.get_example()

        if self.sols:
            s_spec = inspect.getfullargspec(self.sols[0])
            assert self.arg_names[1:] == s_spec.args, \
                f"mismatched problem/solution arguments for {self.name}"
            self.types.update(s_spec.annotations)

        if self._example:
            assert set(self.arg_names[1:]) == set(self._example), f"Bad {self.name}._example"
            self.types.update({v: get_type(x) for v, x in self._example.items() if get_type(x, True)})

        for v in self.arg_names:
            assert v in self.types, f"Cannot determine type of `{v}` in {self.name} -- no annotation/_example"

        if self.__class__ not in [c for entry in problem_registry.values() for c in entry["classes"]]:
            utils.warn(f"Unregistered class `{self.name}`")

    def test(self, target_num_problems=1000):
        self.build(target_num_problems)
        solved = sum((1 if i.sol_srcs else 0) for i in self.instances)
        dur = self.build_time
        utils.info(f"Tested {solved:,}/{len(self.instances):,} instances of {self.name} in {dur:0.2f} seconds")

    def build(self, target_num_problems, max_random_attempts=100):
        if self._built_target == target_num_problems:
            return
        self._seen_problems = set()
        self._built_target = target_num_problems
        self.random.reseed()
        self.instances = []
        start_time = time.perf_counter()
        if self._example:
            self.add(self._example)
        if target_num_problems > 1:
            self.gen(target_num_problems - 1)
        while len(self.instances) < target_num_problems:
            n = len(self.instances)
            for _ in range(max_random_attempts):
                self.gen_random()
                if n != len(self.instances):  # added a problem
                    break
            if len(self.instances) == n:  # failed max_random_attempts, give up
                break

        if len(self.arg_names) == 1 and len(self.instances) == 0:  # no inputs, just add empty inputs
            self.add({})

        assert self.instances, f"{self.name} did not generate any problem instances"
        self.build_time = time.perf_counter() - start_time

    def gen(self, target_num_problems):
        pass

    def gen_random(self):
        pass

    def add(self, inp: dict, test=True):

        assert set(inp) == set(self.arg_names[1:]), \
            f"Instance #{len(self.instances)} keys mismatch in {self.name}"
        for v in inp:
            assert get_type(inp[v], ignore_errors=True) in (None, self.types[v]), \
                f"Instance #{len(self.instances)} variable `{v}` type mismatch in {self.name}"

        # p = v.inject(self.prob_src)
        # s = v.inject(self.sol_src) if self.sol_src else None
        # f = v.inject(self.fast_prob_src) if self.fast_prob_src else None
        if str(inp) in self._seen_problems:
            return  # don't add duplicate problems
        self._seen_problems.add(str(inp))

        var_name = self.arg_names[0]

        instance = Instance(
            get_src(self.sat, "sat", inp, self.types, add_type_assertion=True),
            f"{self.name}_{len(self.instances)}"
        )
        if test:
            for s in self.sols:
                instance.test(get_src(s, "sol", inp))

        self.instances.append(instance)


def get_src(f: Callable, new_function_name=None, defaults={}, types={}, add_type_assertion=False):
    try:
        src = inspect.getsource(f)
        spec = inspect.getfullargspec(f)
    except OSError:
        utils.error("Cannot use inspect, happens in some interpreters... Try running in ipython.")
        raise

    if spec.defaults:  # combine defaults, with defaults over-riding spec.defaults
        defaults = {**dict(zip(spec.args[-len(spec.defaults):], spec.defaults)), **defaults}
    assert all(var in spec.args for var in defaults), f"Defaults {defaults} not all in spec.args"

    for v, t in spec.annotations.items():
        assert v not in types or types[v] == t, f"Annotation mismatch in {src}"

    types = {**spec.annotations, **types}

    de_indent = min([len(line) - len(line.lstrip(" ")) for line in src.splitlines() if line.strip()])
    src = "\n".join([line[de_indent:] for line in src.splitlines()]).strip()

    if src.startswith("@staticmethod"):
        src = src[len("@staticmethod"):].strip()
    assert src.startswith("def ")
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
    top = """# The Python Reasoning Challenge dataset summary
This document summarizes the dataset stored in .json files.
Each .json file contains a number of related problems with one or more instances each.

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
            section += f"### {problem.name} ({link} {i + 1:,}/{n:,})\n\n"
            section += f"**Description:**\n{problem.desc}\n\n"
            section += f"**Problem:**\n\n```python\n{problem.instances[0].src}\n```\n"
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
