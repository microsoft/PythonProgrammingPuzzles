from utils import load_json
from pebble import ProcessPool
import multiprocessing as mp
from concurrent.futures import TimeoutError
from typing import List, Set, Tuple, Dict

import utils
import sys
import re
from copy import deepcopy

sys.setrecursionlimit(5000)


def no_print(*_args, **_kwargs):
    pass


def run_judge(judge, f, tests):
    answer_type = list(judge.__annotations__.values())[0]
    for x in tests:
        y = f(**deepcopy(x))  # so f cannot cheat and change the input x
        if not utils.type_check(y, answer_type):
            raise TypeError
        assert judge(y, **x) is True, f"{f} failed on test {x}"


_ENV = dict(
    List=List,
    Set=Set,
    Tuple=Tuple,
    Dict=Dict,
    type_check=utils.type_check,
    run_judge=run_judge,
    test_puzzle=utils.test_puzzle,
    os=None,
    sys=None,
    input=None,
    open=None,
    print=no_print,
    compile=None,
    copyright=None,
)

_UNSAFE = ["builtin", "__class", "open("]
_SAFE_IMPORTS = {"collections", "copy", "hashlib", "math", "random", "re", "string", "typing"}

MAX_WORKERS = mp.cpu_count() // 2



def unsafe_imports(code):
    """Check if code imports any unsafe modules.

    Args:
        code (str): The code to check.

    Returns:
        bool: True if code imports unsafe modules.
    """
    if "import" not in code:
        return False
    for line in code.split("\n"):
        if "import" in line:
            match = re.search(r"^\s*from\s+([\w\.]+)\s+import\s", line)
            if match:
                modules = [match.group(1)]
            else:
                match = re.search(r"^\s*import\s+(.+)", line)
                if match:
                    modules = match.group(1).split(",")
                else:
                    return True
            if any(m.strip() not in _SAFE_IMPORTS for m in modules):
                return True
    return False


def _judge(code_env):
    code, env = code_env
    if unsafe_imports(code) or any(u in code for u in _UNSAFE):
        return False, Exception(f"unsafe code"), code
    try:
        exec(code, env.copy()) 
        return True, None, code
    except Exception as e:
        return False, e, code


def judge_parallel(src_codes, timeout, max_workers=MAX_WORKERS, env=_ENV):
    codes = utils.dedup(src_codes)
    utils.info(
        f"Judging {len(src_codes):,} codes ({len(src_codes)-len(codes):,} duplicates) with {max_workers} workers"
    )
    successes = set()

    # print("writing to file for debugging before judging")
    # from train import save_json
    #
    # save_json(new_codes, "results/tmp/new_codes.json")
    utils.silence_std_err(True)
    with ProcessPool(max_workers=max_workers) as pool:
        future = pool.map(_judge, [(code, env) for code in codes], timeout=timeout)

        results = future.result()
        i = 0
        while True:
            try:
                success, exc, code = next(results)
                if success:
                    successes.add(codes[i])
            except StopIteration:
                break
            except (TimeoutError, Exception) as error:
                pass
            assert i < len(codes)
            i += 1
        assert i == len(codes)
    utils.silence_std_err(False)
    return [code in successes for code in src_codes]




def test():
    import time

    tests = [
        ("def sol(a: int=10000200001):\n  return (list(range(3 * a))[str(a)])\nx = sol()", False),
        ("print(12)", True),
        ("while True: pass", False),
        ("def sol(): sol()\nsol()", False),
        ("2+2", True),
        ("""1+1""", True),
        ("""assert False,'cats'""", False),
        ("""assert False""", False),
        ("""1[2]""", False),
        ("""1/0""", False),
        (
            """while True:
        pass""",
            False,
        ),
        (
            """for i in range(10**4):
        pass""",
            True,
        ),
        ("print('hello')", True),
    ]

    scores = {}
    tests2 = tests
    pad = " "
    for _ in range(6):
        print(f"n={len(tests2)} timing test" + "*" * 20)
        times = []
        for max_workers in [4, 16, 32, 64, 128]:
            time0 = time.perf_counter()
            res = judge_parallel([test for test, r in tests2], timeout=1, max_workers=max_workers)
            for (test, expected), r in zip(tests2, res):
                assert expected == r, f"Failed expected {expected}, got {r} for {test}"

            times.append((max_workers, time.perf_counter() - time0))

        scores[len(tests2)] = times
        tests2 = tests2 + [(t + pad, r) for (t, r) in tests2]
        pad = pad * 2
    print("mp.cpu_count() =", mp.cpu_count())

    for n, times in scores.items():
        print(n, "tests, [(max_workers, time)] =", times)


if __name__ == "__main__":
    test()
