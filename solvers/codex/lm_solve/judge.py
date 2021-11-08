import hashlib
import os
import time
from utils import load_json
from pebble import ProcessPool
import multiprocessing as mp
from concurrent.futures import TimeoutError
from typing import List, Set, Tuple, Dict, Union, Any
from tqdm import tqdm
import functools

import utils
import ezlog
import signal
import sys

sys.setrecursionlimit(5000)


def _COPY(x):
    return x


_ENV = dict(
    List=List,
    Set=Set,
    Tuple=Tuple,
    Dict=Dict,
    COPY=_COPY,
    type_check=utils.type_check,
    test_puzzle=utils.test_puzzle,
    os=None,
    sys=None,
    input=None,
    open=None,
    print=None,
    compile=None,
    copyright=None,
)
_UNSAFE = ["import", "builtin", "__class", "open("]

MAX_WORKERS = mp.cpu_count() // 2

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "../.cache")  # ".cache"


class SetCache:
    """Simple cache that stores a set of keys. Cannot remove values. Haven't yet implemented iteration."""

    BYTE_LEN = 256 // 8  # for sha256

    def __init__(self, name, path=_CACHE_PATH):
        self._path = path
        self.name = name
        self._filename = os.path.join(path, f"{name}_set.cache")
        self._set = None  # the main set, loaded lazily

    def update(self, keys):
        self._load()
        hashes = [self._hash(k) for k in keys]
        additions = {h for h in hashes if h not in self._set}

        if additions:
            with open(self._filename, "ab") as f:  # append to binary file
                for h in additions:
                    f.write(h)
            self._set.update(additions)

    def add(self, key):
        self.update([key])

    def _hash(self, key):
        return hashlib.sha256(bytes(str(key), encoding="utf8")).digest()

    def __contains__(self, key: str):
        self._load()
        h = self._hash(key)
        return h in self._set

    def __delitem__(self, key):
        raise NotImplementedError

    def __len__(self):
        self._load()
        return len(self._set)

    def _load(self):
        if self._set is None:  # only load if not already loaded
            if not os.path.exists(self._path):
                ezlog.warn(f"Creating path for `{self.name}` cache")
                os.makedirs(self._path)

            time0 = time.perf_counter()
            if os.path.exists(self._filename):
                with open(self._filename, "rb") as f:  # read binary file
                    data = f.read()
                    self._set = {
                        data[j : j + self.BYTE_LEN] for j in range(0, len(data), self.BYTE_LEN)
                    }
            else:
                self._set = set()
            dur = time.perf_counter() - time0
            ezlog.info(f"Loaded `{self.name}` cache of {len(self):,} items in {dur:.1f}s")


# def judge_wrapped(code_env):
#     p = mp.Process(target=_judge)
#     p.start()
#     p.join(timeout)
#     if p.is_alive():
#         print("killing process")
#         p.kill()
#         p.join()


# def worker_init_fn():
#     sys.setrecursionlimit(
#         5000
#     )  # if this is too big then recursive programs can break the whole system
#     global print
#     print("New worker thread starting")

#     def handler(signum, frame):
#         print("Caught SIGALRM, raising exception")
#         raise TimeoutError("Timeout caught through SIGALRM")

#     old_print = print
#     print = lambda *args, **kwargs: old_print(f"[Process {os.getpid()}]", *args, **kwargs)

#     # Set the signal handler and a 5-second alarm
#     signal.signal(signal.SIGALRM, handler)


# def judge_wrapped(code_env, timeout):
#     code, env = code_env
#     print(code)
#     try:
#         signal.alarm(int(timeout))
#         res = _judge(code, env)
#         signal.alarm(0)  # disable the alarm
#         return res
#     except TimeoutError as e:
#         return False, e, code
#     finally:
#         print(f"done with {code}")


def _judge(code_env):
    code, env = code_env
    for u in _UNSAFE:
        if u in code:
            return False, Exception(f'unsafe: "{u}" not allowed in code'), code
    try:
        _env = env.copy()
        exec(code, _env)  # not sure if copy() is necessary
        return True, None, code
    except Exception as e:
        return False, e, code


# Cache judge results (which are nondeterministic due to timeout) for reproducibility
_judge_success = SetCache("judge_success")
_judged_batches = SetCache("judged_batches")


# def judge_parallel(src_codes, timeout: int, max_workers=MAX_WORKERS, env=_ENV, force_compute=False):
#     if timeout > 0:
#         assert timeout >= 1, "fractional timeout not supported, only ints (rounds down)"
#     global _judge_success, _judged_batches
#     # force_compute means no cache
#     if force_compute or src_codes not in _judged_batches:
#         new_codes = utils.dedup(code for code in src_codes if code not in _judge_success)
#         if new_codes:
#             ezlog.info(
#                 f"Judging {len(new_codes)}/{len(src_codes)} new codes (removing duplicates/things in cache)"
#             )
#             successes = []

#             print("writing to file for debugging before judging")
#             from train import save_json

#             save_json(new_codes, "results/tmp/new_codes.json")

#             with mp.Pool(processes=max_workers, initializer=worker_init_fn) as pool:
#                 with tqdm(total=len(new_codes), desc="Judging") as pbar:
#                     for success, exc, code in pool.imap_unordered(
#                         func=functools.partial(judge_wrapped, timeout=timeout),
#                         iterable=[(code, env) for code in new_codes],
#                     ):
#                         if success:
#                             successes.append(code)
#                         else:
#                             print(exc)
#                         pbar.update()
#                     _judge_success.update(successes)
#         _judged_batches.add(src_codes)
#     return [code in _judge_success for code in src_codes]


def judge_parallel(src_codes, timeout, max_workers=MAX_WORKERS, env=_ENV, force_compute=False):
    global _judge_success, _judged_batches
    # force_compute means no cache
    if force_compute or src_codes not in _judged_batches:
        new_codes = utils.dedup(code for code in src_codes if code not in _judge_success)
        if new_codes:
            ezlog.info(
                f"Judging {len(new_codes)}/{len(src_codes)} new codes (removing duplicates/things in cache)"
            )
            successes = []

            # print("writing to file for debugging before judging")
            # from train import save_json
            #
            # save_json(new_codes, "results/tmp/new_codes.json")

            with ProcessPool(max_workers=max_workers) as pool:
                future = pool.map(_judge, [(code, env) for code in new_codes], timeout=timeout)

                results = future.result()

                i = 0
                while True:
                    try:
                        success, exc, code = next(results)
                        if success:
                            successes.append(new_codes[i])
                    except StopIteration:
                        _judge_success.update(successes)
                        break
                    except (TimeoutError, Exception) as error:
                        pass
                    assert i < len(new_codes)
                    i += 1
                assert i == len(new_codes)
        _judged_batches.add(src_codes)

    return [code in _judge_success for code in src_codes]

    #                 itr = pool.imap_unordered(
    #                     func=_judge, iterable=[(code, env) for code in new_codes]
    #                 )
    #                 i = 0
    #                 while True:
    #                     try:
    #                         success, exc = itr.next(timeout)
    #                         if success:
    #                             successes.append(new_codes[i])
    #                         print("yay")
    #                     except StopIteration:
    #                         _judge_success.update(successes)
    #                         break
    #                     except (TimeoutError, Exception) as error:
    #                         print(error)
    #                         pass
    #                     assert i < len(new_codes)
    #                     i += 1
    #                     pbar.update()
    #                 assert i == len(new_codes)
    #     _judged_batches.add(src_codes)
    # return [code in _judge_success for code in src_codes]

    #         future = pool.map(_judge, , timeout=timeout)

    #         results = future.result()

    #         i = 0
    #         while True:
    #             try:
    #                 success, exc = next(results)
    #                 if success:
    #                     successes.append(new_codes[i])
    #             except StopIteration:
    #                 _judge_success.update(successes)
    #                 break
    #             except (TimeoutError, Exception) as error:
    #                 pass
    #             assert i < len(new_codes)
    #             i += 1
    #         assert i == len(new_codes)
    # _judged_batches.add(src_codes)


if __name__ == "__main__":
    import sys
    import pebble

    # worker_init_fn()
    # res = judge_wrapped(("while True: pass", _ENV), 1)
    res = judge_parallel(
        [
            "def sol(a: int=10000200001):\n  return (list(range(3 * a))[str(a)])\nx = sol()",
            "while True: pass",
            "def sol(): sol()\nsol()",
            "1",
        ],
        timeout=1,
    )
    print(res)

    sys.exit(0)

    # res = judge_parallel(["def foo():\n  foo()\nfoo()"], 1)

    # sys.exit(0)
    # for code in load_json("results/tmp/new_codes.json"):
    #     print(code)
    #     judge_wrapped((code, _ENV), 1)
    # sys.exit(0)

    judge_parallel(load_json("results/tmp/new_codes.json"), 1, max_workers=1)

    sys.exit(0)
    res = judge_parallel(
        [
            """1+1
        """,
            """assert False,'cats'""",
            """assert False""",
            """1[2]""",
            """1/0""",
            """while True:
        pass""",
            """for i in range(10**5):
        pass""",
        ]
        + [f"while True:\n {' '*n} pass" for n in range(60)],
        timeout=1,
        max_workers=4,
    )
    print(res)
