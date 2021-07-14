import hashlib
import os
import time
from pebble import ProcessPool
import multiprocessing
from concurrent.futures import TimeoutError
from typing import List, Set, Tuple, Dict, Union, Any

import utils
import ezlog

def _COPY(x):
    return x

_ENV = dict(List=List, Set=Set, Tuple=Tuple, Dict=Dict, COPY=_COPY,
            os=None, sys=None, input=None, open=None, print=None, compile=None, copyright=None)
_UNSAFE = ["import", "builtin", "__class"]

MAX_WORKERS = multiprocessing.cpu_count() // 2

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
        return hashlib.sha256(bytes(str(key), encoding='utf8')).digest()

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
                    self._set = {data[j:j + self.BYTE_LEN] for j in range(0, len(data), self.BYTE_LEN)}
            else:
                self._set = set()
            dur = time.perf_counter() - time0
            ezlog.info(f"Loaded `{self.name}` cache of {len(self):,} items in {dur:.1f}s")


def _judge(code_env):
    code, env = code_env
    for u in _UNSAFE:
        if u in code:
            return False

    try:
        exec(code, env.copy())  # not sure if copy() is necessary
        return True
    except Exception as e:
        return False


# Cache judge results (which are nondeterministic due to timeout) for reproducibility
_judge_success = SetCache('judge_success')
_judged_batches = SetCache('judged_batches')

def judge_parallel(src_codes, timeout, max_workers=MAX_WORKERS, env=_ENV, force_compute=False):
    global _judge_success, _judged_batches

    if force_compute or src_codes not in _judged_batches:
        new_codes = utils.dedup(code for code in src_codes if code not in _judge_success)
        if new_codes:
            ezlog.info(f"Judging {len(new_codes)}/{len(src_codes)} new codes (removing duplicates/things in cache)")
            successes = []


            with ProcessPool(max_workers=max_workers) as pool:
                future = pool.map(_judge, [(code, env) for code in new_codes], timeout=timeout)

                results = future.result()

                i = 0
                while True:
                    try:
                        if next(results):
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


if __name__ == "__main__":
    res = judge_parallel([
        """1+1
        """,
        """assert False,'cats'""",
        """assert False""",
        """1[2]""",
        """1/0""",
        """while True:
        pass""",
        """for i in range(10**5):
        pass"""
    ], timeout=1.0)
    print(res)
