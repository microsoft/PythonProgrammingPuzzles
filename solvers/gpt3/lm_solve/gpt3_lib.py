import os
import json
import openai
import ezlog
import time
import datetime

assert 'OPENAI_API_KEY' in os.environ, "Need to set environment variable `OPENAI_API_KEY`"
openai.api_key = os.environ['OPENAI_API_KEY']


_CACHE_PATH = os.path.join(os.path.dirname(__file__), "../.cache")
_CACHE_FILENAME = os.path.join(_CACHE_PATH, "gpt3.cache")
_ENCODING = "utf-8"

_cache = None


# the cache file is just a list of (query params dictionary encoded as a string but without n, result list)
# multiple queries with the same params (except for n) are merged into a single big list
def _save_line(item, comment=None):
    global _cache
    assert _cache is not None
    with open(_CACHE_FILENAME, "a", encoding=_ENCODING) as f:
        f.write(str(item)+ ((" # " + comment + "\n") if comment else "\n"))

def _load_cache():
    global _cache

    assert _cache is None, "gpt3 cache already loaded"

    if not os.path.exists(_CACHE_PATH):
        ezlog.warn("Creating cache path")
        os.makedirs(_CACHE_PATH)

    _cache = {}

    if os.path.exists(_CACHE_FILENAME):
        time0 = time.perf_counter()
        with open(_CACHE_FILENAME, "r", encoding=_ENCODING) as f:
            for k, v in [eval(line) for line in f.readlines()]:
                if k not in _cache:
                    _cache[k] = v
                else:
                    _cache[k].extend(v)
        ezlog.info(f"Loaded gpt3 cache in {time.perf_counter()-time0:.1f}s")
    else:
        ezlog.warn("No gpt3 cache yet")



def query(prompt, n=10, max_tokens=150, temp=1.0, max_batch=32, stop=None, notes=None, cache_only=False, verbose=True):
    """Query gpt3

    :param prompt: Up to 2048 tokens (about 3-4k chars)
    :param n: number of answers, None returns all cached answers
    :param max_tokens:
    :param temp: 0.9 seems to work well
    :param max_batch: max to query at once
    :param stop: string to stop at or '' if not to stop
    :param notes: notes you want to save or change in case you want to run the same query more than once!
    :return: list of answers and then the response items
    """
    global _cache
    if _cache is None:
        _load_cache()

    if temp == 0 and n > 1:
        ezlog.debug("Temp 0: no point in running more than one query")
        n = 1

    key = str(dict(prompt=prompt, max_tokens=max_tokens, temp=temp, max_batch=max_batch, stop=stop, rep=notes))
    cached = _cache.get(key, [])
    if n is None:
        return cached[:]

    if len(cached) >= n:
        return cached[:n]

    if cache_only:
        pass
        1/0
    assert not cache_only, "Entry not found in cache"
    if verbose:
        print("/"*100)
        print("Querying GPT3 with prompt:")
        print(prompt)
        s = stop and stop.replace('\n', '\\n')
        print(f"/// n={n} ({n-len(cached)} new) max_tokens={max_tokens} temp={temp} max_batch={max_batch} stop={s}")
        print("/"*100)

    time0 = time.perf_counter()

    new = []
    n -= len(cached)

    while n > 0:
        m = min(n, max_batch)

        res = openai.Completion.create(
            engine="davinci-msft",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temp,
            n=m,
            stop=stop or None
        )

        new += [c["text"] for c in res["choices"]]
        n -= m

    _save_line((key, new), f"{time.perf_counter() - time0:.1f}s {datetime.datetime.now()}")
    ans = _cache[key] = cached + new
    return ans[:]

# old code
# # to persist calls to the API...
# _disk_cache = joblib.Memory(os.path.join(os.path.dirname(__file__), ".cache"), verbose=1).cache
#
#
# @_disk_cache
# def query(prompt, n=10, max_tokens=150, temperature=1.0, max_batch=32):
#     """Query gpt3
#
#     :param prompt: Up to 2048 tokens (about 3-4k chars)
#     :param n: number of answers
#     :param max_tokens:
#     :param temperature:
#     :param max_batch: max to query at once
#     :return: list of answers and then the response items
#     """
#     if temperature == 0 and n > 1:
#         ezlog.debug("Temp 0: no point in running more than one query")
#         n = 1
#
#     responses = []
#     while n > 0:
#         m = min(n, max_batch)
#         prompt_summary = prompt if len(prompt) < 80 else f"{prompt[:40]}...{prompt[-40:]}"
#         ezlog.warn(f"**** Running GPT3 query: temp {temperature}, n={m}, prompt={prompt_summary}")
#         time0 = time.perf_counter()
#         responses.append(openai.Completion.create(
#             engine="davinci-msft",
#             prompt=prompt,
#             max_tokens=max_tokens,
#             temperature=temperature,
#             n=m
#         ))
#         ezlog.info(f"**** Got response in {time.perf_counter()-time0}s...")
#         n -= m
#
#     return [c["text"] for r in responses for c in r["choices"]], responses


