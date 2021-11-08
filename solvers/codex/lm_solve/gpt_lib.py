import os
import json
import openai
import ezlog
import time
import datetime

assert 'OPENAI_API_KEY' in os.environ, "Need to set environment variable `OPENAI_API_KEY`"
openai.api_key = os.environ['OPENAI_API_KEY']
OPEN_AI_ENGINE_SUFFIX = os.environ.get('OPEN_AI_ENGINE_SUFFIX', '') # add extension such as -msft to engine names

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "../.cache")
_CACHE_ENCODING = "utf-8"


# the cache file is just a list of (query params dictionary encoded as a string but without n, result list)
# multiple queries with the same params (except for n) are merged into a single big list
class Cache:
    def __init__(self, filename):
        self.filename = filename
        self._cache = None

    def _load_cache(self):
        """for lazy loading"""
        assert self._cache is None, "gpt cache already loaded"

        if not os.path.exists(_CACHE_PATH):
            ezlog.warn("Creating cache path")
            os.makedirs(_CACHE_PATH)

        self._cache = {}

        if os.path.exists(self.filename):
            time0 = time.perf_counter()
            with open(self.filename, "r", encoding=_CACHE_ENCODING) as f:
                for k, v in [eval(line) for line in f.readlines()]:
                    if k not in self._cache:
                        self._cache[k] = v
                    else:
                        self._cache[k].extend(v)
            ezlog.info(f"Loaded cache `{self.filename}` in {time.perf_counter() - time0:.1f}s")
        else:
            ezlog.warn("No gpt cache yet")

    def defrag(self):
        if self._cache is None:
            self._load_cache()
        
        # def helper(k):  # remove max_batch
        #     k2 = eval(k)
        #     del k2["max_batch"]
        #     return str(k2)

        if self._cache:
            with open(self.filename, "w", encoding=_CACHE_ENCODING) as f:
                # f.write("\n".join([str((helper(k), v)) for k, v in self._cache.items()]+[""]))
                f.write("\n".join([str((k, v)) for k, v in self._cache.items()]+[""]))
            ezlog.info("Defragged cache")
        else:
            ezlog.warn("No cache to defrag")


    def get(self, item):
        if self._cache is None:
            self._load_cache()

        return self._cache.get(item, []).copy()  # no monkey business changing cache

    def extend(self, key, values):
        if self._cache is None:
            self._load_cache()

        v = self._cache.setdefault(key, [])
        v.extend(values)

        with open(self.filename, "a", encoding=_CACHE_ENCODING) as f:
            f.write(str((key, values)) + "\n")

        return v.copy()  # no monkey business changing cache


BATCH_SIZES = {
    "davinci": 32,
    "davinci-codex": 128, 
    "cushman-codex": 128
}

CACHES = {cache: Cache(os.path.join(_CACHE_PATH, cache + ".cache")) for cache in BATCH_SIZES}

def query(prompt, n=10, max_tokens=150, temp=1.0, stop=None, notes=None, cache_only=False, verbose=True,
          max_retries=10, engine="cushman-codex"):
    """Query gpt

    :param prompt: Up to 2048 tokens (about 3-4k chars)
    :param n: number of answers, None returns all cached answers
    :param max_tokens:
    :param temp: 0.9 seems to work well
    :param stop: string to stop at or '' if not to stop
    :param notes: notes you want to save or change in case you want to run the same query more than once!
    :return: list of answers and then the response items
    """
    global BATCH_SIZES
    global CACHES
    cur_cache = CACHES[engine]
    max_batch = BATCH_SIZES[engine]
    engine += OPEN_AI_ENGINE_SUFFIX # add tail to engine name

    if temp == 0 and n > 1:
        ezlog.debug("Temp 0: no point in running more than one query")
        n = 1

    key = str(dict(prompt=prompt, max_tokens=max_tokens, temp=temp, stop=stop, rep=notes))

    cached = cur_cache.get(key)

    if n is None:
        return cached

    if len(cached) >= n:
        return cached[:n]

    assert not cache_only, f'Entry not found in cache with prompt "{json.dumps(prompt)}"'
    if verbose:
        print("/" * 100)
        print(f"Querying GPT {engine} with prompt:")
        print(prompt)
        s = stop and stop.replace('\n', '\\n')
        print(f"/// n={n} ({n - len(cached)} new) max_tokens={max_tokens} temp={temp} max_batch={max_batch} stop={s}")
        print("/" * 100)

    time0 = time.perf_counter()

    new = []
    n -= len(cached)

    while n > 0:
        m = min(n, max_batch)

        try_number = 0
        while True:
            try:
                res = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temp,
                    n=m,
                    stop=stop or None
                )
                break
            except (openai.error.RateLimitError, openai.error.APIError):
                if try_number == max_retries:
                    print("Rate limit error: Giving up!")
                    raise
                sleep_secs = 10 * (2 ** try_number)
                try_number += 1
                print(f"Rate limit error #{try_number}: Sleeping for {sleep_secs} seconds...")
                time.sleep(sleep_secs)

        new += [c["text"] for c in res["choices"]]
        n -= m

    return cur_cache.extend(key, new)
