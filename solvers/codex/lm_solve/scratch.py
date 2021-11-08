from collections import Counter

def chars(filename, max_ord=300):
    with open(filename, "r", encoding="utf8") as f:
        counts = Counter(ord(c) for c in f.read())
    print(counts.most_common())
    print("max", max(counts))
    missing = [i for i in range(max_ord) if i not in counts]
    print("missing", missing)

chars(".cache/davinci-codex.cache")        

import time
time0 = time.perf_counter()
with open(".cache/davinci-codex.cache", "r", encoding="utf8") as f:
    lines = f.readlines()
time1 = time.perf_counter()
print("duration", time1 - time0)

time0 = time.perf_counter()
with open(".cache/davinci-codex.cache", "r", encoding="utf8") as f:
    lines = f.readlines()
time1 = time.perf_counter()
print("duration", time1 - time0)

import json
time0 = time.perf_counter()
elines = [json.loads(l) for l in lines]
time1 = time.perf_counter()
print("duration", time1 - time0)

len(lines), len(set(lines))
len(elines[0][0])
list(eval(elines[0][0]))
