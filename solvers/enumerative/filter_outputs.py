import json
import sys
import os


inp_file = sys.argv[1]
if len(sys.argv) > 2:
    shift = int(sys.argv[2])
else:
    shift = 0
out_file = os.path.splitext(inp_file)[0]


with open(inp_file, 'r') as f:
    data = json.load(f)

thresholds = [100, 1000, 10000, 100000, 1000000]
for t in thresholds:
    t = t
    out = []
    suc = 0
    for p in data:
        #if not p["name"].startswith("Study"):
        #    continue
        if p["sols"][-1] != "" and p["sol_tries"][-1] + shift <= t:
            out.append(p)
            suc += 1
        else:
            out.append(dict(name=p["name"], sat=p["sat"], sols=[]))

    print(f"t={t}: solutions: {suc}/ {len(out)}")

    with open(out_file + f"_{t}.json", "w") as fw:
        json.dump(out, fw, indent=4)
