"""Aggregate several seed runs into mean / spread per condition."""
import glob
import json
import statistics as st
import sys

files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "data/results_s*.json"))
runs = {}
for f in files:
    r = json.load(open(f))
    for c in r["results"]:
        name = c["condition"].split("/")[-1]
        runs.setdefault(name, []).append((c["all"], c["direct"], c["derived"], c["general"]))

print("condition".ljust(12) + "n".rjust(3) + "  ".join(h.rjust(12) for h in ("all", "direct", "2-hop", "general")))
for cond, rows in runs.items():
    cells = []
    for i in range(4):
        vals = [r[i] for r in rows]
        m = st.mean(vals) * 100
        sd = st.stdev(vals) * 100 if len(vals) > 1 else 0.0
        cells.append("{:5.1f} +-{:4.1f}".format(m, sd))
    print(cond.ljust(12) + str(len(rows)).rjust(3) + "  ".join(cells))
