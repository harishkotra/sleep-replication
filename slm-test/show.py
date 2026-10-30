import json
import sys

r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "data/results_7b-v2.json"))
rows = [("condition", "all", "direct", "derived", "general", "ctx")]
for c in r["results"]:
    rows.append((c["condition"], "{:.0%}".format(c["all"]), "{:.0%}".format(c["direct"]),
                 "{:.0%}".format(c["derived"]), "{:.0%}".format(c["general"]), str(c["ctx_words"])))
for row in rows:
    print("".join(str(x).ljust(11) for x in row))
for c in r["results"]:
    if c["condition"] in ("sft_raw", "sleep"):
        print("\n== " + c["condition"])
        for it in c["items"][5:11]:
            print("  Q: {}\n     got: {}  [{}]".format(it["q"][:60], it["got"][:60], it["ok"]))
