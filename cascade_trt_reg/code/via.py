import json, sys
for f in sys.argv[1:]:
    for l in open(f):
        r = json.loads(l)
        if r.get("mode") != "single" or "ctrt" not in r:
            continue
        s, b = r.get("latency_S_only_us") or {}, r.get("latency_via_B_us") or {}
        print(f.split("/")[-1], r.get("config"), r["p"], r["ctrt"]["engine"], r.get("rep"), "S_only %.1f" % s.get("mean", 0),
              "via_B %.1f (n %s)" % (b.get("mean", 0), b.get("n")), "B part %.1f" % (b.get("mean", 0) - s.get("mean", 0)))
