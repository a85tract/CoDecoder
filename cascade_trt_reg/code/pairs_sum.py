import json, os, sys
O = sys.argv[1]
for h in sys.argv[2:]:
    f = os.path.join(O, f"pairs_{h}.jsonl" if not h.startswith("alone_pipe") else f"{h}.jsonl")
    if not os.path.exists(f):
        continue
    for l in open(f):
        r = json.loads(l)
        if "ctrt" not in r:
            continue
        m = r.get("mode")
        v = r["latency_us"]["mean"] if m == "single" and isinstance(r.get("latency_us"), dict) else r.get("per_shot_us")
        p999 = r["latency_us"].get("p999") if m == "single" and isinstance(r.get("latency_us"), dict) else None
        ck = r.get("clock") or {}
        print(h, r.get("config"), r.get("p"), m, r["ctrt"]["engine"], "rep", r.get("rep"), "val %.3f" % v if v else v,
              "p999 %.1f" % p999 if p999 else "", "fracB", r.get("frac_B"), "fail", r.get("fail"),
              "clk", ck.get("clock_min_ratio") and round(ck.get("clock_min_ratio"), 3), r["date"][11:19])
