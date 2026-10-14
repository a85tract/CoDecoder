import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    t = r["trt"].get("b1") or {}
    print(r["net"], r["engine"], r["rep"], "rt_mean %.1f p999 %.1f gpu %.1f | h2h %.3f dev %.3f | clk %s %s | graph %s" % (
        r["host_rt_b1"]["mean"], r["host_rt_b1"]["p999"], sum(r["gpu_us_b1"]) / 5, r["amortized"]["h2h"],
        r["amortized"]["dev"], r["clock_b1"].get("clock_min_ratio"), round(r["clock_bB"].get("clock_min_ratio") or 0, 3),
        t.get("cuda_graph")))
