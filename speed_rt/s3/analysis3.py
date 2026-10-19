"""SPEED-RT stage 3's frozen analysis (speed_rt/STAGE3_batched.md sections 6 and 7). Standard library only.

    python -m speed_rt.s3.analysis3 --records speed_rt/s3/records --out OUT

Reads: records/manifests (the draw's manifests), records/gpu/<card>/{decide,time}/ (gpu3.py's records), records/cpu/
(win3.py's run3.json, harness.py's window.json and every process's cpu3.json, the session reference), the beacon record
speed_rt/s3/beacon.json and stage 2's Grace fit. Checks every record's shot files against the manifests and the
manifests' seeds against the beacon, takes only accepted processes (GPU: the clock window accepted and outputs equal to
the decision pass; CPU: a passed window, exit 0, a record, one common start per group) and withholds any value whose
slot is not established. Writes OUT/numbers.json (every value and repetition), OUT/table.md and OUT/claims.json."""
import argparse
import glob
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REL)
from speed_rt import sets  # noqa: E402

FIT = os.path.join(REL, "speed_rt/s2/surface/results/fit_s2.json")
GRACE = 1.5097605583565314
ROUNDS = {"surf-d7": 7, "surf-d9": 9, "surf-d11": 11, "surf-d13": 13, "surf-d15": 15, "surf-d17": 17,
          "bb72-p2": 6, "bb72-p3": 6, "gross-p4": 12}
N_GPU = {"gross-p4": 3}            # fresh GPU processes per (point, mode); 5 elsewhere
N_RUNS = 3
SETS = {"surf-d7": "s3-surf-d7", "surf-d9": "s3-surf-d9", "surf-d11": "s3-surf-d11", "surf-d13": "s3-surf-d13",
        "surf-d15": "s3-surf-d15", "surf-d17": "s3-surf-d17", "bb72-p2": "s3-bb72-p2", "bb72-p3": "s3-bb72-p3",
        "gross-p4": "s3-gross-p4"}
# compositions: (point, tau name, our CPU item or None when nothing can be routed, fallback baseline, other baselines)
COMPS = {}
for _d in (7, 9, 11, 13, 15, 17):
    COMPS[f"surf-d{_d}"] = (f"surf-d{_d}", "q", f"surf-d{_d}.ours", f"surf-d{_d}.cpm", [f"surf-d{_d}.pm"])
for _p in ("bb72-p2", "bb72-p3"):
    COMPS[f"{_p}.balanced"] = (_p, "q", f"{_p}.bal", f"{_p}.sector", [f"{_p}.full"])
    COMPS[f"{_p}.accurate"] = (_p, "q", f"{_p}.acc", f"{_p}.full", [f"{_p}.sector"])
COMPS["gross-p4.balanced"] = ("gross-p4", "bal", "gross-p4.bal", "gross-p4.sector", ["gross-p4.full"])
COMPS["gross-p4.accurate"] = ("gross-p4", "acc", "gross-p4.acc", "gross-p4.full", ["gross-p4.sector"])
BASELINES = sorted({b for c in COMPS.values() for b in [c[3]] + c[4]})
NS = (1, 8)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def up(x, sig=4):
    """x rounded up to sig significant digits (our times)."""
    if x is None or x == 0:
        return x
    e = math.floor(math.log10(abs(x))) - sig + 1
    return round(math.ceil(x / 10 ** e - 1e-9) * 10 ** e, max(0, -e))


def down(x, sig=3):
    """x rounded down to sig significant digits (speed-ups)."""
    if x is None or x == 0:
        return x
    e = math.floor(math.log10(abs(x))) - sig + 1
    return round(math.floor(x / 10 ** e + 1e-9) * 10 ** e, max(0, -e))


def manifests(rec_dir, beacon_path):
    """{file name: (set, chunk, seed, sha256)} of the draw's manifests, every seed re-derived from the beacon."""
    b = json.load(open(beacon_path))
    sf = os.path.join(REL, "speed_rt/s3/sets_s3.json")
    seeds = sets.all_seeds(b["randomness"], sf)
    out = {}
    m = json.load(open(os.path.join(rec_dir, "manifests", "srt-s3-surf.json")))
    for f, e in m["files"].items():
        assert seeds[e["set"]][e["j"]] == e["seed"], f"{f}: seed"
        out[f] = (e["set"], e["j"], e["seed"], e["sha256"])
    import csv
    for path in sorted(glob.glob(os.path.join(rec_dir, "manifests", "manifest_s3_*.csv"))):
        for r in csv.DictReader(open(path)):
            name = {("bb72", "0.002"): "s3-bb72-p2", ("bb72", "0.003"): "s3-bb72-p3",
                    ("bb144", "0.004"): "s3-gross-p4"}[(r["code"], r["p"])]
            assert seeds[name][int(r["chunk"])] == int(r["seed"]), f"{r['file']}: seed"
            out[r["file"]] = (name, int(r["chunk"]), int(r["seed"]), r["sha256"])
    return out, b


def files_ok(files, man, point):
    want = sorted((f, e[3]) for f, e in man.items() if e[0] == SETS[point])
    return sorted(tuple(x) for x in files) == want


def exits(path):
    """{(point, slot, rep, attempt): exit code} from a pod's progress.txt (run3.sh)."""
    out = {}
    if os.path.exists(path):
        for ln in open(path):
            f = ln.split()
            if len(f) == 10 and f[1] == "time" and f[4] == "rep" and f[6] == "attempt" and f[8] == "exit":
                out[(f[2], f[3], int(f[5]), int(f[7]))] = int(f[9])
    return out


def gpu(rec_dir, man):
    """{card: {point: {"decide": rec, "<slot>:all": [attempts], "<slot>": slowest of the established repetitions}}};
    slot "route:<tau>" or "alone"; per repetition the first accepted attempt (window accepted, outputs equal to the
    decision pass, exit 0 in the pod's progress log); a repetition without one is not established and withholds G."""
    out = {}
    for cd in sorted(glob.glob(os.path.join(rec_dir, "gpu", "*"))):
        card = os.path.basename(cd)
        res = out[card] = {}
        rc = exits(os.path.join(cd, "progress.txt"))
        for f in sorted(glob.glob(os.path.join(cd, "decide", "decide_*.json"))):
            d = json.load(open(f))
            assert files_ok(d["files"], man, d["point"]), f"{f}: shot files differ from the manifest"
            res.setdefault(d["point"], {})["decide"] = d
        for f in sorted(glob.glob(os.path.join(cd, "time", "time_*.json"))):
            t = json.load(open(f))
            pt = t["point"]
            key = "alone" if t["mode"] == "alone" else f"route:{t['tau_name']}"
            att = t.get("attempt", 1)
            code = rc.get((pt, key, t["rep"], att))
            ok = bool(t["outputs_equal_decision_pass"] and t["window"].get("accepted") is True and code == 0
                      and files_ok(t["files"], man, pt))
            res.setdefault(pt, {}).setdefault(key + ":all", []).append(
                dict(rep=t["rep"], attempt=att, us=t["per_shot_us"], ok=ok, exit=code, file=os.path.relpath(f, rec_dir)))
        for pt, r in res.items():
            for k in [k for k in r if k.endswith(":all")]:
                need = N_GPU.get(pt, 5)
                vals = []
                for rep in range(1, need + 1):
                    acc = sorted((x for x in r[k] if x["rep"] == rep and x["ok"]), key=lambda x: x["attempt"])
                    if acc:
                        vals.append(acc[0]["us"])
                r[k[:-4]] = max(vals) if len(vals) == need else None        # our side: the slowest repetition
    return out


def flags_differ(rec_dir, card, ref_card, pt, taus):
    """The number of timing shots whose routing flag differs between two cards' decision passes (numpy for the npz)."""
    import numpy as np
    za = np.load(os.path.join(rec_dir, "gpu", card, "decide", f"decide_{pt}.npz"))
    zb = np.load(os.path.join(rec_dir, "gpu", ref_card, "decide", f"decide_{pt}.npz"))
    return int(sum((za[f"routed_{t}"] != zb[f"routed_{t}"]).sum() for t in taus))


def cpu(rec_dir, man):
    """{(item, N): {"runs": [{rep, us, ...}], ...}} from accepted windows."""
    win_dir = os.path.join(rec_dir, "cpu", "win")
    runs = {}
    for wf in sorted(glob.glob(os.path.join(win_dir, "*", "window.json"))):
        w = json.load(open(wf))
        if not w["passed"]:
            continue
        groups = {}
        for j in w["jobs"]:
            assert j["rc"] == 0 and j["record"], wf
            jd = os.path.join(os.path.dirname(wf), os.path.basename(j["out"]))
            for cf in sorted(glob.glob(os.path.join(jd, "*", "cpu3.json"))):
                c = json.load(open(cf))
                assert files_ok(c["files"], man, c["point"]), f"{cf}: shot files differ from the manifest"
                groups.setdefault(c["gid"], []).append(c)
        for gid, cs in groups.items():
            n = cs[0]["n"]
            assert len(cs) == n and len({c["T_ns"] for c in cs}) == 1 and sorted(c["k"] for c in cs) == list(range(n)), gid
            assert sum(c["rows_mine"] for c in cs) == cs[0]["rows_total"], gid
            wall = (max(c["t1_ns"] for c in cs) - cs[0]["T_ns"]) / 1e9
            rep = cs[0]["rep"]
            key = (cs[0]["item"], n)
            prev = [x for x in runs.get(key, []) if x["rep"] == rep]
            assert not prev, f"{key} rep {rep} twice"
            runs.setdefault(key, []).append(dict(rep=rep, wall_s=wall, shots=cs[0]["timing_shots"], rows=cs[0]["rows_total"],
                                                 us=1e6 * wall / cs[0]["timing_shots"], window=w["window"],
                                                 late_ms=max(c["late_ns"] for c in cs) / 1e6,
                                                 failures=sum(c.get("failures", 0) for c in cs)))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beacon", default=os.path.join(REL, "speed_rt/s3/beacon.json"))
    a = ap.parse_args()
    fit = json.load(open(FIT))
    assert fit["complete"] and fit["fit"]["common"] and set(fit["fit"]["applied"].values()) == {GRACE}, "stage 2's fit"
    man, beacon = manifests(a.records, a.beacon)
    G = gpu(a.records, man)
    C = cpu(a.records, man)
    run3 = json.load(open(os.path.join(a.records, "cpu", "win", "run3.json")))
    sref_ok = bool(run3["sessionref_within_1p10_of_stage2"])
    plan = json.load(open(os.path.join(a.records, "cpu", "plan.json")))
    planned = {(it, j["N"]) for b in plan["blocks"] for w in b["windows"] for j in w["jobs"] for it in j["items"]}
    mask_card = "h100" if "h100" in G else (sorted(G)[0] if G else None)       # the CPU terms' routing flags
    decided = {pt: r["decide"] for pt, r in G.get(mask_card, {}).items() if "decide" in r}
    differ = {}                                   # another card's composites need its flags equal to the mask card's
    for card in G:
        for pt, r in G[card].items():
            if card != mask_card and "decide" in r and pt in decided:
                differ[(card, pt)] = flags_differ(a.records, card, mask_card, pt, sorted(r["decide"]["taus"]))

    def cpu_stat(item, n, ours):
        """(raw us per timing shot or None, the runs); ours: slowest of 3, baseline: fastest of 3; 0 when no row."""
        if item is None:
            return 0.0, []
        pt = item.split(".")[0]
        if item.endswith((".ours", ".bal", ".acc")) and pt in decided:
            tn = {"ours": "q", "bal": "bal" if pt == "gross-p4" else "q", "acc": "acc" if pt == "gross-p4" else "q"}[item.split(".")[1]]
            if decided[pt]["taus"][tn]["routed"] == 0:
                return 0.0, []
        rs = C.get((item, n), [])
        if len({r["rep"] for r in rs}) < N_RUNS:
            return None, rs
        v = [r["us"] for r in rs]
        return (max(v) if ours else min(v)), rs

    numbers = dict(what="SPEED-RT stage 3 numbers (speed_rt/s3/analysis3.py)", grace_factor=GRACE,
                   sessionref_within_1p10_of_stage2=sref_ok, beacon_round=beacon["round"], cards=sorted(G), gpu={},
                   cpu={}, compositions={}, alone={}, baselines={}, not_timed=run3.get("not_timed", []),
                   mask_card=mask_card, flags_differ={f"{c}.{p}": v for (c, p), v in sorted(differ.items())})
    for card in G:
        numbers["gpu"][card] = {pt: {k: v for k, v in r.items() if k != "decide"} for pt, r in G[card].items()}
    for (item, n), rs in sorted(C.items()):
        numbers["cpu"][f"{item}.N{n}"] = rs
    for b in BASELINES:
        pt = b.split(".")[0]
        for n in NS:
            raw, _ = cpu_stat(b, n, ours=False)
            numbers["baselines"][f"{b}.N{n}"] = dict(
                raw_us_shot=raw, grace_us_shot=None if raw is None or not sref_ok else raw * GRACE,
                raw_us_round=None if raw is None else raw / ROUNDS[pt],
                grace_us_round=None if raw is None or not sref_ok else raw * GRACE / ROUNDS[pt],
                planned=(b, n) in planned)
    claims = []
    for card in G:
        for pt in ROUNDS:
            g = G[card].get(pt, {}).get("alone")
            numbers["alone"][f"{card}.{pt}"] = dict(G_us_shot=g, us_round=None if g is None else g / ROUNDS[pt],
                                                    below_1us_round=None if g is None else g / ROUNDS[pt] < 1.0)
        for cname, (pt, tau, item, fb, others) in COMPS.items():
            g = G[card].get(pt, {}).get(f"route:{tau}")
            if card != mask_card and differ.get((card, pt), 1) != 0:
                g = None                                  # withheld: its routing differs from the CPU terms' flags
            for n in NS:
                c_raw, _ = cpu_stat(item, n, ours=True)
                row = dict(card=card, composition=cname, N=n, G_us_shot=g, C_raw_us_shot=c_raw)
                for basis, f in (("grace", GRACE), ("raw", 1.0)):
                    ok = g is not None and c_raw is not None and (basis == "raw" or sref_ok)
                    t = g + c_raw * f if ok else None
                    pr = None if t is None else t / ROUNDS[pt]
                    row[basis] = r = dict(T_us_shot=t, us_round=pr, us_round_up=up(pr),
                                          below_1us_round=None if pr is None else pr < 1.0,
                                          pipelined_us_round=None if not ok else max(g, c_raw * f) / ROUNDS[pt],
                                          speedups={})
                    for bl in [fb] + others:
                        b_raw, _ = cpu_stat(bl, n, ours=False)
                        s = None if t is None or b_raw is None else b_raw * f / t
                        r["speedups"][bl] = dict(fallback=bl == fb, value=s, value_down=down(s), holds=None if s is None else s > 1.0)
                numbers["compositions"][f"{card}.{cname}.N{n}"] = row
                claims.append(dict(card=card, composition=cname, N=n,
                                   grace_us_round=row["grace"]["us_round_up"], raw_us_round=row["raw"]["us_round_up"],
                                   below_1us_grace=row["grace"]["below_1us_round"], below_1us_raw=row["raw"]["below_1us_round"],
                                   speedups_grace={k: v["value_down"] for k, v in row["grace"]["speedups"].items()},
                                   speedups_raw={k: v["value_down"] for k, v in row["raw"]["speedups"].items()}))
    # failure counts on the timing shots, descriptive only: the network's on its unrouted shots (the H100's decision
    # pass) plus the fallback's on the routed shots (the N = 1 run of repetition 1); a baseline's on the shots it decoded
    fails = {}
    for cname, (pt, tau, item, fb, others) in COMPS.items():
        d = decided.get(pt)
        if d is None:
            continue
        net = d["taus"][tau]["network_failures_unrouted"]
        fr = [r for r in C.get((item, 1), []) if r["rep"] == 1]
        routed = d["taus"][tau]["routed"]
        fb_f = 0 if routed == 0 else (fr[0]["failures"] if fr else None)
        fails[cname] = dict(shots=d["shots"], routed=routed, network_unrouted=net, fallback_routed=fb_f,
                            total=None if fb_f is None else net + fb_f,
                            codecoder_alone=d["taus"][tau]["network_failures_all"])
    for b in BASELINES:
        fr = [r for r in C.get((b, 1), []) if r["rep"] == 1]
        if fr:
            fails[b] = dict(shots=fr[0]["rows"], total=fr[0]["failures"])
    numbers["failures_descriptive"] = fails
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "numbers.json"), "w") as fh:
        json.dump(numbers, fh, indent=1, sort_keys=True)
    with open(os.path.join(a.out, "claims.json"), "w") as fh:
        json.dump(claims, fh, indent=1, sort_keys=True)
    lines = ["# SPEED-RT stage 3: throughput mode (batch 4,096), us per round", "",
             f"Grace factor {GRACE} (stage 2's); session reference within 1.10 of stage 2's: {sref_ok}.", ""]
    for card in G:
        lines += [f"## {card}", "", "| composition | N | G us/shot | C raw us/shot | us/round Grace | us/round raw | < 1 us (Grace) | speed-ups Grace | speed-ups raw |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for cname in COMPS:
            for n in NS:
                r = numbers["compositions"][f"{card}.{cname}.N{n}"]
                sg = ", ".join(f"{k}: {v['value_down']}" for k, v in r["grace"]["speedups"].items())
                sr = ", ".join(f"{k}: {v['value_down']}" for k, v in r["raw"]["speedups"].items())
                lines.append(f"| {cname} | {n} | {r['G_us_shot']} | {r['C_raw_us_shot']} | {r['grace']['us_round_up']} | "
                             f"{r['raw']['us_round_up']} | {r['grace']['below_1us_round']} | {sg} | {sr} |")
        lines += ["", "| CoDecoder alone | G us/shot | us/round | < 1 us |", "|---|---|---|---|"]
        for pt in ROUNDS:
            r = numbers["alone"][f"{card}.{pt}"]
            lines.append(f"| {pt} | {r['G_us_shot']} | {up(r['us_round'])} | {r['below_1us_round']} |")
        lines.append("")
    lines += ["## Baselines (workstation)", "", "| baseline | N | raw us/shot | Grace us/round | raw us/round |", "|---|---|---|---|---|"]
    for k, r in numbers["baselines"].items():
        lines.append(f"| {k.rsplit('.N', 1)[0]} | {k.rsplit('.N', 1)[1]} | {r['raw_us_shot']} | {r['grace_us_round']} | {r['raw_us_round']} |")
    lines += ["", f"Not timed (budget): {[x['name'] for x in numbers['not_timed']]}", ""]
    with open(os.path.join(a.out, "table.md"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"{len(claims)} claims; cards {sorted(G)}; not timed {[x['block'] for x in numbers['not_timed']]}")


if __name__ == "__main__":
    main()
