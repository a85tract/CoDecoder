"""A1's checks of the matched conditions (revision/REVISION_PLAN.md 2.1), run by the dry jobs and again before the
main runs.
  matched  CPU: each arm built as its trainer builds it (torch seed s, head seed s + 1,000): the trunk equals the seed's
           trunk file and differs between seeds; arms that share a head module start it identically (S-VA / S-V vote
           head; S-P / S-PA pooled head; S-PA's training-only line read-out = S-VA's head; B-JT / B-J; B-P / B-PA;
           B-PA's training-only translate read-out = B-JT's head); writes the size table: parameters, read-out
           multiply-adds and their ratio to the deployed read-out (the pooled heads must be within 10 %)
  traces   the per-step traces (arch.train --trace) of the arms of one seed: the same batches (x and y hashes) and the
           same learning rate at every step, and other batches for the other seed
  bbchunks BB runs (no per-step trace): the eval rows' chunk index and chunk digest equal across the arms of a seed
  checks   the runs' not-trained checks (train_check.json, bb log.json) -> one JSON list
  graph    each arm's decision file at batch one: a CUDA graph of the decision path (the A3 implementation) against
           eager fp32 on the dry test shots (equal decisions, largest confidence difference) and its replay time (a
           shared node: a smoke test and the dry time axis, never A3's numbers)
    python -m a1.check matched --code surface --trunks DIR --out FILE
    python -m a1.check traces --files T1.csv T2.csv ... --seeds 1 1 ... --out FILE
    python -m a1.check checks --code surface --runs RUN ... --out FILE
    python -m a1.check graph --code surface --reads DIR --arms ... --seeds ... --kind dry --chunks 4-9 --out FILE"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from a1 import seeds as a1seeds
from a1 import trunk as a1trunk

SURFACE = {"S-VA": ("vote", False), "S-V": ("vote", False), "S-L": ("line", False), "S-P": ("pooled", False),
           "S-PA": ("pooled", True)}
BB = {"B-JT": ("joint", False), "B-J": ("joint", False), "B-P": ("pooled", False), "B-PA": ("pooled", True)}


def matched(a):
    arms = SURFACE if a.code == "surface" else BB
    dep = a1seeds.DEPLOYED[a.code]
    out = {"code": a.code, "seeds": {}, "sizes": []}
    trunks = {}
    for s in a1seeds.SEEDS:
        ck = torch.load(os.path.join(a.trunks, f"trunk_{a.code}_s{s}.pt"), map_location="cpu", weights_only=False)
        trunks[s] = ck["state"]
        models = {arm: a1trunk.build(a.code, s, h, a.design, aux)[0] for arm, (h, aux) in arms.items()}
        sd = {arm: m.state_dict() for arm, m in models.items()}
        res = {}
        for arm, st in sd.items():
            res[f"{arm} trunk = trunk file"] = all(torch.equal(st[k], v) for k, v in ck["state"].items())
        pairs = [("S-VA", "S-V", "head."), ("S-P", "S-PA", "head."), ("S-VA", "S-PA", ("head.", "aux."))] \
            if a.code == "surface" else [("B-JT", "B-J", "head."), ("B-P", "B-PA", "head."), ("B-JT", "B-PA", ("head.", "aux."))]
        for x, y, pre in pairs:
            px, py_ = (pre, pre) if isinstance(pre, str) else pre
            kx = sorted(k for k in sd[x] if k.startswith(px))
            ky = sorted(k for k in sd[y] if k.startswith(py_))
            same = [k[len(px):] for k in kx] == [k[len(py_):] for k in ky] and \
                all(torch.equal(sd[x][u], sd[y][v]) for u, v in zip(kx, ky))
            res[f"{x} {px}* = {y} {py_}*"] = same
        out["seeds"][s] = res
        if s == a1seeds.SEEDS[0]:
            from importlib import import_module
            net = import_module("arch.net" if a.code == "surface" else "bb.net")
            for arm, m in models.items():
                n_dec = sum(v.numel() for k, v in m.named_parameters() if not k.startswith("aux."))
                if a.code == "surface":
                    ro = net.head_macs(m.cfg, m.m)["inference"]["total"]
                    ro0 = net.head_macs(models[dep].cfg, m.m)["inference"]["total"]
                    tot = None
                else:
                    ro, ro0, tot = net.readout_macs(m), net.readout_macs(models[dep]), net.macs(m)["total"]
                out["sizes"].append({"arm": arm, "head": arms[arm][0], "training_only_readout": arms[arm][1],
                                     "params_decision_path": n_dec, "params_training": net.n_params(m),
                                     "readout_macs": ro, "vs_deployed": ro / ro0, "macs_total": tot})
    out["trunks_differ_between_seeds"] = not all(torch.equal(trunks[1][k], trunks[2][k]) for k in trunks[1])
    ok = all(all(v.values()) for v in out["seeds"].values()) and out["trunks_differ_between_seeds"] and \
        all(abs(r["vs_deployed"] - 1) <= 0.10 for r in out["sizes"] if r["head"] == "pooled")
    out["pass"] = ok
    json.dump(out, open(a.out, "w"), indent=1)
    print("MATCHED", json.dumps(out), flush=True)
    if not ok:
        raise SystemExit("matched conditions FAIL")


def traces(a):
    rows = {}
    for f, s in zip(a.files, a.seeds):
        rows[f] = (s, [(r["step"], r["x_sha1"], r["y_sha1"], r["lr"]) for r in csv.DictReader(open(f))])
    res = {"files": len(rows), "same_seed_equal": True, "other_seed_differ": True, "steps": {}}
    for f, (s, r) in rows.items():
        res["steps"][os.path.basename(f)] = len(r)
        for g, (t, q) in rows.items():
            if f >= g:
                continue
            n = min(len(r), len(q))
            eq = r[:n] == q[:n]
            if s == t and not eq:
                res["same_seed_equal"] = False
            if s != t and any(u[1] == v[1] for u, v in zip(r[:n], q[:n])):
                res["other_seed_differ"] = False
    res["pass"] = res["same_seed_equal"] and res["other_seed_differ"]
    json.dump(res, open(a.out, "w"), indent=1)
    print("TRACES", json.dumps(res), flush=True)
    if not res["pass"]:
        raise SystemExit("traces FAIL")


def bbchunks(a):
    """BB runs: the eval rows' (samples, chunk, chunk digest) of the arms of one seed are equal; other seeds' differ."""
    rows = {}
    for run, s in zip(a.runs, a.seeds):
        lg = json.load(open(os.path.join(os.environ["BB_CKPT"], run, "log.json")))
        rows[run] = (s, [(r["samples"], r["chunk"], r["chunk_sha"]) for r in lg["log"]], lg.get("stream_id"))
    res = {"runs": {k: {"seed": v[0], "stream_id": v[2], "evals": len(v[1])} for k, v in rows.items()},
           "same_seed_equal": all(v[1] == w[1] for v in rows.values() for w in rows.values() if v[0] == w[0]),
           "other_seed_differ": all(not set(x[2] for x in v[1]) & set(x[2] for x in w[1])
                                    for v in rows.values() for w in rows.values() if v[0] != w[0])}
    res["pass"] = res["same_seed_equal"] and res["other_seed_differ"] and all(v[1] for v in rows.values())
    json.dump(res, open(a.out, "w"), indent=1)
    print("BBCHUNKS", json.dumps(res), flush=True)
    if not res["pass"]:
        raise SystemExit("bbchunks FAIL")


def checks(a):
    out = []
    for run in a.runs:
        if a.code == "surface":
            p = os.path.join(os.environ["ICMP_CKPT"], run, "train_check.json")
            rec = json.load(open(p)) if os.path.exists(p) else {"run": run, "missing": True}
        else:
            p = os.path.join(os.environ["BB_CKPT"], run, "log.json")
            rec = json.load(open(p)).get("train_check", {"run": run, "missing": True}) if os.path.exists(p) else \
                {"run": run, "missing": True}
        out.append(rec)
    json.dump(out, open(a.out, "w"), indent=1)
    print("CHECKS", json.dumps(out), flush=True)


@torch.no_grad()
def graph(a):
    from a1 import read as a1read
    from a1 import sets
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    det, _, _ = sets.load(a.code, a.kind, sets.parse_range(a.chunks))
    det = det[:a.shots]
    res = {"network_us": {}, "rows": [], "note": "dry: a shared node, not A3's session"}
    for arm, s in zip(a.arms, a.seeds):
        f = os.path.join(a.reads, f"{arm}_s{s}_decision.pt")
        ck = torch.load(f, map_location="cpu", weights_only=False)
        if a.code == "surface":
            from arch import net
            model = net.net_from_cfg(ck["cfg"])
        else:
            from bb import net
            model = net.build(ck["cfg"])
        model.load_state_dict(ck["state"])
        model = model.to(dev).eval()
        x = torch.zeros(1, det.shape[1], dtype=torch.uint8, device=dev)
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                out = model(x)
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = model(x)
        y = out[0]
        diffs, worst, ts = 0, 0.0, []
        for i in range(len(det)):
            x.copy_(torch.as_tensor(det[i:i + 1], device=dev))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
            e = model(x)[0]
            if a.code == "surface":
                diffs += int((e > 0) != (y > 0))
                worst = max(worst, float((e - y).abs().max()))
            else:
                ce, cg = model.decide(e.float()), model.decide(y.float())
                diffs += int((ce[1] != cg[1]).any())
                worst = max(worst, float((ce[2] - cg[2]).abs().max()))
        us = 1e6 * float(np.median(ts))
        res["network_us"].setdefault(arm, {})[str(s)] = us
        res["rows"].append({"arm": arm, "seed": s, "shots": len(det), "graph_vs_eager_decisions_differ": diffs,
                            "max_abs_conf_diff": worst, "median_us": us, "mean_us": 1e6 * float(np.mean(ts)),
                            "p99_us": 1e6 * float(np.quantile(ts, 0.99))})
        print("GRAPH", json.dumps(res["rows"][-1]), flush=True)
    res["handoff_us"] = a.handoff_us
    json.dump(res, open(a.out, "w"), indent=1)
    if any(r["graph_vs_eager_decisions_differ"] for r in res["rows"]):
        raise SystemExit("graph FAIL: a CUDA graph's decision differs from eager")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["matched", "traces", "bbchunks", "checks", "graph"])
    ap.add_argument("--code", choices=["surface", "bb72"], default="surface")
    ap.add_argument("--trunks", default="")
    ap.add_argument("--design", default="data/bb/design_bb72.json")
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=[])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--reads", default="")
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--kind", default="")
    ap.add_argument("--chunks", default="")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--handoff-us", type=float, default=12.5, help="graph: the handoff written to the timing file")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"matched": matched, "traces": traces, "bbchunks": bbchunks, "checks": checks, "graph": graph}[a.cmd](a)


if __name__ == "__main__":
    main()
