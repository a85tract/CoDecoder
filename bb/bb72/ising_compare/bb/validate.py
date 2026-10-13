"""Engine validation and choice (bb/PLAN.md 2.4), on the registered GPU model (PLAN 2.8, Q9):
  check   python -m bb.validate check --code bb72 --p 0.003 --a TAG_A --b TAG_B --ref TAG_F --gpu MODEL [--out V.jsonl]
          two fresh processes' reads of a cell's stored monitor chunk 0 (never cal or dev; `bb.route infer --split
          monitor --chunks 0 --engine KIND`) and the fp32 module's read of it: (1) the two are bitwise equal (decisions
          and margins); (2) the decisions agree with fp32 on at least 99.9 % of shots; (3) the margin differs from
          fp32's by at most 5 % of fp32's median margin on at least 99 % of shots; (4) the signature is a batch-1 read
          on a whole card (no MIG slice) of MODEL. The thresholds catch implementation bugs only. Reported beside, not
          gated: the routed sets at a few q (each read's own q-quantile of its margin) against fp32's.
  early   python -m bb.validate early --rows T.csv [--out E.json]: the port's early timing check, from bb.latency rows
          with random weights (--design) in this session: per rung the slowest process's GPU term; the fused engine
          passes iff it is at most 40 % of the faster of graph and inductor
  choose  python -m bb.validate choose --checks V.jsonl --rows T.csv --code bb72 --nets SHA,SHA --ps 0.003,0.002
          [--early E.json] --out C.json: per code the validated rung (its check passed on every network and cell) with the smallest provisional GPU term
          (PLAN 2.7: per network the slowest of its processes in this session; per rung the larger over the code's
          networks); the fused rung only if it passed the early check. Also the provisional GPU term and handoff of
          each network under the chosen rung (bb.route frontier --gpu-us, --handoff-us).
  add     python -m bb.validate add --choice C.json --checks V.jsonl --rows T.csv --code bb144 --nets SHA --ps 0.004
          --out A.json: a network that comes after its code's choice (gross run 22; the choice is fixed before the
          code's first cal read): the chosen rung's check must pass on its every cell and its 3 processes give its
          provisional GPU term and handoff (written to A.json, in choose's "provisional" form). C.json is only read:
          no rung is re-chosen."""
import argparse
import csv
import json
import os

import numpy as np

from bb import route
from bb.timing import SESSION

AGREE, MARGIN_FRAC, MARGIN_SHARE, EARLY = 0.999, 0.05, 0.99, 0.40
Q_INFO = (0.01, 0.02, 0.05, 0.10)


def _read(code, p, tag, split="monitor"):
    z = np.load(route.net_path(code, p, split, 0, tag))
    return z["pred"], z["margin"], json.loads(str(z["sig"]))


def check(a):
    pa, ma, sa = _read(a.code, a.p, a.a, a.split)
    pb, mb, sb = _read(a.code, a.p, a.b, a.split)
    pf, mf, sf = _read(a.code, a.p, a.ref, a.split)
    med = float(np.median(mf))
    d = np.abs(ma.astype(np.float64) - mf)
    r = {"code": a.code, "p": a.p, "kind": sa["kind"], "net_sha256": sa["net_sha256"], "shots": len(pa),
         "bitwise": bool(np.array_equal(pa, pb) and np.array_equal(ma.view(np.uint32), mb.view(np.uint32))
                         and sa == sb),
         "agree": float((pa == pf).all(1).mean()), "fp32_median_margin": med,
         "margin_within": float((d <= MARGIN_FRAC * med).mean()), "margin_absdiff_p99": float(np.quantile(d, 0.99)),
         "gpu": sa["gpu"], "batch": sa["batch"], "ref_kind": sf["kind"], "same_net": sa["net_sha256"] == sf["net_sha256"]}
    r["split"] = a.split
    r["fail"] = int(np.load(route.net_path(a.code, a.p, a.split, 0, a.a))["fail"].sum())
    r["fail_fp32"] = int(np.load(route.net_path(a.code, a.p, a.split, 0, a.ref))["fail"].sum())
    r["routing_agree"] = {str(q): float(((ma < np.quantile(ma, q)) == (mf < np.quantile(mf, q))).mean()) for q in Q_INFO}
    r["sig_ok"] = (sa["batch"] == 1 and "MIG" not in sa["gpu"] and a.gpu in sa["gpu"] and sf["kind"] == "fp32"
                   and r["same_net"])
    r["pass"] = bool(r["bitwise"] and r["agree"] >= AGREE and r["margin_within"] >= MARGIN_SHARE and r["sig_ok"])
    print("check", json.dumps(r), flush=True)
    if a.out:
        with open(a.out, "a") as fh:
            fh.write(json.dumps(r) + "\n")


def _gpu_terms(path):
    """{(kind, net sha prefix): (slowest GPU term ms, slowest handoff ms, processes)} of this session's bb.latency rows."""
    out = {}
    for r in csv.DictReader(open(path)):
        if r["session"] != SESSION or r["role"] not in ("gpu_term", "handoff"):
            continue
        k = tuple(r["name"].split(":", 1))
        g, h, n = out.get(k, (0.0, 0.0, set()))
        v = float(r["value_ms"])
        out[k] = (max(g, v), h, n | {r["process"]}) if r["role"] == "gpu_term" else (g, max(h, v), n)
    return out


def early(a):
    t = _gpu_terms(a.rows)
    ms = {k: v[0] for (k, s), v in t.items() if s.startswith("random") and len(v[2]) >= 3}
    r = {"gpu_term_ms": ms, "rule": f"fused <= {EARLY} x min(graph, inductor), slowest of 3 processes each"}
    r["pass"] = bool("fused" in ms and "graph" in ms and "inductor" in ms
                     and ms["fused"] <= EARLY * min(ms["graph"], ms["inductor"]))
    r["ratio"] = ms["fused"] / min(ms["graph"], ms["inductor"]) if {"fused", "graph", "inductor"} <= set(ms) else None
    print("early", json.dumps(r), flush=True)
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)


def choose(a):
    checks = [json.loads(x) for x in open(a.checks)]
    nets = a.nets.split(",")
    ps = [float(x) for x in a.ps.split(",")]
    early_ok = json.load(open(a.early))["pass"] if a.early and os.path.exists(a.early) else False
    t = _gpu_terms(a.rows)
    kinds = sorted({c["kind"] for c in checks})
    rungs = {}
    for k in kinds:
        cs = [c for c in checks if c["kind"] == k and c["code"] == a.code]
        ok = (len(cs) == len(nets) * len(ps) and all(c["pass"] for c in cs) and (k != "fused" or early_ok)
              and {(c["net_sha256"], float(c["p"])) for c in cs} == {(n, p) for n in nets for p in ps})
        terms = [t.get((k, n[:12])) for n in nets]
        ok = ok and all(x is not None and len(x[2]) >= 3 for x in terms)
        rungs[k] = {"validated": ok, "checks": len(cs), "gpu_term_ms": max(x[0] for x in terms) if ok else None,
                    "per_net": {n[:12]: (x[0], x[1]) for n, x in zip(nets, terms) if x}}
    ok = {k: v for k, v in rungs.items() if v["validated"]}
    pick = min(ok, key=lambda k: ok[k]["gpu_term_ms"]) if ok else None
    r = {"code": a.code, "session": SESSION, "rungs": rungs, "early_pass": early_ok, "choice": pick,
         "provisional": ({n[:12]: {"gpu_us": 1e3 * rungs[pick]["per_net"][n[:12]][0],
                                   "handoff_us": 1e3 * rungs[pick]["per_net"][n[:12]][1]} for n in nets} if pick else None)}
    print("choose", json.dumps(r), flush=True)
    json.dump(r, open(a.out, "w"), indent=1)
    if pick is None:
        raise SystemExit("no rung validated: no cal read (PLAN 2.4)")


def add(a):
    c = json.load(open(a.choice))
    k, nets, ps = c["choice"], a.nets.split(","), [float(x) for x in a.ps.split(",")]
    cs = [x for x in map(json.loads, open(a.checks)) if x["kind"] == k and x["code"] == a.code]
    t = _gpu_terms(a.rows)
    terms = [t.get((k, n[:12])) for n in nets]
    ok = (c["code"] == a.code and len(cs) == len(nets) * len(ps) and all(x["pass"] for x in cs)
          and {(x["net_sha256"], float(x["p"])) for x in cs} == {(n, p) for n in nets for p in ps}
          and all(x is not None and len(x[2]) >= 3 for x in terms))
    r = {"code": a.code, "session": SESSION, "choice": k, "choice_session": c["session"], "checks": len(cs),
         "validated": ok, "provisional": ({n[:12]: {"gpu_us": 1e3 * x[0], "handoff_us": 1e3 * x[1]}
                                          for n, x in zip(nets, terms)} if ok else None)}
    print("add", json.dumps(r), flush=True)
    json.dump(r, open(a.out, "w"), indent=1)
    if not ok:
        raise SystemExit(f"the chosen rung {k} is not validated on {a.nets}: no cal read (PLAN 2.4)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--code", required=True)
    c.add_argument("--p", type=float, required=True)
    for k in ("a", "b", "ref", "gpu"):
        c.add_argument(f"--{k}", required=True)
    c.add_argument("--out", default="")
    c.add_argument("--split", default="monitor", help="the stored chunk 0 read (stage 3 on the H100: cal)")
    e = sub.add_parser("early")
    e.add_argument("--rows", required=True)
    e.add_argument("--out", default="")
    h = sub.add_parser("choose")
    for k in ("checks", "rows", "code", "nets", "ps", "out"):
        h.add_argument(f"--{k}", required=True)
    h.add_argument("--early", default="")
    d = sub.add_parser("add")
    for k in ("choice", "checks", "rows", "code", "nets", "ps", "out"):
        d.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    {"check": check, "early": early, "choose": choose, "add": add}[a.cmd](a)


if __name__ == "__main__":
    main()
