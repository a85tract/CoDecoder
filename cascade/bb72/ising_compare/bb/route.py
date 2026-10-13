"""Confidence routing of ConvLines-BB (bb/DESIGN.md section 5, bb/PLAN.md 2.3-2.4 and 3): network reads through the
deployed engine, the calibration frontier and policy, the commit guard before any dev or test read, the paired
evaluation and the registered test size.
  infer     python -m bb.route infer --ckpt CKPT --code bb72 --p 0.003 --split cal --chunks 0-19 --tag TAG
            [--engine graph]; a dev or test read takes --policy POLICY instead (its engine; the commit guard). Per
            chunk BB_PREDS/<code>_p<p>_<split><j>_net-<TAG>.npz: pred (n, 12), margin (n,), fail (n,), sig (the engine
            signature, bb.engine), meta (checkpoint, run)
  frontier  python -m bb.route frontier --code bb72 --p 0.003 --chunks 0-19 --tag TAG --replica 1 --gpu-us G
            [--out policy.json]: on the calibration split only (DESIGN 5.1). Per routed fraction q of GRID: tau_q =
            the q-quantile of the margin, routed = margin < tau_q, failures = the network's on the kept shots + the
            fallback's on the routed ones, projected composite = G + H + (1 / N) x the fallback's recorded time
            on the routed shots, H = max(1.1 us, --handoff-us) (PLAN 2.7's provisional rule); admissible iff it is
            at most LIMIT_MS of the cell. Policy: among the admissible q the smallest with at most M - 2 sqrt(2 M)
            failures, M = round(0.90 x the fallback's failures), else the one with the fewest (ties to the smaller
            q); no admissible q, no policy. Screen (PLAN 2.5): a policy exists, it fails fewer times than the
            fallback, and the network alone fails at most 3 times as often as the fallback. status = pass or stop
  apply     python -m bb.route apply --policy POLICY --split dev --chunks 0-19 [--opponents ...] [--out A.json]: the
            routed decisions at the policy's tau, paired (exact McNemar, Newcombe 95 %) against each opponent on the
            same shots, plus the network alone (the commit guard for dev and test)
  size      python -m bb.route size --apply A.json [--out S.json]: the registered test size of the cell (PLAN 3.3)
            from the test replica's dev read against the gating opponents
  pair      python -m bb.route pair --code bb144 --p 0.004 --split monitor --chunks 10-14 --tags A,B
            [--min-reduction 0.25] [--out P.json]: gross2's exploratory arm comparison (bb/PLAN_gross2.md; never dev
            or test): two network reads of the same shots paired as in apply (alpha 0.05); reduction = 1 - fail_a /
            fail_b, qualifies = reduction >= --min-reduction with the Newcombe 95 % interval of LER(a) - LER(b)
            entirely below zero
Commit guard (PLAN 2.0, 2.5): a dev or test read runs only in a clean deployed tree (DEPLOYED without "+dirty") holding the
policy file of every cell and replica of the code (CELLS) in data/bb/policies, each with the sha256 that
bb/scripts/deploy.sh recorded from the deployed commit (DEPLOYED_POLICIES), so a policy created or edited after the
deployment never passes; both policies of the cell read must have passed their screen, the checkpoint must be the
policy's network and the engine signature the policy's. apply on the test split uses ALPHA_TEST (PLAN 3.2)."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import types

import numpy as np

from bb import opp, shots

GRID = [0, .005, .01, .02, .03, .04, .06, .08, .10, .12, .15, .20]
FINE = [round(.001 * k, 3) for k in range(2, 11)]        # --fine (gross2 stage d): q = 0.2 % ... 1.0 % in 0.1 % steps
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POLICIES = os.path.join(REPO, "ising_compare", "data", "bb", "policies")
CELLS = {"bb72": (0.003, 0.002), "bb144": (0.004,)}
REPLICAS = (1, 2)
GATING = ("relay", "bposd", "gpu_bposd")
# PLAN 2.7, frozen 2026-10-06: the provisional admissibility limit of each cell = the fastest gating opponent's stage-1
# development batch-1 time (DESIGN 6: cudaq-qec on a Derecho A100, the smaller of its stage-1 means)
LIMIT_MS = {("bb72", 0.003): 1.06, ("bb72", 0.002): 0.63, ("bb144", 0.004): 15.8}
HANDOFF_US = 1.1
ALPHA_DEV, ALPHA_TEST, POWER = 0.05, 0.05 / 3, 0.90       # PLAN 2.6 and 3.2-3.3
FLOOR = {("bb72", 0.003): 200_000, ("bb72", 0.002): 300_000, ("bb144", 0.004): 100_000}
CAP = {"bb72": 1_000_000, "bb144": 300_000}


def _newcombe():
    """qldpc_neural/bench/paired.py's newcombe10, loaded as a file (its torch-importing qnd.train import is stubbed)."""
    if "qnd.train" not in sys.modules:
        tr = types.ModuleType("qnd.train")
        tr.parse_range = shots.parse_range
        sys.modules["qnd.train"] = tr
    spec = importlib.util.spec_from_file_location("qldpc_paired", os.path.join(REPO, "qldpc_neural", "bench", "paired.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.newcombe10


def paired(fa, fb, alpha=ALPHA_DEV):
    """fa, fb boolean failures on the same shots: LER(a) - LER(b), Newcombe (1 - alpha) interval, exact McNemar p."""
    from scipy.stats import binomtest, norm
    fa, fb = np.asarray(fa, bool), np.asarray(fb, bool)
    n = len(fa)
    both, oa, ob = int((fa & fb).sum()), int((fa & ~fb).sum()), int((~fa & fb).sum())
    p = binomtest(oa, oa + ob, 0.5).pvalue if oa + ob else 1.0
    dt, lo, hi = _newcombe()(both, oa, ob, n - both - oa - ob, float(norm.ppf(1 - alpha / 2)))
    verdict = "better" if (hi < 0 and p < alpha) else ("worse" if (lo > 0 and p < alpha) else "level")
    return dict(shots=n, fail_a=int(fa.sum()), fail_b=int(fb.sum()), only_a=oa, only_b=ob, mcnemar_p=p, diff_ler=dt,
                diff_lo=lo, diff_hi=hi, alpha=alpha, verdict=verdict)


def net_path(code, p, split, j, tag):
    return opp.pred_path(code, p, split, j, f"net-{tag}")


def policy_name(code, p, replica):
    return f"{code}_p{p}_rep{replica}.json"


def guard(path):
    """The commit guard of a dev or test read: the policy at path, or SystemExit."""
    dep, man = (os.path.join(REPO, f) for f in ("DEPLOYED", "DEPLOYED_POLICIES"))
    tag = open(dep).read().strip() if os.path.exists(dep) else ""
    if not tag or tag.endswith("+dirty"):
        raise SystemExit(f"commit guard: the deployed tree is not a clean commit ({tag or 'no DEPLOYED file'})")
    lines = open(man).read().split("\n") if os.path.exists(man) else [""]
    if not lines[0].startswith(f"commit {tag}"):
        raise SystemExit(f"commit guard: no DEPLOYED_POLICIES of the deployed commit {tag}")
    committed = {ln.split()[1]: ln.split()[0] for ln in lines[1:] if ln.strip()}     # file name -> sha256

    def read(f):                                         # the bytes deploy.sh hashed from the commit, or SystemExit
        fp = os.path.join(POLICIES, f)
        raw = open(fp, "rb").read() if os.path.exists(fp) else b""
        if committed.get(f) != hashlib.sha256(raw).hexdigest():
            raise SystemExit(f"commit guard: {f} is not a policy of the deployed commit {tag} (missing, created or "
                             f"edited after the deployment)")
        return json.loads(raw)
    if os.path.dirname(os.path.realpath(path)) != os.path.realpath(POLICIES):
        raise SystemExit(f"commit guard: {path} is not a committed policy in data/bb/policies")
    pol = read(os.path.basename(path))
    code, p = pol["code"], pol["p"]
    if os.path.basename(path) != policy_name(code, p, pol["replica"]):
        raise SystemExit(f"commit guard: {path} is not named as its cell and replica")
    pols = {(q, r): read(policy_name(code, q, r)) for q in CELLS[code] for r in REPLICAS}
    for r in REPLICAS:
        if pols[(p, r)]["status"] != "pass":
            raise SystemExit(f"commit guard: {policy_name(code, p, r)} did not pass its calibration screen")
    return pol


def infer(a):
    import torch
    from bb import engine
    dev = torch.device("cuda")
    pol = None
    if a.split in ("dev", "test"):
        if not a.policy:
            raise SystemExit("a dev or test read needs --policy (the commit guard)")
        pol = guard(a.policy)
        if (pol["code"], pol["p"], pol["tag"]) != (a.code, a.p, a.tag):
            raise SystemExit("commit guard: --code, --p and --tag must be the policy's")
    model, ck = engine.load_net(a.ckpt, dev, a.allow_incomplete)
    if ck["cfg"]["code"] != a.code:
        raise SystemExit(f"{a.ckpt} is a {ck['cfg']['code']} network")
    sha = engine.sha256_file(a.ckpt)
    if pol and sha != pol["engine"]["net_sha256"]:
        raise SystemExit("commit guard: the checkpoint is not the policy's network")
    eng = engine.Deployed(model, pol["engine"]["kind"] if pol else a.engine, sha)
    sig = eng.signature()
    meta = {"ckpt": os.path.abspath(a.ckpt), "run": ck["hp"]["run"], "run_id": ck["hp"]["run_id"],
            "samples": ck.get("samples"), "status": ck.get("status")}
    if pol:
        diff = {k for k in set(sig) | set(pol["engine"]) if sig.get(k) != pol["engine"].get(k)}
        if diff and not (a.gpu and diff == {"gpu"} and sig["gpu"] == a.gpu):   # STAGE3_PLAN.md 0: the H100, disclosed
            raise SystemExit(f"commit guard: the engine signature differs from the policy's: "
                             f"{ {k: (sig.get(k), pol['engine'].get(k)) for k in diff} }")
        if diff:
            meta["registered_gpu"] = pol["engine"]["gpu"]
    os.makedirs(opp.PREDS, exist_ok=True)
    tot = n = 0
    for j in shots.parse_range(a.chunks):
        det, obs = shots.load(a.code, a.p, a.split, j)
        bits, margin = eng.run(det)
        fail = (bits != obs).any(1)
        np.savez(net_path(a.code, a.p, a.split, j, a.tag), pred=bits, margin=margin, fail=fail,
                 sig=json.dumps(sig, sort_keys=True), meta=json.dumps(meta))
        tot += int(fail.sum())
        n += len(fail)
        blind = a.split == "test"                          # STAGE3_PLAN.md 5: no test count before the analysis
        print(f"net-{a.tag} {a.split} chunk {j}: {'stored' if blind else f'{int(fail.sum())} / {len(fail)} failures'}",
              flush=True)
    print(json.dumps({"tag": a.tag, "split": a.split, "chunks": a.chunks, "shots": n,
                      "failures": "" if a.split == "test" else tot, "engine": sig, "meta": meta}), flush=True)


def _load_net(code, p, split, chunks, tag):
    """(pred, margin, engine signature, meta) of a network read; every chunk must carry the same signature."""
    z = [np.load(net_path(code, p, split, j, tag)) for j in chunks]
    if any("sig" not in x.files for x in z):
        raise SystemExit(f"net-{tag}: a read without an engine signature (stage 1); read it again through bb.engine")
    sigs = {str(x["sig"]) for x in z}
    if len(sigs) != 1:
        raise SystemExit(f"net-{tag}: chunks read with different engine signatures")
    return (np.concatenate([x["pred"] for x in z]), np.concatenate([x["margin"] for x in z]), json.loads(sigs.pop()),
            json.loads(str(z[0]["meta"])))


def frontier(a):
    if a.split != "cal":
        raise SystemExit("the frontier is fitted on the calibration split only")
    chunks = shots.parse_range(a.chunks)
    obs = shots.load_many(a.code, a.p, "cal", chunks)[1]
    bits, margin, sig, meta = _load_net(a.code, a.p, "cal", chunks, a.tag)
    if sig["kind"] == "fp32":
        raise SystemExit("a policy is fitted on a deployed engine's read only")
    fb, fsec = opp.load_preds(a.code, a.p, "cal", chunks, a.fallback)
    nf, ff = (bits != obs).any(1), (fb != obs).any(1)
    limit, handoff = LIMIT_MS[(a.code, a.p)], max(HANDOFF_US, a.handoff_us)
    rows = []
    for q in sorted(set(GRID) | set(FINE)) if a.fine else GRID:
        tau = float(np.quantile(margin, q)) if q > 0 else -math.inf
        routed = margin < tau
        r = {"q": q, "tau": tau, "routed": float(routed.mean()), "failures": int((nf & ~routed).sum() + (ff & routed).sum()),
             "fallback_ms_routed_mean": float(1e3 * fsec[routed].mean()) if routed.any() else 0.0,
             "composite_ms": (a.gpu_us + handoff) / 1e3 + float(1e3 * fsec[routed].sum()) / len(obs)}
        r["admissible"] = r["composite_ms"] <= limit
        rows.append(r)
    M = round(0.90 * int(ff.sum()))
    cand = [r for r in rows if r["admissible"]]
    ok = [r for r in cand if r["failures"] <= M - 2 * math.sqrt(2 * M)]
    pol = (ok[0] if ok else min(cand, key=lambda r: (r["failures"], r["q"]))) if cand else None
    screen = {"policy_exists": pol is not None, "policy_beats_fallback": bool(pol) and pol["failures"] < int(ff.sum()),
              "net_within_3x_fallback": int(nf.sum()) <= 3 * int(ff.sum())}
    out = {"code": a.code, "p": a.p, "split": "cal", "chunks": a.chunks, "tag": a.tag, "replica": a.replica,
           "network": meta, "engine": sig, "fallback": a.fallback, "shots": len(obs), "net_alone": int(nf.sum()),
           "fallback_alone": int(ff.sum()), "target": M, "gpu_us": a.gpu_us, "handoff_us": handoff,
           "limit_ms": limit, "sources": {"gpu_us": a.gpu_src, "limit_ms": "LIMIT_MS (PLAN 2.7, frozen)",
                                          "cpu": f"the {a.fallback} cal read's recorded per-shot times (bb.opp)"},
           "frontier": rows, "policy": pol, "screen": screen, "status": "pass" if all(screen.values()) else "stop",
           "rank_key": [pol["failures"], pol["composite_ms"]] if pol else None}
    for r in rows:
        print(json.dumps(r), flush=True)
    print("policy", json.dumps(pol), "net alone", int(nf.sum()), a.fallback, int(ff.sum()), flush=True)
    print("screen", json.dumps(screen), "status", out["status"], flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)


def apply(a):
    pol = guard(a.policy) if a.split in ("dev", "test") else json.load(open(a.policy))
    code, p, tag, fbn, tau = pol["code"], pol["p"], pol["tag"], pol["fallback"], pol["policy"]["tau"]
    chunks = shots.parse_range(a.chunks)
    obs = shots.load_many(code, p, a.split, chunks)[1]
    bits, margin, sig, _ = _load_net(code, p, a.split, chunks, tag)
    if sig != pol["engine"]:
        raise SystemExit("the read's engine signature differs from the policy's")
    fb, _ = opp.load_preds(code, p, a.split, chunks, fbn)
    routed = margin < tau
    ours = (np.where(routed[:, None], fb, bits) != obs).any(1)
    alpha = ALPHA_TEST if a.split == "test" else ALPHA_DEV  # PLAN 3.2: each comparison of the test at 0.05 / 3
    res = {"policy": os.path.abspath(a.policy), "replica": pol["replica"], "tau": tau, "split": a.split,
           "chunks": a.chunks, "shots": len(obs), "routed": float(routed.mean()), "failures": int(ours.sum()),
           "network_alone": int((bits != obs).any(1).sum()), "alpha": alpha, "vs": {}}
    for o in a.opponents.split(","):
        op, _ = opp.load_preds(code, p, a.split, chunks, o)
        res["vs"][o] = paired(ours, (op != obs).any(1), alpha)
    if all(o in res["vs"] for o in GATING):
        res["accuracy_gate"] = all(res["vs"][o]["verdict"] == "better" for o in GATING)   # PLAN 2.6 (i), 3.2
    print(json.dumps(res, indent=1), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


def pair(a):
    if a.split in ("dev", "test"):
        raise SystemExit("pair compares exploratory reads only, never dev or test")
    chunks = shots.parse_range(a.chunks)
    obs = shots.load_many(a.code, a.p, a.split, chunks)[1]
    ta, tb = a.tags.split(",")
    (pa, _, sa, _), (pb, _, sb, _) = (_load_net(a.code, a.p, a.split, chunks, t) for t in (ta, tb))
    r = dict(a=ta, b=tb, kind_a=sa["kind"], kind_b=sb["kind"], split=a.split, chunks=a.chunks,
             decisions_differ=int((pa != pb).any(1).sum()), **paired((pa != obs).any(1), (pb != obs).any(1)))
    r["ratio"] = r["fail_b"] / max(1, r["fail_a"])
    r["reduction"] = 1 - r["fail_a"] / max(1, r["fail_b"])
    r["qualifies"] = r["reduction"] >= a.min_reduction and r["diff_hi"] < 0
    print("pair", json.dumps(r), flush=True)
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)


def _newcombe_hi(a, b, c, d, z):
    """Vectorised upper end of paired.py's newcombe10 for LER(1) - LER(2) (a both fail, b only 1, c only 2, d neither)."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n

    def wilson(k):
        p = k / n
        den = 1 + z * z / n
        ctr = (p + z * z / (2 * n)) / den
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return ctr - h, ctr + h
    _, u1 = wilson(a + b)
    l2, _ = wilson(a + c)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = np.where(den > 0, np.maximum(0.0, a * d - b * c) / np.sqrt(np.maximum(den, 1)), 0.0)
    return p1 - p2 + np.sqrt(np.maximum(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))


def power(cnt, n, alpha=ALPHA_TEST, reps=20000, seed=20261006):
    """P(the registered test decision passes at n shots): cnt = (both fail, only ours, only the opponent's, shots)
    from dev; the decision = exact two-sided McNemar p < alpha and the Newcombe (1 - alpha) interval entirely below 0;
    by simulation with a fixed seed (deterministic)."""
    from scipy.stats import binom, norm
    both, oa, ob, N = cnt
    X = np.random.default_rng(seed).multinomial(n, np.array([both, oa, ob, N - both - oa - ob]) / N, size=reps)
    a, b, c, d = (X[:, i].astype(np.float64) for i in range(4))
    m = b + c
    pm = np.where(m > 0, np.minimum(1.0, 2 * binom.cdf(np.minimum(b, c), m, 0.5)), 1.0)
    z = float(norm.ppf(1 - alpha / 2))
    hi = _newcombe_hi(a, b, c, d, z)
    nc = _newcombe()
    for i in range(5):                                     # the vectorised interval must equal paired.py's
        assert abs(hi[i] - nc(int(a[i]), int(b[i]), int(c[i]), int(d[i]), z)[2]) < 1e-12
    return float(np.mean((pm < alpha) & (hi < 0)))


def size(a):
    """PLAN 3.3: per gating opponent o, n_o = the smallest multiple of 10,000 from the cell's floor up to its cap at
    which the test replica's comparison with o passes with probability >= 1 - (1 - POWER) / 3 (so all three pass
    with probability >= POWER); n = max n_o; any n_o above the cap: the cell is not tested."""
    res = json.load(open(a.apply))
    pol = json.load(open(res["policy"]))
    cell = (pol["code"], pol["p"])
    target = 1 - (1 - POWER) / len(GATING)
    out = {"apply": os.path.abspath(a.apply), "cell": list(cell), "replica": pol["replica"], "alpha": ALPHA_TEST,
           "per_comparison_power": target, "floor": FLOOR[cell], "cap": CAP[cell[0]], "opponents": {}}
    try:
        pols = [json.load(open(os.path.join(POLICIES, policy_name(*cell, r)))) for r in REPLICAS]
        best = min(pols, key=lambda q: (q["rank_key"][0], q["rank_key"][1], q["replica"]))["replica"]
        out["test_replica"] = best
        if best != pol["replica"]:
            raise SystemExit(f"replica {pol['replica']} is not the cell's test replica ({best}, PLAN 3.1)")
    except FileNotFoundError:
        out["test_replica"] = "not checked: the cell's policies are not in data/bb/policies"
    for o in GATING:
        r = res["vs"][o]
        cnt = (r["fail_a"] - r["only_a"], r["only_a"], r["only_b"], r["shots"])
        n_o = next((n for n in range(FLOOR[cell], CAP[cell[0]] + 1, 10_000) if power(cnt, n) >= target), None)
        out["opponents"][o] = {"dev_counts": cnt, "n": n_o, "power_at_cap": power(cnt, CAP[cell[0]])}
    ns = [v["n"] for v in out["opponents"].values()]
    out["n"] = None if None in ns else max(ns)
    out["decision"] = f"test {out['n']} shots" if out["n"] else "not tested: underpowered at the cap"
    print(json.dumps(out, indent=1), flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["infer", "frontier", "apply", "size", "pair"])
    ap.add_argument("--ckpt")
    ap.add_argument("--code")
    ap.add_argument("--p", type=float)
    ap.add_argument("--split", default="cal")
    ap.add_argument("--chunks")
    ap.add_argument("--tag")
    ap.add_argument("--engine", default="graph", help="infer on cal: graph | graph32 | inductor | fp32 (bb.engine)")
    ap.add_argument("--allow-incomplete", action="store_true", help="smoke tests only: read an unfinished run")
    ap.add_argument("--fine", action="store_true", help="frontier: add q = 0.2 %% ... 1.0 %% in 0.1 %% steps (gross2)")
    ap.add_argument("--gpu", default="", help="infer: the disclosed GPU of a stage-3 read (STAGE3_PLAN.md 0); the "
                    "engine signature may differ from the policy's in 'gpu' only, and must name this GPU")
    ap.add_argument("--fallback", default="relay")
    ap.add_argument("--replica", type=int, choices=REPLICAS)
    ap.add_argument("--gpu-us", type=float, help="frontier: the provisional GPU term, us per batch-1 call")
    ap.add_argument("--gpu-src", default="", help="frontier: where --gpu-us comes from (rows, job)")
    ap.add_argument("--handoff-us", type=float, default=0.0, help="frontier: the measured handoff, us (max with 1.1)")
    ap.add_argument("--policy")
    ap.add_argument("--opponents", default="relay,bposd,gpu_bposd,relay_full")
    ap.add_argument("--apply")
    ap.add_argument("--tags", help="pair: TAG_A,TAG_B")
    ap.add_argument("--min-reduction", type=float, default=0.25, help="pair: the reduction a qualifying arm needs")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.cmd == "frontier" and (a.gpu_us is None or a.replica is None):
        raise SystemExit("frontier needs --gpu-us and --replica")
    {"infer": infer, "frontier": frontier, "apply": apply, "size": size, "pair": pair}[a.cmd](a)


if __name__ == "__main__":
    main()
