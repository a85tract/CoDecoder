"""D9-SHALLOW's registered analysis (PLAN.md sections 1, 5 and 6). Deterministic (no clock, no randomness, sorted keys): it
runs twice and the two outputs must be byte-identical.
    PYTHONPATH=<tree>/surface_v2:<tree> python d9_shallow/analysis_d9s.py --records RECORDS --out OUT --commit F
RECORDS (d9_shallow/collect_d9s.sh lays it out):
  testsets/   beacon_d9s.json, d9s_test.json, d9s_timing.json (commit B; must equal the tree's d9_shallow/testsets/)
  shots/      d9stest/ (100 chunks), d9stiming/ (20 chunks), as drawn on the pod
  pod/        the GPU session: pre/prescreen.json, attempts.json, session.json, masks/, the gpu_term records, watch/,
              handoff/surface/low/ (the mask sent to the Linux workstation)
  reads/      d9stest_base.npz/.json, d9stest_d9_b1w64_sh.npz/.json, d9stest_d9_b3_sv2s1.npz/.json
  cpu/        the CPU session on the Linux workstation as speed_rt/cpu_ws/export.py writes it (cpu_session.json,
              cpu_attempts.json, cpu/<slot>/<record>.json)
OUT/analysis.json (every quantity, every check, every reason a quantity is missing, the sha256 of every input),
OUT/fit_d9s.json (this session's Grace fit, speed_rt.analysis_s.fit_s2 unchanged), OUT/table.md.

The claim (B1's pair = d9_b1w64_sh at q = 0.06, tau 2.337686538696289, routed shots to correlated PM): all of
  (1) its composite a + b + c on the Grace basis is strictly below plain PM's 26.296678815736907 us;
  (2) it fails less often than plain PM: exact two-sided McNemar p < 0.025 and the whole 97.5 % Newcombe interval of
      the difference below 0;
  (3) the same against correlated PM, and its composite strictly below correlated PM's 63.916141817786674 us.
a = the slowest of the 5 accepted GPU processes; b = the slowest of the 3 accepted CPU processes' term_us (correlated
PM's time over the routed timing shots divided by all 200,000 timing shots), times this session's applied factor for
correlated PM; c = max(1.1 us, the router's estimate) of the accepted handoff process. Missing evidence makes a part
"not established", with no fallback value."""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "surface_v2"), ROOT, HERE]

import numpy as np  # noqa: E402

import beacon_d9s  # noqa: E402
from convlines import circuits, stats, testset  # noqa: E402
from speed_rt import analysis_s as AS  # noqa: E402
from timing import deployed  # noqa: E402

FZ = json.load(open(os.path.join(HERE, "freeze_d9s.json")))
D = 9
B1, B3 = "d9_b1w64_sh", "d9_b3_sv2s1"
NETS = (B1, B3)
TAU = FZ["networks"][B1]["tau"]
ALPHA = FZ["claim"]["alpha"]
SPLIT = {"test": "d9stest", "timing": "d9stiming"}


class Missing(Exception):
    pass


def need(c, why):
    if not c:
        raise Missing(why)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def load(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError) as e:
        raise Missing(f"{os.path.basename(p)}: {e.__class__.__name__}")


def shots(R, commit):
    """The beacon record, both manifests (equal to the tree's) and every chunk: seeds re-derived, exact coverage, sha256,
    the set's digest. -> ({set: manifest}, the test set's observables)."""
    b = beacon_d9s.load(os.path.join(R, "testsets", "beacon_d9s.json"), commit)
    for f in ("beacon_d9s.json", "d9s_test.json", "d9s_timing.json"):
        need(sha256_file(os.path.join(R, "testsets", f)) == sha256_file(os.path.join(HERE, "testsets", f)),
             f"{f}: not commit B's file")
    mans, obs_test = {}, None
    for s, n in beacon_d9s.CHUNKS.items():
        m = load(os.path.join(R, "testsets", f"d9s_{s}.json"))
        need(m["set"] == s and m["split"] == SPLIT[s] and m["chunks"] == n and m["d"] == D and m["p"] == circuits.P,
             f"{s}: the manifest's set")
        need(m["beacon"] == dict(round=b["round"], randomness=b["randomness"], freeze_commit=b["freeze_commit"]),
             f"{s}: the manifest names another beacon")
        need(m["circuit_sha256"] == circuits.CIRCUIT_SHA256[D] == FZ["circuit"]["sha256"], f"{s}: the circuit")
        fs = sorted(m["files"].items(), key=lambda kv: kv[1]["j"])
        need([v["j"] for _, v in fs] == list(range(n)), f"{s}: the chunks are not j = 0..{n - 1}")
        need(all(f == testset.name(SPLIT[s], D, v["j"]) and v["seed"] == b["seeds"][s][v["j"]] and v["set"] == s
                 for f, v in fs), f"{s}: a chunk's name or seed does not re-derive from the beacon")
        d = os.path.join(R, "shots", SPLIT[s])
        need(sorted(os.listdir(d)) == sorted(m["files"]), f"{s}: the shot directory does not hold exactly the manifest's chunks")
        sh = [testset.load(os.path.join(d, f), D, v["sha256"]) for f, v in fs]
        det, obs = np.concatenate([x[0] for x in sh]), np.concatenate([x[1] for x in sh])
        need(len(obs) == n * testset.CHUNK == m["shots"], f"{s}: {len(obs)} shots")
        need(deployed.shots_sha256(det, obs) == m["shots_sha256"], f"{s}: the set's digest")
        mans[s] = m
        if s == "test":
            obs_test = obs
    return b, mans, obs_test


def reads(R, man, obs):
    """The four decoders' answers on the test shots, aligned with the shot files' observables."""
    z = np.load(os.path.join(R, "reads", "d9stest_base.npz"))
    j = load(os.path.join(R, "reads", "d9stest_base.json"))
    need(np.array_equal(z["obs"], obs), "base reads: the observables differ from the shot files'")
    need(j["shots_sha256"] == man["shots_sha256"] and j["code_digest"] == FZ["code_digest"], "base reads: shots or code")
    need(len(z["pm"]) == len(z["cpm"]) == len(obs), "base reads: length")
    out = dict(pm=z["pm"].astype(np.uint8), cpm=z["cpm"].astype(np.uint8))
    rec = {"base": j}
    for n in NETS:
        z = np.load(os.path.join(R, "reads", f"d9stest_{n}.npz"))
        j = load(os.path.join(R, "reads", f"d9stest_{n}.json"))
        need(np.array_equal(z["obs"], obs) and len(z["c"]) == len(obs) and z["c"].dtype == np.float32, f"{n} reads: shots")
        need(j["network"] == n and j["network_sha256"] == FZ["networks"][n]["sha256"], f"{n} reads: the weights")
        need(j["shots_sha256"] == man["shots_sha256"] and j["code_digest"] == FZ["code_digest"], f"{n} reads: shots or code")
        need(j["logits"] == "deployed" and j["gpu"] == FZ["gpu"], f"{n} reads: not the deployed engine on the H100")
        out[n] = z["c"]
        rec[n] = j
    return out, rec


def slots(att, item, net, rounds):
    """The accepted attempt of each round's slot (at most 3 attempts, the accepted one the last), else Missing."""
    acc = []
    for r in rounds:
        xs = sorted((x for x in att if x["item"] == item and x["network"] == net and x["round"] == r),
                    key=lambda x: x["attempt"])
        need(1 <= len(xs) <= 3 and [x["attempt"] for x in xs] == list(range(len(xs))), f"{item} {net} round {r}: attempts")
        ok = [x for x in xs if x["accepted"]]
        need(len(ok) == 1 and ok[0] is xs[-1], f"{item} {net} round {r}: not established")
        acc.append(ok[0])
    return acc


def gpu(R, man_t):
    """a of both networks, c of B1, and the mask sent to the Linux workstation, from the counted GPU session."""
    P = os.path.join(R, "pod")
    q, why, reps = {}, {}, {}
    pre = load(os.path.join(P, "pre", "prescreen.json"))
    need(pre.get("passed") is True, "the pod's pre-screen did not pass")
    core = int(pre["core"])
    att = load(os.path.join(P, "attempts.json"))
    mk = {}
    for n in NETS:
        z = np.load(os.path.join(P, "masks", f"val_{n}.npz"))
        c = z["c"]
        need(str(z["network"]) == n and float(z["tau"]) == FZ["networks"][n]["tau"], f"mask {n}: network or tau")
        need(str(z["shots_sha256"]) == man_t["shots_sha256"] and len(c) == man_t["shots"], f"mask {n}: not the timing shots")
        need(np.array_equal(z["routed"].astype(bool), np.abs(c.astype(np.float32).astype(np.float64)) < float(z["tau"])),
             f"mask {n}: routed is not |c| < tau")
        mk[n] = dict(routed=int(z["routed"].astype(bool).sum()), sha256=sha256_file(os.path.join(P, "masks", f"val_{n}.npz")))

    def check(x, handoff):
        r = load(os.path.join(P, x["record"]))
        w = load(os.path.join(P, x["watch"]))
        need(x["exit"] == 0 and w.get("exit") == 0 and w.get("rule") == "throttle" and w.get("clocks", {}).get("ok") is True
             and w.get("gpu_alone") is True and int(w.get("core", core)) == core, f"{x['record']}: the watch record")
        need(r["accepted"] is True and all(r["acceptance"].values()) and r["network"] == x["network"]
             and r["network_sha256"] == FZ["networks"][x["network"]]["sha256"] and r["code_digest"] == FZ["code_digest"]
             and r["gpu"] == FZ["gpu"] and r["core"] == core and r.get("limit") is None and r["handoff"] is handoff,
             f"{x['record']}: the record")
        return r

    for n in NETS:
        key = f"a.{n}"
        try:
            recs = [check(x, False) for x in slots(att, "G1", n, range(1, 6))]
            need(all(r["shots"] == 200000 and r["outputs_equal_deployed"] is True for r in recs), f"{key}: shots or outputs")
            reps[key] = [r["per_shot_mean"] for r in recs]
            q[key] = max(reps[key])
        except (Missing, KeyError, OSError) as e:
            q[key], why[key] = None, str(e)
    try:
        r = check(slots(att, "G2", B1, [1])[0], True)
        need(r["tau"] == TAU and r["routed"] == mk[B1]["routed"] and r["handoff_us"] == max(1.1, r["handoff_estimate_us"]),
             "c: the handoff record's tau, routed shots or rule")
        q["c"] = r["handoff_us"]
        reps["c"] = dict(estimate_us=r["handoff_estimate_us"], routed=r["routed"], routed_fraction=r["routed_fraction"])
    except (Missing, KeyError, OSError) as e:
        q["c"], why["c"] = None, str(e)
    h = os.path.join(P, "handoff", "surface", "low", f"val_{B1}.npz")
    hand = sha256_file(h) if os.path.exists(h) else None
    need(hand is None or hand == mk[B1]["sha256"], "the hand-off mask is not the session's mask")
    return q, why, reps, mk, hand, pre


def cpu(R, man_t, mk, hand):
    """This session's Grace fit (speed_rt.analysis_s, unchanged) and b of B1."""
    C = os.path.join(R, "cpu")
    freeze = json.load(open(os.path.join(ROOT, "speed_rt", "freeze_surf.json")))
    man_r = json.load(open(os.path.join(ROOT, "speed_rt", "testsets", "srt-ref.json")))
    circ = json.load(open(os.path.join(ROOT, "speed_rt", "circuits", "ref_circuits.json")))["circuits"]
    ok, why_c = AS.qualifies_cpu(C, False)
    if not ok:
        return dict(complete=False, missing={"cpu_session": why_c}, fit=None), None, None, why_c
    f2 = AS.fit_s2(C, freeze, man_r, circ, False)
    try:
        get, meta, code = AS.cpu_reader(C, freeze, False)
        need(meta["cpu"] == FZ["cpu"], f"the CPU session's CPU is {meta['cpu']}")
        recs = get("C-term", B1)
        need(hand is not None, "no hand-off mask")
        need(all(r["what"] == "term" and r["network"] == B1 and r["tau"] == TAU and r["mask_sha256"] == hand
                 and r["shots_sha256"] == man_t["shots_sha256"] and r["shots"] == man_t["shots"]
                 and r["routed"] == mk[B1]["routed"] and r["code_digest"] == code == FZ["code_digest"] for r in recs),
             "b: a term record's network, mask, shots, routed count or code")
        return f2, max(r["term_us"] for r in recs), [r["term_us"] for r in recs], None
    except (Missing, AS.Missing, KeyError, OSError) as e:
        return f2, None, None, str(e)


def r6(x):
    return None if x is None else round(float(x), 6)


def cmp(name, fa, fb, verdict):
    s = stats.paired(fa, fb, ALPHA)
    s.update(name=name, role="claim" if verdict else "descriptive (95 % interval beside; no verdict)")
    if not verdict:
        s.pop("verdict")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True, help="the freeze commit F (40 hex), which the beacon record must name")
    a = ap.parse_args()
    R = a.records
    out = dict(what="D9-SHALLOW registered test of d = 9 B1 (d9_shallow/analysis_d9s.py)", freeze=FZ["id"], missing={})
    b, mans, obs = shots(R, a.commit)                               # no result at all without the shots' provenance
    out["beacon"] = dict(round=b["round"], randomness=b["randomness"], freeze_commit=b["freeze_commit"], push_time=b["push_time"])
    out["shots"] = {s: dict(shots=m["shots"], shots_sha256=m["shots_sha256"], list_sha256=m["list_sha256"]) for s, m in mans.items()}
    # accuracy on the 1,000,000 test shots
    acc = None
    try:
        rd, rrec = reads(R, mans["test"], obs)
        c1 = rd[B1]
        routed = np.abs(c1.astype(np.float64)) < TAU
        pair = np.where(routed, rd["cpm"], (c1 > 0).astype(np.uint8))
        F = dict(pair=pair != obs, b1=(c1 > 0).astype(np.uint8) != obs, b3=(rd[B3] > 0).astype(np.uint8) != obs,
                 pm=rd["pm"] != obs, cpm=rd["cpm"] != obs)
        acc = dict(failures={k: int(v.sum()) for k, v in F.items()}, routed_test=int(routed.sum()),
                   routed_share_test=float(routed.mean()),
                   claim={"pair vs plain PM": cmp("pair vs plain PM", F["pair"], F["pm"], True),
                          "pair vs correlated PM": cmp("pair vs correlated PM", F["pair"], F["cpm"], True)},
                   descriptive={k: cmp(k, F[x], F[y], False) for k, (x, y) in {
                       "pair vs d9_b3_sv2s1": ("pair", "b3"), "B1 alone vs plain PM": ("b1", "pm"),
                       "B1 alone vs correlated PM": ("b1", "cpm"), "B1 alone vs d9_b3_sv2s1": ("b1", "b3"),
                       "d9_b3_sv2s1 vs plain PM": ("b3", "pm"), "d9_b3_sv2s1 vs correlated PM": ("b3", "cpm")}.items()},
                   reads={k: {kk: v[kk] for kk in sorted(v) if kk not in ("start", "secs", "host", "node")} for k, v in rrec.items()})
    except (Missing, KeyError, OSError) as e:
        out["missing"]["accuracy"] = str(e)
    out["accuracy"] = acc
    # timing: a, c from the pod; b from the Linux workstation, converted by this session's fit
    T = dict(a=None, b_raw=None, b_grace=None, c=None)
    try:
        q, why, reps, mk, hand, pre = gpu(R, mans["timing"])
        T.update(a=q[f"a.{B1}"], c=q["c"], a_reps=reps.get(f"a.{B1}"), a_b3=q[f"a.{B3}"], a_b3_reps=reps.get(f"a.{B3}"),
                 c_detail=reps.get("c"), routed_timing=mk[B1]["routed"], mask_sha256=mk[B1]["sha256"], handoff_sha256=hand,
                 prescreen=dict(passed=pre["passed"], core=pre["core"], card=pre.get("card")))
        out["missing"].update(why)
        f2, braw, breps, whyb = cpu(R, mans["timing"], mk, hand)
        if whyb:
            out["missing"]["b"] = whyb
        fit = f2.get("fit")
        if not f2.get("complete"):
            out["missing"]["fit"] = f"the Grace fit is not established: {f2.get('missing')}"
        T.update(b_raw=braw, b_raw_reps=breps, fit_complete=bool(f2.get("complete")),
                 factor_corr=fit["applied"]["corr"] if fit else None, fit_common=fit["common"] if fit else None,
                 b_grace=braw * fit["applied"]["corr"] if (fit and braw is not None) else None)
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "fit_d9s.json"), "x") as fh:
            json.dump(f2, fh, indent=1, sort_keys=True, allow_nan=False)
            fh.write("\n")
    except (Missing, KeyError, OSError) as e:
        out["missing"]["timing"] = str(e)
    est = all(T[k] is not None for k in ("a", "b_grace", "c"))
    T["composite_grace"] = T["a"] + T["b_grace"] + T["c"] if est else None
    T["composite_raw"] = T["a"] + T["b_raw"] + T["c"] if all(T[k] is not None for k in ("a", "b_raw", "c")) else None
    T["established"] = est
    tp, tc = FZ["targets"]["plain_pm_us"], FZ["targets"]["corr_pm_us"]
    T["composite_b3_descriptive"] = T["a_b3"] + 1.1 if T.get("a_b3") is not None else None
    out["timing"] = T
    # the claim (intersection-union): (1), (2), (3)
    v2 = acc["claim"]["pair vs plain PM"]["verdict"] == "better" if acc else None
    v3a = acc["claim"]["pair vs correlated PM"]["verdict"] == "better" if acc else None
    v1 = T["composite_grace"] < tp if est else None
    v3t = T["composite_grace"] < tc if est else None
    crit = {"(1) composite < plain PM's": v1, "(2) fewer failures than plain PM": v2,
            "(3) fewer failures than correlated PM": v3a, "(3) composite < correlated PM's": v3t}
    if all(v is True for v in crit.values()):
        verdict = "holds"
    elif any(v is False for v in crit.values()):
        verdict = "does not hold"
    else:
        verdict = "not established"
    out["claim"] = dict(criteria=crit, verdict=verdict, alpha=ALPHA, targets_us=dict(plain_pm=tp, corr_pm=tc),
                        paper_row="B1's pair (d9_b1w64_sh, q = 0.06)" if verdict == "holds" else "the 3-block row (d9_b3_sv2s1)",
                        rule=FZ["publication_rule"])
    out["inputs"] = {os.path.relpath(os.path.join(dp, f), R): sha256_file(os.path.join(dp, f))
                     for dp, _, fs in sorted(os.walk(R)) for f in sorted(fs)}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "analysis.json"), "x") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, allow_nan=False)
        fh.write("\n")
    with open(os.path.join(a.out, "table.md"), "x") as fh:
        fh.write(table(out))
    print(f"claim: {verdict}; criteria {crit}; missing {sorted(out['missing'])}", flush=True)


def table(o):
    T, A, C = o["timing"], o["accuracy"], o["claim"]
    f = lambda x, n=3: "not established" if x is None else f"{x:.{n}f}"  # noqa: E731
    L = ["# D9-SHALLOW registered test: results (d9_shallow/analysis_d9s.py)", "",
         f"Claim (B1's pair at q = 0.06 against plain PM and correlated PM, alpha = {o['claim']['alpha']}): **{C['verdict']}**. "
         f"Paper's d = 9 row: {C['paper_row']}.", "", "| criterion | result |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in C["criteria"].items()]
    L += ["", "| timing term | us per shot |", "|---|---|",
          f"| a (H100, slowest of 5) | {f(T.get('a'))} |", f"| b raw (Ryzen 9 9950X, slowest of 3) | {f(T.get('b_raw'))} |",
          f"| applied factor (corr) | {f(T.get('factor_corr'), 4)} |", f"| b on the Grace basis | {f(T.get('b_grace'))} |",
          f"| c (max(1.1, router estimate)) | {f(T.get('c'))} |", f"| composite, Grace basis (decides) | {f(T.get('composite_grace'))} |",
          f"| composite, raw basis (beside) | {f(T.get('composite_raw'))} |",
          f"| targets: plain PM / correlated PM (Grace) | {C['targets_us']['plain_pm']} / {C['targets_us']['corr_pm']} |",
          f"| d9_b3_sv2s1: a, a + 1.1 (descriptive) | {f(T.get('a_b3'))}, {f(T.get('composite_b3_descriptive'))} |"]
    if A:
        L += ["", "| decoder | failures on 1,000,000 test shots |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in A["failures"].items()]
        L += ["", "| comparison | role | only a / only b | exact McNemar p | diff | 97.5 % interval | 95 % interval |",
              "|---|---|---|---|---|---|---|"]
        for d in (A["claim"], A["descriptive"]):
            for k, s in d.items():
                L.append(f"| {k} | {s['role'].split(' ')[0]}{' (' + s['verdict'] + ')' if 'verdict' in s else ''} | "
                         f"{s['only_a']} / {s['only_b']} | {s['mcnemar_p']:.3g} | {s['diff_ler']:+.6f} | "
                         f"({s['diff_lo']:+.6f}, {s['diff_hi']:+.6f}) | ({s['diff_lo95']:+.6f}, {s['diff_hi95']:+.6f}) |")
    L += ["", "Missing: " + ("; ".join(f"{k}: {v}" for k, v in sorted(o["missing"].items())) or "nothing"), ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
