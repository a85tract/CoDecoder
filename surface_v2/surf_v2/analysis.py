"""SURF-v2's one-time analysis (surf_v2/REGISTRATION.md; the research plan's section 4.7): frozen with the freeze commit,
run once after the last read as one CPU job, and run a second time on the same inputs only to show byte-identical
outputs.

Inputs: the freeze record (surf_v2/freeze.json: every network's sha256, q and tau, the read path's code digest, the
config's sha256, any training-failure exception), surf_v2/config.json, the read directories (convlines.evaluate --save:
the 12 networks on the calibration and the test set, every baseline on the test set; BP-OSD on its subset, possibly in
shares), the test and calibration sets' chunk files (each checked against testsets/<set>.json), the timing session
directories (E1; none, one or two: a path that does not exist is no session; ordered by their start checks' times and
labelled session 1, 2, ...) and the CPU term's directory (Alpine). It refuses any read whose record or arrays do not
match it, whose network sha256, tau, set manifest or code digest is not the freeze's, that was not made by the deployed
engine at batch one, whose observables differ from its set's, or whose saved decisions differ from its c, tau and
correlated PM's answers; any calibration read but the one freeze.json names; any timing record not made on every
validation shot on the registered GPU (config timing gpu_model); and CPU-term records from more than one job, of another
read path, or of another mask than the deployed engine's at the frozen tau. Without a timing session, with more than
two session directories or two sessions on one GPU (then no session counts), or when the CPU term cannot be matched to
the counted session (a network without its three repetitions, or routing at d = 9-17 another number of validation shots
than the session's handoff), families A and C are reported in full and every comparison of family B is "not
established", with the reason.

Outputs (OUT must not exist; nothing in them depends on the time, the host or OUT): analysis.json (every number, verdict
and input digest), tables.md, family_A.csv, family_B.csv, family_C.csv, composite.csv, pair_curves.csv, frontiers.csv,
paper_rows.csv, pair_curves.png / .pdf, frontiers.png / .pdf.

    python -m surf_v2.analysis --freeze surf_v2/freeze.json --reads READS... --shots TEST --cal-shots CAL
        --sessions [S1 [S2]] --cpu CPU --out OUT
--dry: dry runs only (the dry set and freeze; stand-in reads allowed and labelled; never a result)."""
import argparse
import csv
import datetime
import glob
import hashlib
import io
import json
import os

import numpy as np

from convlines import beacon, stats, testset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONFIG = os.path.join(HERE, "config.json")
NV = ("plain_pm", "corr_pm", "ising_m1_pm", "ising_m4_pm", "ising_m1_corr", "ising_m4_corr", "bposd_minllr", "bposd_bp60")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def sha_arr(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def rel(p, base):
    return os.path.relpath(os.path.abspath(p), os.path.abspath(base))


# ---- reads ------------------------------------------------------------------------------------------------------------
def scan(dirs):
    """{(set, d, decoder): [(record, npz path)]} of every read record under dirs."""
    out = {}
    for d in dirs:
        for j in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
            r = json.load(open(j))
            if r.get("what", "").startswith("SURF-v2 read"):
                out.setdefault((r["set"], r["d"], r["decoder"]), []).append((r, j[:-5] + ".npz"))
    return out


def merged(reads, key, chunks, digests):
    """One decoder's arrays over exactly `chunks` (shares joined in chunk order), every file checked against its record
    and every record naming the set's manifest (testsets/<set>.json of this tree). Shares of other chunks of the same set
    are left aside (the dry set is calibration and test in one)."""
    assert key in reads, f"no read of {key}"
    man = sha_file(os.path.join(ROOT, testset.SETS[key[0]]))
    parts = sorted((x for x in reads[key] if set(testset.parse_range(x[0]["chunks"])) <= set(chunks)),
                   key=lambda x: testset.parse_range(x[0]["chunks"])[0])
    got = [j for r, _ in parts for j in testset.parse_range(r["chunks"])]
    assert got == chunks, f"{key}: the reads cover chunks {got[:3]}...{got[-3:]} ({len(got)}), not the set's {len(chunks)}"
    arrs, recs = {}, []
    for r, f in parts:
        assert r["manifest_sha256"] == man, f"{f}: read on another manifest of {key[0]}"
        assert sha_file(f) == r["npz_sha256"], f"{f}: not the file its record names"
        z = np.load(f)
        for k, v in r["arrays"].items():
            assert sha_arr(z[k]) == v["sha256"], f"{f}: array {k} is not its record's"
            arrs.setdefault(k, []).append(z[k])
        recs.append(r)
        digests[os.path.basename(f)] = r["npz_sha256"]
        digests[os.path.basename(f)[:-4] + ".json"] = sha_file(f[:-4] + ".json")
    return {k: np.concatenate(v) for k, v in arrs.items()}, recs


def check_beacon(tset, din):
    """The test set's seeds come from the registered beacon: the record passes convlines.beacon.load, it is the record
    the test manifest names, and every chunk's seed is re-derived from its randomness (testset.seeds)."""
    bp = os.path.join(HERE, "beacon.json")
    b = beacon.load(bp)
    assert testset.seeds(tset, bp), f"testsets/{tset}.json: seeds not derived from {bp}"
    din["surf_v2/beacon.json"] = sha_file(bp)
    return dict(freeze_commit=b["freeze_commit"], round=b["round"], push_time=b["push_time"],
                push_activity_id=b["push_activity"]["id"], push_source=b.get("push_source"), randomness=b["randomness"])


def set_obs(set_, d, chunks, shot_dir):
    """The observables of a set's chunks, each chunk file checked against the set's manifest."""
    man = testset.manifest(set_)
    out = []
    for j in chunks:
        f = testset.name(man["split"], d, j)
        raw = open(os.path.join(shot_dir, f), "rb").read()
        assert testset.sha256_bytes(raw) == man["files"][f]["sha256"], f"{f}: not the manifest's chunk"
        out.append(np.load(io.BytesIO(raw))["obs"].astype(np.uint8))
    return np.concatenate(out)


def check_net(recs, fz, name, tau, dry, cal=False):
    """A network read's records against the freeze: weights, tau, the deployed engine at batch one, the code digest."""
    f = fz["networks"][name]
    notes = []
    for r in recs:
        assert r["kind"] == "network" and r["network"] == name, f"{name}: {r['npz']} is another decoder's"
        assert r["network_sha256"] == f["sha256"], f"{r['npz']}: weights {r['network_sha256'][:12]} are not the frozen ones"
        assert (r["tau"] is None) if cal else (r["tau"] == tau), f"{r['npz']}: tau {r['tau']} is not {None if cal else tau}"
        if r.get("standin") or r["routing"] != "deployed":
            assert dry, f"{r['npz']}: not a read of the deployed engine at batch one"
            notes.append("stand-in" if r.get("standin") else f"routing {r['routing']}")
        else:
            assert r["batch_one"], f"{r['npz']}: not at batch one"
            assert r["code_digest"] == fz["read_code_digest"] or dry, f"{r['npz']}: another read path's code"
    return sorted(set(notes))


def check_opp(recs, fz, dry):
    notes = []
    for r in recs:
        assert r["kind"] == "opponent"
        if r.get("standin"):
            assert dry, f"{r['npz']}: a stand-in read"
            notes.append("stand-in")
        else:
            assert r["code_digest"] == fz["read_code_digest"] or dry, f"{r['npz']}: another read path's code"
    return sorted(set(notes))


# ---- statistics -------------------------------------------------------------------------------------------------------
def holm(ps, alpha):
    """Holm's step-down adjusted p-values and rejections (ties in the stable sort order)."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [None] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj, [a <= alpha for a in adj]


def tau_q(c, q):
    return float(np.quantile(np.abs(c), q)) if q > 0 else 0.0


def pair(c, corr, tau):
    r = np.abs(c) < tau
    return np.where(r, corr, (c > 0).astype(np.uint8)).astype(np.uint8), r


# ---- timing -----------------------------------------------------------------------------------------------------------
def session(sdir, cfg, fz, names, dry):
    """A timing session's verdict and every network's GPU term and handoff (README section 4; config timing), with its
    start check's time and GPU. Without attempts.json the session is lost (or ended at its start check)."""
    t = cfg["timing"]
    out = dict(source=os.path.basename(os.path.dirname(os.path.normpath(sdir))), networks={}, problems=[])
    chk = {}
    for w in ("start", "end"):
        f = os.path.join(sdir, f"gpucheck_{w}.json")
        rec = json.load(open(f)) if os.path.exists(f) else None
        chk[w] = bool(rec and rec["passed"])
        if not chk[w]:
            out["problems"].append(f"GPU check at the {w} missing or failed")
        if w == "start":
            out.update(start=rec.get("time") if rec else None,
                       gpu_uuid=rec.get("clocks", {}).get("gpu_uuid") if rec else None)
    out["gpu_check"] = chk
    if not os.path.exists(os.path.join(sdir, "attempts.json")):
        out["problems"].append("no attempts.json: the session is lost or ended at its start check")
        out.update(lost=True, qualified=False)
        return out
    att = json.load(open(os.path.join(sdir, "attempts.json")))
    for name in names:
        d = fz["networks"][name]["d"]
        need_h = t["handoff_processes_d7"] if d == 7 else t["handoff_processes_d9_17"]
        rows = att.get(name, [])
        gpu, hand, rej = [], [], 0
        for x in rows:
            f = os.path.join(sdir, x["file"]) if x.get("file") else None
            rec = json.load(open(f)) if f and os.path.exists(f) else None
            if rec and not dry:                                   # every process of the session as registered
                assert rec.get("limit") is None and rec.get("gpu") == t["gpu_model"], \
                    f"{f}: limit {rec.get('limit')} on {rec.get('gpu')}, not every validation shot on the registered GPU"
                assert x["kind"] == "handoff" or rec.get("shots") == t["val_shots"], f"{f}: {rec.get('shots')} shots"
            ok = bool(rec and rec.get("accepted") and (rec.get("network_sha256") == fz["networks"][name]["sha256"])
                      and (dry or rec.get("code_digest") == fz["read_code_digest"]))
            if ok and x["kind"] == "handoff" and d != 7:
                ok = rec.get("tau") == fz["networks"][name]["tau"]
            if not ok:
                rej += 1
                continue
            (hand if x["kind"] == "handoff" else gpu).append(rec)
        n = dict(rejected=rej, gpu_accepted=len(gpu), handoff_accepted=len(hand))
        gpu, hand = gpu[:t["gpu_processes"]], hand[:need_h]
        if gpu:
            n["a_us"] = max(r["per_shot_mean"] for r in gpu)
            n["a_reps_us"] = [r["per_shot_mean"] for r in gpu]
        if hand:
            n["c_us"] = max(r["handoff_us"] for r in hand)
            n["c_reps_us"] = [r["handoff_us"] for r in hand]
            n["c_estimate_us"] = [r.get("handoff_estimate_us") for r in hand]
            if d != 7 and hand[0].get("limit") is None:          # the router's routing on all validation shots
                n["handoff_routed"] = hand[0]["routed"]
        n["qualified"] = rej <= t["max_replacements"] and len(gpu) == t["gpu_processes"] and len(hand) == need_h
        if not n["qualified"]:
            out["problems"].append(f"{name}: {len(gpu)} GPU-term and {len(hand)} handoff processes accepted, {rej} rejected")
        out["networks"][name] = n
    out["qualified"] = all(chk.values()) and all(n["qualified"] for n in out["networks"].values())
    out["lost"] = False
    return out


def started(s):
    """A session's place in time: its start check's time (a session without one first, in the order given)."""
    return (1, datetime.datetime.strptime(s["start"], "%Y-%m-%dT%H:%M:%S%z").timestamp()) if s["start"] else (0, 0.0)


def sessions(paths, cfg, fz, names, dry, din):
    """The timing sessions: every directory of paths (a path that does not exist is no session), ordered by its start
    check, labelled session 1, 2, ..., its files' sha256 in din. -> (sessions, the counted one or None, why none counts
    or None). The first qualifying session counts, else the last; more than two directories, or two sessions on one GPU,
    leave none counted, and family B is then not established for that reason."""
    found = [s for s in paths if os.path.isdir(s)]               # a session never pulled leaves no directory
    sess = sorted(((s, session(s, cfg, fz, names, dry)) for s in found), key=lambda x: started(x[1]))
    for i, (sdir, x) in enumerate(sess, 1):
        x["label"] = f"session {i}"
        for f in sorted(glob.glob(os.path.join(sdir, "*.json"))):
            din[f"session_{i}/{os.path.basename(f)}"] = sha_file(f)
    sess = [x for _, x in sess]
    uu = [x["gpu_uuid"] for x in sess if x["gpu_uuid"]]
    two = sorted({u for u in uu if uu.count(u) > 1})                    # a second session runs on a new host
    bad = ([f"{len(sess)} timing session directories, more than two"] if len(sess) > 2 else []) + \
          ([f"two timing sessions on one GPU, {' and '.join(two)}"] if two else [])
    if bad:
        return sess, None, "; ".join(bad)
    count = next((s for s in sess if s["qualified"]), sess[-1] if sess else None)   # session 2 counts only if 1 failed
    if len(sess) == 2 and sess[0]["qualified"]:
        sess[1]["problems"].append("not counted: session 1 qualified (a second session follows a technical failure only)")
    return sess, count, None


def cpu(cdir, cfg, fz, names, dry, routed):
    """Every network's CPU term (the slowest of three repetitions) and the node's ratio per d (median of three refs), all
    from one job of the frozen read path, each term on the deployed engine's mask at the frozen tau. -> (terms, problems).
    A network without its three terms and refs, or whose term at d = 9-17 routed another number of validation shots than
    the counted session's handoff (routed: name -> that count, if known), gets no term but a problem: the CPU term is then
    not matched to the counted session, and family B is not established. (A session lost before its masks were pulled
    leaves no CPU term at all.)"""
    terms = {}
    for f in sorted(glob.glob(os.path.join(cdir, "term_*_rep*.json"))):
        r = json.load(open(f))
        terms.setdefault(r["network"], []).append(r)
    out, used, problems = {}, [], []
    for name in names:
        fn = fz["networks"][name]
        d = fn["d"]
        rr = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(cdir, f"ref_d{d}_rep*.json")))]
        refs = [r["per_block_us"] for r in rr]
        ts = terms.get(name, [])
        used += ts + rr
        val = json.load(open(os.path.join(ROOT, "routing", "manifest.json")))["val"][str(d)]["shots_sha256"]
        for r in ts:
            assert r.get("tau") == fn["tau"], f"{name}: its CPU term's mask is not at the frozen tau"
            assert r["shots_sha256"] == val or dry, f"{name}: its CPU term did not read the studies' validation shots"
            assert r.get("mask_kind") == "deployed" or dry, f"{name}: its CPU term's mask is not the deployed engine's"
        if len(ts) != cfg["timing"]["cpu_reps"] or len(refs) != cfg["timing"]["cpu_reps"]:
            problems.append(f"{name}: {len(ts)} term and {len(refs)} ref repetitions")
            continue
        if routed.get(name) is not None and any(r["routed"] != routed[name] for r in ts):
            problems.append(f"{name}: its CPU term routed {sorted({r['routed'] for r in ts})} validation shots, the counted "
                            f"session's handoff {routed[name]}")
            continue
        g = float(next(x for x in csv.DictReader(open(os.path.join(ROOT, "timing", "grace_basis.csv")))
                       if x["decoder"] == "correlated PM" and int(x["d"]) == d)["grace_us_per_shot"])
        med = float(np.median(refs))
        b_node = max(r["term_us"] for r in ts)
        out[name] = dict(term_reps_us=[r["term_us"] for r in ts], b_node_us=b_node, ref_reps_us=refs, ref_median_us=med,
                         G_us=g, ratio=g / med, b_grace_us=b_node * g / med, routed_val=[r["routed"] for r in ts],
                         b_nvidia_us=b_node * cfg["nvidia_basis_d17"]["G17_us"] / med if d == 17 else None)
    jobs = sorted({str(r.get("slurm_job")) for r in used})
    assert len(jobs) <= 1 and "None" not in jobs, f"the CPU terms and refs come from jobs {jobs}, not from one"
    assert dry or all(r.get("code_digest") == fz["read_code_digest"] for r in used), "a CPU term of another read path"
    return out, problems


# ---- figures ----------------------------------------------------------------------------------------------------------
def figures(out, curves, fronts, cfg, dry):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.hashsalt": "surf-v2", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False})
    col = {"new": "#0072B2", "deployed": "#D55E00"}
    tag = " (DRY RUN, stand-ins)" if dry else ""
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.4), constrained_layout=True)
    for i, d in enumerate(cfg["distances"]):
        a = ax.flat[i]
        for role, rows in curves[str(d)].items():
            q = [100 * x["q"] for x in rows]
            a.plot(q, [x["failures_per_million"] for x in rows], marker="o", ms=2.5, lw=1.2, color=col[role],
                   label=f"{role} ({rows[0]['network']})")
            ch = [x for x in rows if x["chosen"]]
            if ch:
                a.plot([100 * ch[0]["q"]], [ch[0]["failures_per_million"]], marker="s", ms=5, mfc="none", color=col[role])
        a.set_title(f"d = {d}")
        a.set_xlabel("routed fraction q (%)")
        a.set_ylabel("pair failures per million")
        a.legend(fontsize=5.5, frameon=False)
    fig.suptitle("Pair failures on the test set, tau_q fitted on the calibration set; squares: the q in use" + tag,
                 fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"pair_curves.{ext}"), dpi=200, metadata={"CreationDate": None, "ModDate": None}
                    if ext == "pdf" else {"Software": None})
    plt.close(fig)
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.4), constrained_layout=True)
    for i, d in enumerate(cfg["distances"]):
        a = ax.flat[i]
        f = fronts.get(str(d))
        if f:
            pts = [(x["projected_us"], x["failures"], x["q"]) for x in f["table"] if x["projected_us"] is not None]
            a.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=2.5, lw=1.2, color=col["new"])
            for p in pts:
                a.annotate(f"{100 * p[2]:g}", (p[0], p[1]), fontsize=5, xytext=(2, 2), textcoords="offset points")
            a.axvline(f["bar_us"], color="k", lw=0.8, ls=":")
            a.axhline(f["limit"], color="k", lw=0.8, ls="--")
            if f["chosen"] is not None:
                a.plot([f["chosen"]["projected_us"]], [f["chosen"]["failures"]], marker="s", ms=6, mfc="none", color="k")
        a.set_title(f"d = {d}" + (f" ({f['network']})" if f else " (no new network)"), fontsize=7)
        a.set_xlabel("projected composite (us)")
        a.set_ylabel("calibration failures in 200,000" if not dry else "calibration failures")
    fig.suptitle("Calibration frontiers of the new networks: dotted, the speed bar; dashed, the limit M - 2 sqrt(2M); "
                 "labels, q in %" + tag, fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"frontiers.{ext}"), dpi=200, metadata={"CreationDate": None, "ModDate": None}
                    if ext == "pdf" else {"Software": None})
    plt.close(fig)


def write_csv(path, rows):
    keys = list(rows[0])
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def fmt(x, nd=3):
    return "" if x is None else (f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}" if isinstance(x, int) else str(x))


# ---- the analysis -----------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", default=os.path.join(HERE, "freeze.json"))
    ap.add_argument("--reads", nargs="+", required=True)
    ap.add_argument("--shots", required=True, help="the test set's chunk files (dry: the dry set's)")
    ap.add_argument("--cal-shots", default="", help="the calibration set's chunk files (default: --shots, as dry runs)")
    ap.add_argument("--sessions", nargs="*", required=True,
                    help="the timing session directories ($E/pod*/session), ordered by their start checks; a path "
                         "that does not exist (the glob of no session) is no session; more than 2, or two on one GPU, "
                         "count none")
    ap.add_argument("--cpu", required=True, help="the CPU term's directory")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(CONFIG))
    fz = json.load(open(a.freeze))
    assert fz["config_sha256"] == sha_file(CONFIG), "surf_v2/config.json is not the frozen one"
    assert bool(fz.get("dry")) == a.dry, "a dry freeze needs --dry, and only a dry freeze takes it"
    D = cfg["distances"]
    tset = cfg["dry"]["set"] if a.dry else cfg["test"]["set"]
    cset = cfg["dry"]["set"] if a.dry else cfg["calibration"]["set"]
    exc = {int(k): v for k, v in fz.get("exceptions", {}).items()}
    reads = scan(a.reads)
    din = {"freeze.json": sha_file(a.freeze), "config.json": sha_file(CONFIG),
           f"testsets/{tset}.json": sha_file(os.path.join(ROOT, testset.SETS[tset])),
           f"testsets/{cset}.json": sha_file(os.path.join(ROOT, testset.SETS[cset]))}
    bk = None if a.dry else check_beacon(tset, din)
    res = dict(what="SURF-v2's one-time analysis (surf_v2/analysis.py)" + (" -- DRY RUN, never a result" if a.dry else ""),
               dry=a.dry, freeze_commit=bk["freeze_commit"] if bk else None, beacon=bk, test_set=tset, distances={},
               inputs=din)
    roles_all, C_tests, rowsA, curves, fronts, notes = {}, [], [], {}, {}, {}
    for d in D:
        k = str(d)
        nets = cfg["networks"][k]
        roles = {r: nets[r] for r in ("new", "deployed") if not (r == "new" and d in exc)}
        roles_all[k] = roles
        ch = testset.parse_range(cfg["dry"]["test_chunks"] if a.dry else cfg["test"]["chunks"][k])
        bch = testset.parse_range(cfg["dry"]["bposd_chunks"] if a.dry else cfg["test"]["bposd_chunks"][k])
        cch = testset.parse_range(cfg["dry"]["cal_chunks"] if a.dry else cfg["calibration"]["chunks"])
        assert bch == ch[:len(bch)], "the BP-OSD subset is the first chunks of the test"
        obs = set_obs(tset, d, ch, a.shots)
        cobs = set_obs(cset, d, cch, a.cal_shots or a.shots)
        nb = len(bch) * testset.CHUNK
        opp, dec = {}, {}
        for key in NV:
            x, recs = merged(reads, (tset, d, key), bch if key.startswith("bposd") else ch, din)
            assert np.array_equal(x["obs"], obs[:len(x["obs"])]), f"d{d} {key}: observables differ from the test set's"
            dec[key] = x["dec"]
            notes[f"d{d} {key}"] = check_opp(recs, fz, a.dry)
        corr = dec["corr_pm"]
        net = {}
        for role, name in roles.items():
            fn = fz["networks"][name]
            x, recs = merged(reads, (tset, d, name), ch, din)
            assert np.array_equal(x["obs"], obs), f"d{d} {name}: observables differ from the test set's"
            notes[f"d{d} {name}"] = check_net(recs, fz, name, fn["tau"], a.dry)
            c = x["c"].astype(np.float32)
            p, r = pair(c, corr, fn["tau"])
            assert np.array_equal(r, x["routed"].astype(bool)) and np.array_equal(p, x["dec"]), \
                f"d{d} {name}: the saved routing or decisions are not |c| < tau with correlated PM's answers"
            xc, rc = merged(reads, (cset, d, name), cch, din)
            check_net(rc, fz, name, None, a.dry, cal=True)
            assert [r["npz_sha256"] for r in rc] == [fn["cal_read_sha256"]], \
                f"d{d} {name}: not the calibration read the freeze names"
            assert np.array_equal(xc["obs"], cobs), f"d{d} {name}: calibration observables differ from the set's"
            net[role] = dict(name=name, c=c, pair=p, routed=r, alone=(c > 0).astype(np.uint8), cal_c=xc["c"])
        row = net["new"] if "new" in net else net["deployed"]
        fp = row["pair"] != obs
        A = {}
        for key in NV:
            n = nb if key.startswith("bposd") else len(obs)
            s = stats.paired(fp[:n], dec[key] != obs[:n], cfg["families"]["A"]["alpha"])
            A[key] = dict(s, verdict="better" if s["verdict"] == "better" else "not shown", baseline=cfg["baselines"][key])
            rowsA.append(dict(d=d, network=row["name"], baseline=key, shots=n, pair_failures=s["fail_a"],
                              baseline_failures=s["fail_b"], only_pair=s["only_a"], only_baseline=s["only_b"],
                              mcnemar_p=s["mcnemar_p"], diff_ler=s["diff_ler"], lo=s["diff_lo"], hi=s["diff_hi"],
                              verdict=A[key]["verdict"]))
        if "new" in net:
            for test, kk in (("network alone (q = 0)", "alone"), ("pair at its own q", "pair")):
                s = stats.paired(net["new"][kk] != obs, net["deployed"][kk] != obs, 0.05)
                C_tests.append(dict(d=d, test=test, new=net["new"]["name"], deployed=net["deployed"]["name"],
                                    new_failures=s["fail_a"], deployed_failures=s["fail_b"], only_new=s["only_a"],
                                    only_deployed=s["only_b"], mcnemar_p=s["mcnemar_p"], diff_ler=s["diff_ler"],
                                    lo95=s["diff_lo"], hi95=s["diff_hi"]))
        curves[k] = {}
        for role, x in net.items():
            fn = fz["networks"][x["name"]]
            rows = []
            for q in cfg["qrule"]["grid"][k]:
                t = tau_q(x["cal_c"], q)
                p, r = pair(x["c"], corr, t)
                rows.append(dict(network=x["name"], q=q, tau_q=t, routed_test=int(r.sum()), failures=int((p != obs).sum()),
                                 failures_per_million=1e6 * float((p != obs).mean()),
                                 chosen=bool(fn.get("q") is not None and abs(q - fn["q"]) < 1e-12 and role != "deployed")))
            curves[k][role] = rows
        if "new" in net and fz["networks"][net["new"]["name"]].get("policy_file"):
            pf = os.path.join(ROOT, fz["networks"][net["new"]["name"]]["policy_file"])
            assert sha_file(pf) == fz["networks"][net["new"]["name"]]["policy_sha256"], f"{pf} is not the frozen policy"
            fronts[k] = dict(json.load(open(pf)), network=net["new"]["name"])
            din[os.path.relpath(pf, ROOT)] = sha_file(pf)
        res["distances"][k] = dict(
            row_network=row["name"], exception=exc.get(d), shots=len(obs), bposd_shots=nb,
            failures={**{x["name"]: dict(pair=int((x["pair"] != obs).sum()), alone=int((x["alone"] != obs).sum()),
                                         routed_test=int(x["routed"].sum()), routed_share=float(x["routed"].mean()),
                                         q=fz["networks"][x["name"]].get("q"), tau=fz["networks"][x["name"]]["tau"])
                         for x in net.values()},
                      **{key: int((dec[key] != obs[:len(dec[key])]).sum()) for key in NV}},
            family_A=A, accuracy_holds=all(v["verdict"] == "better" for v in A.values()))
    # family C: Holm over every test of the six distances
    adj, rej = holm([t["mcnemar_p"] for t in C_tests], cfg["families"]["C"]["alpha_familywise"])
    for t, p, r in zip(C_tests, adj, rej):
        t.update(holm_p=p, verdict=("better" if t["only_new"] < t["only_deployed"] else "worse") if r else "inconclusive")
    res["family_C"] = dict(tests=C_tests, m=len(C_tests), left_out=sorted(exc),
                           note="Holm over the tests of the distances with a new network")
    # timing
    names = sorted({n for r in roles_all.values() for n in r.values()})
    sess, count, anomaly = sessions(a.sessions, cfg, fz, names, a.dry, din)
    qualified = bool(count and count["qualified"])
    counted = count["networks"] if count else {}
    cp, cpu_problems = cpu(a.cpu, cfg, fz, names, a.dry, {n: x.get("handoff_routed") for n, x in counted.items()})
    for f in sorted(glob.glob(os.path.join(a.cpu, "*.json"))):
        din["cpu/" + os.path.basename(f)] = sha_file(f)
    why = (anomaly if anomaly else "no timing session" if count is None else "timing session not qualified" if not
           qualified else "CPU term not matched to the counted session" if cpu_problems else None)  # B not established
    comp, rowsB, rowsComp = {}, [], []
    for n in names:
        t, c = counted.get(n, {}), cp.get(n, {})
        d = fz["networks"][n]["d"]
        tot = (t["a_us"] + c["b_grace_us"] + t["c_us"]) if ("a_us" in t and "c_us" in t and c) else None
        comp[n] = dict(d=d, a_us=t.get("a_us"), b_grace_us=c.get("b_grace_us"), c_us=t.get("c_us"), composite_us=tot,
                       b_node_us=c.get("b_node_us"), ratio=c.get("ratio"), qualified=qualified,
                       composite_nvidia_basis_us=(t["a_us"] + c["b_nvidia_us"] + t["c_us"]) if d == 17 and tot else None)
        reg = cfg["registered_composites_us"][str(d)] if n == cfg["networks"][str(d)]["deployed"] else None
        rowsComp.append(dict(d=d, network=n, role=next(r for r, x in roles_all[str(d)].items() if x == n), a_gpu_us=t.get("a_us"),
                             b_cpu_grace_us=c.get("b_grace_us"), c_handoff_us=t.get("c_us"), composite_us=tot,
                             registered_composite_us=reg["composite"] if reg else None,
                             composite_nvidia_basis_us=comp[n]["composite_nvidia_basis_us"]))
    for d in D:
        k = str(d)
        rn = res["distances"][k]["row_network"]
        fn = fz["networks"][rn]
        B = {}
        for key in NV:
            reg = cfg["registered_times_us"][k][key]
            us = comp[rn]["composite_us"]
            if fn.get("outcome") == "no admissible q":
                v = "fails (no admissible q)"
            elif why:
                v = f"not established ({why})"
            else:
                v = "holds" if us is not None and us < reg else "fails"
            B[key] = dict(baseline=cfg["baselines"][key], registered_us=reg, pair_us=us,
                          margin_us=(reg - us) if us is not None else None, verdict=v)
            rowsB.append(dict(d=d, network=rn, baseline=key, pair_us=us, registered_us=reg, margin_us=B[key]["margin_us"],
                              verdict=v))
        x = res["distances"][k]
        x.update(family_B=B, speed_holds=all(v["verdict"] == "holds" for v in B.values()))
        x["claim_holds"] = x["accuracy_holds"] and x["speed_holds"]
        if d == 17:
            nvb = cfg["nvidia_basis_d17"]["times_us"]
            us = comp[rn]["composite_nvidia_basis_us"]
            x["nvidia_basis_d17"] = {key: dict(baseline_us=nvb[key], pair_us=us, margin_us=(nvb[key] - us) if us else None,
                                               below=bool(us is not None and us < nvb[key])) for key in NV}
    res.update(timing=dict(sessions=sess, session_paths_not_found=len(a.sessions) - len(sess),
                           counting_session=count["label"] if count else None, qualified=qualified,
                           cpu_problems=cpu_problems, family_B_established=why is None,
                           family_B_not_established_because=why, composites=comp, cpu=cp))
    paper = []
    for d in D:
        k = str(d)
        x = res["distances"][k]
        rn = x["row_network"]
        f = x["failures"][rn]
        paper.append(dict(d=d, network=rn, label=("training-failure exception: the deployed network read on the fresh "
                          "test set" if d in exc else fz["networks"][rn].get("outcome", "")), q=f["q"], tau=f["tau"],
                          shots=x["shots"], pair_failures=f["pair"], network_alone_failures=f["alone"],
                          routed_share_test=f["routed_share"], composite_us=comp[rn]["composite_us"],
                          a_gpu_us=comp[rn]["a_us"], b_cpu_grace_us=comp[rn]["b_grace_us"], c_handoff_us=comp[rn]["c_us"],
                          **{f"{key}_failures": x["failures"][key] for key in NV},
                          accuracy_holds=x["accuracy_holds"], speed_holds=x["speed_holds"], claim_holds=x["claim_holds"]))
    res["paper_rows"] = dict(rule=cfg["paper_rows"]["rule"], rows=paper)
    res["pair_curves"] = curves
    res["frontiers"] = {k: dict(network=v["network"], table=v["table"], chosen=v["chosen"], limit=v["limit"],
                                bar_us=v["bar_us"]) for k, v in fronts.items()}
    res["read_notes"] = {k: v for k, v in sorted(notes.items()) if v}
    os.makedirs(a.out)
    with open(os.path.join(a.out, "analysis.json"), "x") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    write_csv(os.path.join(a.out, "family_A.csv"), rowsA)
    write_csv(os.path.join(a.out, "family_B.csv"), rowsB)
    write_csv(os.path.join(a.out, "family_C.csv"), C_tests or [dict(note="no family C test")])
    write_csv(os.path.join(a.out, "composite.csv"), rowsComp)
    write_csv(os.path.join(a.out, "pair_curves.csv"), [dict(d=int(k), role=r, **x) for k, v in curves.items()
                                                       for r, rows in v.items() for x in rows])
    write_csv(os.path.join(a.out, "frontiers.csv"), [dict(d=int(k), network=v["network"], **{kk: x.get(kk) for kk in (
        "q", "tau", "routed", "failures", "projected_us", "admissible", "cost_us")}, chosen=bool(v["chosen"] and
        v["chosen"]["q"] == x["q"])) for k, v in fronts.items() for x in v["table"]] or [dict(note="no frontier")])
    write_csv(os.path.join(a.out, "paper_rows.csv"), paper)
    figures(a.out, curves, res["frontiers"], cfg, a.dry)
    state = ("qualified" if why is None else "NOT qualified: family B not established" if count and not qualified else
             f"qualified; family B not established: {why}" if count else f"family B not established: {why}")
    md = [f"# SURF-v2 analysis{' (DRY RUN: stand-ins, never a result)' if a.dry else ''}", "",
          f"Freeze commit {res['freeze_commit']}; test set {tset}; timing session counted: "
          f"{count['label'] + ' of ' + str(len(sess)) if count else 'none'} ({state}).", "",
          "## The surface table (failures on the test set; BP-OSD on its subset)", "",
          "| d | row network | q | pair | network alone | " + " | ".join(cfg["baselines"][k] for k in NV) + " |",
          "|---" * (5 + len(NV)) + "|"]
    for p in paper:
        md.append(f"| {p['d']} | {p['network']} | {fmt(p['q'], 2)} | {fmt(p['pair_failures'])} | "
                  f"{fmt(p['network_alone_failures'])} | " + " | ".join(fmt(p[f'{k}_failures']) for k in NV) + " |")
    md += ["", "## Families A and B (per d: 8 accuracy and 8 speed comparisons; the claim needs all 16)", "",
           "| d | baseline | pair vs baseline failures | p | 97.5 % interval of the LER difference | A | pair us | "
           "registered us | B |", "|---|---|---|---|---|---|---|---|---|"]
    for d in D:
        x = res["distances"][str(d)]
        for key in NV:
            s, b = x["family_A"][key], x["family_B"][key]
            md.append(f"| {d} | {cfg['baselines'][key]} | {s['fail_a']:,} vs {s['fail_b']:,} | {s['mcnemar_p']:.3g} | "
                      f"[{s['diff_lo']:.3g}, {s['diff_hi']:.3g}] | {s['verdict']} | {fmt(b['pair_us'], 2)} | "
                      f"{fmt(b['registered_us'], 2)} | {b['verdict']} |")
        md.append(f"| {d} | **claim** | | | | {'holds' if x['accuracy_holds'] else 'not all'} | | | "
                  f"{'holds' if x['speed_holds'] else 'not all'}; claim {'HOLDS' if x['claim_holds'] else 'not shown'} |")
    md += ["", "## Family C (new against deployed; McNemar, Holm over all tests at 0.05; 95 % intervals pointwise)", "",
           "| d | test | new vs deployed failures | p | Holm p | 95 % interval | verdict |", "|---|---|---|---|---|---|---|"]
    for t in C_tests:
        md.append(f"| {t['d']} | {t['test']} | {t['new_failures']:,} vs {t['deployed_failures']:,} | {t['mcnemar_p']:.3g} | "
                  f"{t['holm_p']:.3g} | [{t['lo95']:.3g}, {t['hi95']:.3g}] | {t['verdict']} |")
    md += ["", "## Composite times (us per shot; GPU term + CPU term on the Grace basis + handoff)", "",
           "| d | network | role | GPU | CPU (Grace) | handoff | composite | registered |", "|---|---|---|---|---|---|---|---|"]
    for r in rowsComp:
        md.append(f"| {r['d']} | {r['network']} | {r['role']} | {fmt(r['a_gpu_us'])} | {fmt(r['b_cpu_grace_us'])} | "
                  f"{fmt(r['c_handoff_us'])} | {fmt(r['composite_us'])} | {fmt(r['registered_composite_us'])} |")
    with open(os.path.join(a.out, "tables.md"), "x") as fh:
        fh.write("\n".join(md) + "\n")
    for d in D:
        x = res["distances"][str(d)]
        print(f"d = {d}: {x['row_network']}: accuracy {'holds' if x['accuracy_holds'] else 'not all'}, speed "
              f"{'holds' if x['speed_holds'] else 'not all'} -> claim {'HOLDS' if x['claim_holds'] else 'not shown'}",
              flush=True)
    print("family C: " + "; ".join(f"d{t['d']} {t['test']}: {t['verdict']}" for t in C_tests), flush=True)
    print(f"family B: {'established' if why is None else 'not established (' + why + ')'}"
          + "".join(f"; {p}" for p in cpu_problems), flush=True)
    print(f"outputs in {a.out}: " + ", ".join(sorted(os.listdir(a.out))), flush=True)


if __name__ == "__main__":
    main()
