"""The Decoder Switching stage's processes (speed_rt/gapsw). Every timed process is one fresh single-threaded process pinned to
one core; it loads its shots and builds its decoders first, then checks the host (hostwin.py: the timed core idle over 5 s,
the plain-PM probe), then times its loop inside one window whose host counters it keeps, and writes one record
(OUT/rec.json, once) with `accepted` and the reasons. Exit codes: 0 accepted, 4 rejected by a host check (nothing wrong with the
process, replaced by the driver), 3 failed (a check of the process itself, a crash is any other non-zero).

  fe     the front end of gap switching on the 50,000 timing shots of one d, one shot at a time: surf_v2.gapsw.loop (plain PM's
         decode with its weight, node b's bit, the split graph's decode; 50 untimed warm-up shots, perf_counter around each shot),
         term = the mean per shot over all shots; its gaps, weights and answers are checked bit for bit against the mask's
  term   correlated PM on the shots routed at the frozen tau (the mask), one at a time: timing.cpu_term.loop (50 untimed warm-up
         shots, perf_counter around each decode); term = the sum of the routed shots' times / all shots
  mask   untimed, before any timed window: the front end's batch decode of one d's timing shots in parallel shares (fork pool),
         the gap, the routed shots at the frozen tau (surf_v2/gapsw/policy_d<d>.json) -> OUT/mask_d<d>.npz and .json
  probe  the host probe alone (the pre-screen's reference)
  load   an untimed filler (the probe's decode loop until SIGTERM), for the parallel driver (drive_par.py)
  selftest  hostwin.selftest on the core

    python -m speed_rt.gapsw.gs_time fe --d 9 --shots SHOTS --manifest MAN --masks MASKS --probe-file F --probe-ref US --rep 1 --core 8 --out OUT
(arguments of the others: --help of each)"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from speed_rt.gapsw import gslib as G  # noqa: E402
from speed_rt.gapsw import hostwin as H  # noqa: E402

REJECT, FAILED = 4, 3


def gapsw_module():
    from surf_v2 import gapsw
    assert gapsw.gapsw_sha256() == G.GAPSW_SHA256, "surf_v2/gapsw.py is not the frozen one"
    return gapsw


def stage_sha256():
    import hashlib
    fs = sorted(f for f in os.listdir(HERE) if f.endswith((".py", ".sh", ".json")))
    return hashlib.sha256("".join(f"{G.sha256_file(os.path.join(HERE, f))}  {f}\n" for f in fs).encode()).hexdigest()


def base_record(a, kind):
    from convlines import provenance
    return dict(schema="gs-1", kind=kind, rep=getattr(a, "rep", None), core=a.core, start=G.now(), env=G.env_record(),
                code_digest=provenance.code_digest(), release_commit=provenance.commit(G.ROOT), stage_sha256=stage_sha256(),
                gapsw_sha256=G.GAPSW_SHA256, limits=H.LIMITS, reasons=[])


def probe_data(a):
    """The probe's matcher and shots (SURF-v2's validation chunk 0 of d = 9), built before the window."""
    from convlines import circuits
    ndet = circuits.circuit(G.PROBE_D).num_detectors
    det, obs = G.load_chunk(os.path.join(a.probe_file, G.PROBE_FILE) if os.path.isdir(a.probe_file) else a.probe_file, ndet)
    det = np.ascontiguousarray(det)
    assert G.shots_sha256(det, obs) == G.PROBE_SHOTS_SHA256, "the probe's shots are not the registered chunk"
    return circuits.matcher(G.PROBE_D, False), det


def probe_run(m, det, passes):
    """One untimed pass (the caches are cold after the process's loads), then `passes` passes: mean us per shot of each."""
    from timing import cpu_term
    cpu_term.loop(m, det, False)
    return [float(cpu_term.loop(m, det, False)[0].mean()) for _ in range(passes)]


def judged(res):
    """False only in a dry run with GS_DRY_NOCHECK set: the host checks are measured and recorded but decide nothing (never in a
    registered session: the flag needs the manifest's dry mark)."""
    if os.environ.get("GS_DRY_NOCHECK") and res.get("dry"):
        res["ignored_checks"] = True
        return False
    return True


def prechecks(a, res):
    """The host checks right before the window: idle 5 s, then the probe; both recorded. -> True when both hold."""
    if os.environ.get("GS_FORCE_REJECT") and res.get("dry"):
        res["reasons"].append("forced rejection (dry run only: the replacement path)")
    res["pre_idle"] = H.idle_check(a.core)
    m, det = probe_data(a)
    ref = None if getattr(a, "probe_ref", None) in (None, 0) else float(a.probe_ref)
    res["pre_probe"] = H.probe_check(probe_run(m, det, H.LIMITS["probe_passes"]), ref)
    del m
    if res["reasons"]:                           # the dry run's forced rejection: no window is spent on it
        return False
    if not judged(res):                          # a dry run on a noisy host that only measures times
        return True
    if not res["pre_idle"]["ok"]:
        res["reasons"].append("idle: the timed core was busy or stolen before the window")
    if not res["pre_probe"]["ok"]:
        res["reasons"].append("probe: plain PM per block off the session's reference or its spread too large")
    return res["pre_idle"]["ok"] and res["pre_probe"]["ok"]


def finish(a, res, window_ok=True):
    res["end"] = G.now()
    res["accepted"] = bool(not res["reasons"] and window_ok)
    os.makedirs(a.out, exist_ok=True)
    G.write_once(os.path.join(a.out, "rec.json"), res)
    print(f"{res['kind']} rec rec.json accepted {res['accepted']} {res['reasons']}", flush=True)
    raise SystemExit(0 if res["accepted"] else (FAILED if res.get("failed") else REJECT))


def timed(a, res, fn):
    """Pin, check the host, run fn() inside the window -> its result, or None when a pre-check rejected."""
    res["pinned"] = G.pin(a.core)
    assert res["pinned"], "the process is not pinned to one core"
    if not prechecks(a, res):
        finish(a, res, False)
    print("TIMED-START", flush=True)
    w = H.Window(a.core).start()
    out = fn()
    res["window"] = w.stop()
    print("TIMED-END", flush=True)
    if judged(res) and not res["window"]["ok"]:
        res["reasons"].append("window: the timed core was busy by others or stolen during the window")
    return out


def load_mask(a, d, dg, gapsw):
    pol, psha = gapsw.policy(d)
    assert pol["d"] == d and pol["tau"] is not None and not pol["dry"] and pol["gapsw_sha256"] == G.GAPSW_SHA256, \
        f"policy of d = {d}: not the frozen one"
    mrec = json.load(open(os.path.join(a.masks, f"mask_d{d}.json")))
    z = np.load(os.path.join(a.masks, f"mask_d{d}.npz"))
    assert mrec["shots_sha256"] == dg and mrec["tau"] == pol["tau"] and mrec["policy_sha256"] == psha, \
        f"mask of d = {d}: not this set's or this policy's"
    assert mrec["npz_sha256"] == G.sha256_file(os.path.join(a.masks, f"mask_d{d}.npz")), "mask: not the file its record names"
    return pol, psha, mrec, z


# ---- the timed processes --------------------------------------------------------------------------------------------
def fe(a):
    gapsw = gapsw_module()
    man = G.manifest(a.manifest)
    det, obs, dg = G.load_ds(a.shots, man, a.d, a.chunks)
    pol, psha, mrec, z = load_mask(a, a.d, dg, gapsw)
    res = base_record(a, "fe")
    res.update(dry=bool(man.get("dry")), d=a.d, shots=len(det), shots_sha256=dg, tau=pol["tau"], policy_sha256=psha, mask_record=mrec["npz_sha256"])
    front = gapsw.FrontEnd(a.d)
    ext = gapsw.extended(det)
    del det
    out = timed(a, res, lambda: gapsw.loop(front, ext))
    dec, wt, gap, tp, tf = out
    same = bool(np.array_equal(gap, z["gap"]) and np.array_equal(wt, z["weight"]) and np.array_equal(dec, z["dec"]))
    routed = gap < pol["tau"]
    res.update(term_us=float(tf.mean()), term_sd_us=float(tf.std(ddof=1)), plain_in_front_us=float(tp.mean()), routed=int(routed.sum()),
               routed_fraction=float(routed.mean()), equals_mask=same, mask_routed=mrec["routed"])
    if not same or int(routed.sum()) != mrec["routed"]:
        res["reasons"].append("check: the timed loop's answers, weights or gaps differ from the batch mask's")
        res["failed"] = True
    print(f"d{a.d} fe rep {a.rep}: {res['term_us']:.2f} us per shot (plain PM's decode {res['plain_in_front_us']:.2f}), routed "
          f"{res['routed']}", flush=True)
    finish(a, res)


def term(a):
    from convlines import circuits
    from timing import cpu_term
    gapsw = gapsw_module()
    man = G.manifest(a.manifest)
    det, obs, dg = G.load_ds(a.shots, man, a.d, a.chunks)
    pol, psha, mrec, z = load_mask(a, a.d, dg, gapsw)
    routed = z["routed"].astype(bool)
    assert int(routed.sum()) == mrec["routed"] and np.array_equal(routed, z["gap"] < pol["tau"]), "mask: routed is not gap < tau"
    res = base_record(a, "term")
    res.update(dry=bool(man.get("dry")), d=a.d, shots=len(det), shots_sha256=dg, tau=pol["tau"], policy_sha256=psha, mask_record=mrec["npz_sha256"],
               routed=int(routed.sum()), routed_fraction=float(routed.mean()))
    m = circuits.matcher(a.d, correlated=True)
    rdet = np.ascontiguousarray(det[routed])
    del det
    ts, _ = timed(a, res, lambda: cpu_term.loop(m, rdet, True))
    allt = np.zeros(len(routed))                 # the term's per-shot values: the routed shots' times, 0 for the others
    allt[routed] = ts
    res.update(term_us=float(ts.sum() / len(routed)), term_sd_us=float(allt.std(ddof=1)), conditional_mean_us=float(ts.mean()) if len(ts) else None)
    print(f"d{a.d} term rep {a.rep}: {res['term_us']:.3f} us per shot ({res['routed']} routed, conditional mean "
          f"{res['conditional_mean_us']:.2f} us)", flush=True)
    finish(a, res)


def mask(a):
    gapsw = gapsw_module()
    from convlines import provenance, testset
    man = G.manifest(a.manifest)
    det, obs, dg = G.load_ds(a.shots, man, a.d, a.chunks)
    pol, psha = gapsw.policy(a.d)
    assert pol["d"] == a.d and pol["tau"] is not None and not pol["dry"] and pol["gapsw_sha256"] == G.GAPSW_SHA256
    front = gapsw.FrontEnd(a.d)
    n = len(det) // G.CHUNK
    global _FE, _DET
    _FE, _DET = front, det
    t0 = time.time()
    with mp.get_context("fork").Pool(a.workers) as pool:
        parts = pool.map(_share, range(n), chunksize=1)
    dec, wt, gap = (np.concatenate([p[i] for p in parts]) for i in range(3))
    routed = gap < pol["tau"]
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"mask_d{a.d}")
    testset.write_npz(stem + ".npz", dict(dec=dec, weight=wt, gap=gap, routed=routed.astype(np.uint8)))
    import hashlib
    rec = dict(what="Decoder Switching stage: the front end's batch decode of the timing shots and the routed mask", d=a.d,
               shots=len(det), shots_sha256=dg, tau=pol["tau"], q=pol["q"], policy_sha256=psha, routed=int(routed.sum()),
               routed_fraction=float(routed.mean()), gap_zero=int((gap == 0).sum()), gap_min=float(gap.min()),
               gap_sha256=hashlib.sha256(np.ascontiguousarray(gap).tobytes()).hexdigest(),
               dec_sha256=hashlib.sha256(np.ascontiguousarray(dec).tobytes()).hexdigest(), npz_sha256=G.sha256_file(stem + ".npz"),
               workers=a.workers, seconds=time.time() - t0, gapsw_sha256=G.GAPSW_SHA256, env=G.env_record(),
               code_digest=provenance.code_digest(), release_commit=provenance.commit(G.ROOT))
    assert rec["gap_min"] >= 0, "a negative gap"
    G.write_once(stem + ".json", rec)
    print(f"d{a.d}: {rec['routed']} of {len(det)} shots routed at tau {pol['tau']!r} (q {pol['q']}), {rec['seconds']:.0f} s, "
          f"gap sha256 {rec['gap_sha256'][:16]}", flush=True)


def _share(j):
    sl = slice(j * G.CHUNK, (j + 1) * G.CHUNK)
    return _FE.batch(_DET[sl])


def probe(a):
    """The host probe alone: a record with the probe's passes (the pre-screen's reference comes from several of these)."""
    G.pin(a.core)
    res = base_record(a, "probe")
    m, det = probe_data(a)
    res["probe"] = H.probe_check(probe_run(m, det, H.LIMITS["probe_passes"]), None)
    res["end"] = G.now()
    os.makedirs(a.out, exist_ok=True)
    G.write_once(os.path.join(a.out, f"probe_rep{a.rep}.json"), res)
    print(f"probe rep {a.rep}: {res['probe']['values_us']} us per shot, spread {res['probe']['spread']:.4f}", flush=True)


def load(a):
    """An untimed filler: the probe's plain-PM decode loop on the core until SIGTERM (nothing is written). The parallel driver runs it
    on every listed CPU that has no timed process, so that every timed window sees the same number of busy cores."""
    import signal
    G.pin(a.core)
    m, det = probe_data(a)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    from timing import cpu_term
    while not stop:
        cpu_term.loop(m, det, False)


def selftest(a):
    r = H.selftest(a.core)
    print(json.dumps(r))
    raise SystemExit(0 if r["ok"] else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, shots=True, timed_=True):
        if shots:
            p.add_argument("--shots", required=True, help="the directory of the drawn chunk files")
            p.add_argument("--manifest", required=True, help="the draw's manifest (srt-gs.json)")
        p.add_argument("--chunks", type=int, default=None, help="dry runs only: the first N chunks of a DS set")
        if timed_:
            p.add_argument("--core", type=int, required=True)
            p.add_argument("--rep", type=int, required=True)
            p.add_argument("--probe-file", required=True, help="SURF-v2's validation chunk 0 of d = 9 (file or its directory)")
            p.add_argument("--probe-ref", type=float, required=True, help="the session's plain-PM reference (us per shot)")
        p.add_argument("--out", required=True)
    for name, fn in (("fe", fe), ("term", term)):
        p = sub.add_parser(name)
        p.add_argument("--d", type=int, required=True, choices=G.DS_DISTANCES)
        p.add_argument("--masks", required=True)
        common(p)
        p.set_defaults(fn=fn)
    p = sub.add_parser("mask")
    p.add_argument("--d", type=int, required=True, choices=G.DS_DISTANCES)
    p.add_argument("--workers", type=int, default=8)
    common(p, timed_=False)
    p.set_defaults(fn=mask)
    p = sub.add_parser("probe")
    p.add_argument("--core", type=int, required=True)
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--probe-file", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=probe)
    p = sub.add_parser("load")
    p.add_argument("--core", type=int, required=True)
    p.add_argument("--probe-file", required=True)
    p.set_defaults(fn=load)
    p = sub.add_parser("selftest")
    p.add_argument("--core", type=int, required=True)
    p.set_defaults(fn=selftest)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    a.fn(a)


if __name__ == "__main__":
    main()
