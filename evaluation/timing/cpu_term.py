"""The CPU term (b) of the composite and its conversion to the Grace basis.

  term  (-> term_d<d>_rep<r>.json; another network's mask, e.g. SURF-v2's: term_<network>_rep<r>.json)
        the routed shots of a network on its 200,000 validation shots (chunks 0-19, in order) decoded one at a time by
        correlated PyMatching, single-threaded, pinned to one core: 50 untimed warm-up decodes, then perf_counter
        around each matcher.decode(syndrome, enable_correlations=True) (production's loop, below unchanged); term =
        (sum of the routed shots' times) / 200,000, i.e. the routed fraction times the conditional cost of a routed
        shot, in us per shot. The routed shots come from a mask computed once, before the repetitions, as the studies
        did: `python -m timing.deployed` (the deployed executable's c, routed = |c| < tau; --logits fp32 for the
        PyTorch reference's), or the studies' own mask (routing/val_d<d>.npz), which is accepted only for the studies'
        validation shots (the shots' digest is checked for every mask).
  ref   the same loop over all shots of validation chunk 0 (10,000 shots): the node's correlated-PM time per block
        (--plain: plain PM's, the host gate's measurement at d = 9 and 13; not used by convert).
  convert  b_node = the slowest of the term's repetitions; ratio = G(d) / median of the ref repetitions, where G(d) is
        the Grace-basis correlated-PM time per block (timing/grace_basis.csv); b_grace = b_node x ratio.

The post hoc controls Ising M1 / M4 + correlated PM (README, "The two post hoc controls"): their CPU part is the
correlated-PM share, PM on the detectors Ising's pre-decoder leaves.
  residuals  on a GPU, with NVIDIA's code (convlines.opponents): Ising M1 and M4 with the official numerics on
        validation chunk 0 -> OUT/res_d<d>_<M>_val000.npz (pre_L, res, obs) and the content sha256 of res and pre_L.
  share  the same loop over those residual detectors (--res: their directory): the node's correlated-PM time per
        block after the pre-decoder (--plain: plain PM's, the conversion's denominator at d = 13).
  convert --share  per model, share_grace = the median share x factor, factor = Ising's Grace-basis PM share after the
        model at d = 13 (grace_basis.csv: the paper's Table VII) / the node's plain-PM share after the model at d = 13
        (median); beside it, factor_beside = G(13) / the node's correlated PM per block at d = 13 (ref, median). The
        d = 13 shares and ref must be in OUT. It also gives the slowest repetition (node_corr_share_slowest_us): on
        one Apple M4 core, plus the plan time, that is a control's speed on the M4 basis (the slowest counts there, as
        for the M4 basis' other CPU parts).

The studies ran three fresh-process repetitions of each, interleaved, on one exclusive CPU node (study 2: AMD EPYC 7713,
study 3: AMD EPYC 7443), all networks of a study in one job, after the freeze. The ratio converts the node's own time
to the Grace basis with the same node's correlated-PM speed, so the node's absolute speed cancels to first order.

    python -m timing.cpu_term term --d 9 --val DIR --mask MASKS/val_d9.npz --rep 1 --core 8 --out OUT
    python -m timing.cpu_term ref --d 9 --val DIR --rep 1 --core 8 --out OUT
    python -m timing.cpu_term convert --d 9 --out OUT
    python -m timing.cpu_term residuals --d 9 --val DIR --out RES
    python -m timing.cpu_term share --d 9 --model M1 --res RES [--plain] --rep 1 --core 8 --out OUT
    python -m timing.cpu_term convert --share --d 9 --out OUT"""
import argparse
import csv
import glob
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def loop(m, syn, correlated, warmup=50):
    """The official single-shot loop, returning every shot's time (us) and decision (production's code)."""
    kw = {"enable_correlations": True} if correlated else {}
    for i in range(min(warmup, len(syn))):
        m.decode(syn[i], **kw)
    ts = np.empty(len(syn))
    dec = np.empty(len(syn), np.uint8)
    pc = time.perf_counter
    for i in range(len(syn)):
        t0 = pc()
        p = m.decode(syn[i], **kw)
        ts[i] = pc() - t0
        dec[i] = p[0]
    return ts * 1e6, dec


def grace_numerator(d, decoder="correlated PM"):
    for r in csv.DictReader(open(os.path.join(HERE, "grace_basis.csv"))):
        if r["decoder"] == decoder and int(r["d"]) == d:
            return float(r["grace_us_per_shot"]), r["source"]
    raise KeyError((d, decoder))


def median_of(out, pattern, key="per_block_us"):
    v = [json.load(open(f))[key] for f in sorted(glob.glob(os.path.join(out, pattern)))]
    assert v, f"no {pattern} in {out}"
    return float(np.median(v)), v


def pin(core):
    """Pins the process to one core where the OS allows it (Linux); macOS has no affinity call, so the process runs
    unpinned there and its record says so."""
    if not hasattr(os, "sched_setaffinity"):
        return False
    os.sched_setaffinity(0, {core if core >= 0 else max(os.sched_getaffinity(0))})
    return True


def convert_share(a):
    beside, _ = grace_numerator(13)
    ref13, _ = median_of(a.out, "ref_d13_rep*.json")
    res = dict(d=a.d, factor_beside=beside / ref13, node_corr_ref_d13_us=ref13, grace_corr_d13_us=beside, models={})
    for m in ("M1", "M4"):
        g13, src = grace_numerator(13, f"PM share after Ising {m}")
        p13, _ = median_of(a.out, f"share_plain_d13_{m}_rep*.json")
        node, reps = median_of(a.out, f"share_d{a.d}_{m}_rep*.json")
        res["models"][m] = dict(node_corr_share_us=node, node_corr_share_reps_us=reps,
                                node_corr_share_slowest_us=max(reps), node_plain_share_d13_us=p13,
                                grace_share_d13_us=g13, grace_share_d13_source=src, factor=g13 / p13,
                                share_grace_us=node * g13 / p13, share_grace_beside_us=node * res["factor_beside"])
    json.dump(res, open(os.path.join(a.out, f"share_grace_d{a.d}.json"), "w"), indent=1)
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("term", "ref", "convert", "residuals", "share"))
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--val", default="")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--core", type=int, default=-1)
    ap.add_argument("--mask", default="", help="term: timing.deployed's val_d<d>.npz or the studies' routing/val_d<d>.npz")
    ap.add_argument("--plain", action="store_true", help="ref, share: plain PM instead of correlated PM")
    ap.add_argument("--model", choices=("M1", "M4"), help="share: whose residual detectors")
    ap.add_argument("--res", default="", help="share: the directory residuals wrote")
    ap.add_argument("--share", action="store_true", help="convert: the controls' correlated-PM share")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.what == "convert" and a.share:
        return convert_share(a)
    if a.what == "convert":
        terms = [json.load(open(f))["term_us"] for f in sorted(glob.glob(os.path.join(a.out, f"term_d{a.d}_rep*.json")))]
        refs = [json.load(open(f))["per_block_us"] for f in sorted(glob.glob(os.path.join(a.out, f"ref_d{a.d}_rep*.json")))]
        g, src = grace_numerator(a.d)
        ratio = g / float(np.median(refs))
        res = dict(d=a.d, term_reps_us=terms, b_node_us=max(terms), ref_reps_us=refs, node_median_us=float(np.median(refs)),
                   grace_numerator_us=g, grace_numerator_source=src, ratio=ratio, b_grace_us=max(terms) * ratio)
        json.dump(res, open(os.path.join(a.out, f"b_d{a.d}.json"), "w"), indent=1)
        print(json.dumps(res))
        return
    import hashlib
    import pymatching
    import stim
    from convlines import circuits, testset
    if a.what == "residuals":                            # on a GPU: the share's inputs, once
        from convlines import opponents
        det, obs = testset.load(os.path.join(a.val, testset.name("val", a.d, 0)), a.d)
        for m in ("M1", "M4"):
            pipe, tf32, _ = opponents.ising_module(m, a.d)
            pre, res = opponents.ising_predecode(pipe, tf32, det)
            testset.write_npz(os.path.join(a.out, f"res_d{a.d}_{m}_val000.npz"), dict(pre_L=pre, res=res, obs=obs))
            print(f"d{a.d} {m}: residual detectors {res.shape}, {res.sum(1).mean():.3f} per shot (input "
                  f"{det.sum(1).mean():.3f}); content sha256 res {hashlib.sha256(res.tobytes()).hexdigest()} pre_L "
                  f"{hashlib.sha256(pre.tobytes()).hexdigest()}", flush=True)
        return
    res = dict(what=a.what, d=a.d, rep=a.rep, core=a.core, cpu=platform.processor() or platform.machine(),
               python=sys.version.split()[0], numpy=np.__version__, stim=stim.__version__,
               pymatching=pymatching.__version__, load1=os.getloadavg()[0], start=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    try:
        res["cpu"] = open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t")
    except (OSError, IndexError):
        if sys.platform == "darwin":                     # macOS names its CPU here, e.g. "Apple M4"
            import subprocess
            res["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                        text=True).stdout.strip() or res["cpu"]
    if a.what == "share":                                # Ising's residual detectors of validation chunk 0
        assert a.model and a.res, "share needs --model and --res"
        m = circuits.matcher(a.d, correlated=not a.plain)
        f = os.path.join(a.res, f"res_d{a.d}_{a.model}_val000.npz")
        syn = np.ascontiguousarray(np.load(f)["res"], dtype=np.uint8)
        assert syn.shape[1] == m.num_detectors, f"{f}: not d = {a.d}'s detectors"
        res["pinned"] = pin(a.core)
        ts, _ = loop(m, syn, not a.plain)
        res.update(model=a.model, residuals=os.path.basename(f), shots=len(syn),
                   residuals_content_sha256=hashlib.sha256(syn.tobytes()).hexdigest(),
                   kind="plain" if a.plain else "correlated", per_block_us=float(ts.mean()),
                   end=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        print(f"d{a.d} {a.model} share rep {a.rep}: {res['kind']} PM {res['per_block_us']:.2f} us per block",
              flush=True)
        name = f"share{'_plain' if a.plain else ''}_d{a.d}_{a.model}_rep{a.rep}.json"
        json.dump(res, open(os.path.join(a.out, name), "w"), indent=1)
        return
    from convlines import decoder
    from timing import deployed
    chunks = range(20) if a.what == "term" else [0]
    sh = [testset.load(os.path.join(a.val, testset.name("val", a.d, j)), a.d) for j in chunks]
    det = np.concatenate([s[0] for s in sh])
    m = circuits.matcher(a.d, correlated=not a.plain)
    if a.what == "term":                                 # the mask of these shots, computed once beforehand
        assert a.mask and not a.plain, "term needs --mask (and correlated PM)"
        z = deployed.load(a.mask, a.d, det, np.concatenate([s[1] for s in sh]))
        mask = z["routed"]
        assert "tau" not in z or np.isfinite(z["tau"]), f"{a.mask}: c only, no routing (the network had no policy yet)"
        res["pinned"] = pin(a.core)
        ts, _ = loop(m, np.ascontiguousarray(det[mask]), True)
        res.update(network=str(z.get("network", decoder.NETWORKS[a.d])), tau=float(z["tau"]) if "tau" in z else None,
                   shots=len(det), mask=os.path.basename(a.mask), mask_kind=z["kind"],
                   mask_sha256=hashlib.sha256(open(a.mask, "rb").read()).hexdigest(), routed=int(mask.sum()),
                   routed_fraction=float(mask.mean()), term_us=float(ts.sum() / len(det)),
                   conditional_mean_us=float(ts.mean()) if len(ts) else None, shots_sha256=z["shots_sha256"])
        print(f"d{a.d} term rep {a.rep}: {res['term_us']:.3f} us per shot ({res['routed']} routed, conditional mean "
              f"{res['conditional_mean_us'] or 0:.2f} us)", flush=True)
    else:
        res["pinned"] = pin(a.core)
        ts, _ = loop(m, np.ascontiguousarray(det), not a.plain)
        res.update(shots=len(det), kind="plain" if a.plain else "correlated", per_block_us=float(ts.mean()))
        print(f"d{a.d} ref rep {a.rep}: {res['kind']} PM {res['per_block_us']:.2f} us per block", flush=True)
    res["end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    what = "ref_plain" if a.plain else a.what
    who = f"d{a.d}" if res.get("network", decoder.NETWORKS[a.d]) == decoder.NETWORKS[a.d] else res["network"]
    json.dump(res, open(os.path.join(a.out, f"{what}_{who}_rep{a.rep}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
