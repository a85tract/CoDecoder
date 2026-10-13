"""R17 (ising_compare/r17/PLAN_R17.md): study 4's d = 17 row re-registered with the threshold the C4.2 rule chose on
deep17's calibration set K17d (q = 8 %, tau 4.539173126220703), on a new test set T6. The decoder, the opponents and
their settings are study 4's (freeze c500ef8e, frozen tree f1c6a0e9, S4-R3's parts and records); tau, the test set
and, after stage E3, the GPU engine change. Run from the frozen tree's ising_compare/ with
PYTHONPATH=.:../neural_emulation; every pin comes from r17.json. S4-R3's driver (scripts/s4r3/r3.py, ising-worker
a31d525e) is the model for every step; the frozen functions it calls are the same.
  pins         every host: the frozen code, Stim, numpy and the circuit (icmp.t4.frozen_checks), the bundle's files and
               S4-R3's records against r17.json
  sample       x86-64 with AVX2: T6, 100 chunks of 10,000 shots, seeds 91,170,000 + j, icmp.t4's file format; --dry
               draws V17b chunks 0-1 from their own seeds instead, checks them against V17b and writes them as T6dry
  t5regen      T5 chunks again from S4-R3's manifest (each file equal to its sha256), for the hopper check
  mask         the q = 8 % validation routing mask: |double(c)| < tau on V17b, c from the development record
  cputerm      one repetition of the CPU term on that mask (speed2.s11.cpu12's loop, correlated PM, one pinned core)
  hopper       the GPU node, before any T6 read: every Ising part on T5 chunks 0-19, + plain and + correlated PM,
               against S4-R3's H100 decisions, and the torch residuals of V17b chunk 0 against PB's files (ours is
               checked beside it by the frozen speed2.s19.t4ours val on all 200,000 V17b shots)
  read         the GPU node: ours (the frozen executable at the registered tau; both thresholds come from its c),
               every Ising part + plain and + correlated PM (one residual, packed, for both), plain and correlated PM
  bposd        an A100 node: the two BP-OSD arms on T6 chunks 0-19 or a share of them
  bposd-merge  the shares of each arm joined in chunk order
  cpurecord    the CPU job's files as one record (s4_cpu.sbatch's format)
  inventory    the pre-commit: the sha256 of every analysis input and the exact command
  analyze      once: the 16 cells at q = 8 %, the preservation test against the registered tau, the composite"""
import argparse
import csv
import glob
import hashlib
import io
import json
import os
import subprocess
import sys
import time

import numpy as np

D = 17
M = 4896                                 # detectors of the d = 17 verdict circuit (speed2.s19.vsets)
CHUNK = 10_000
BASE, DRY_BASE = 91_000_000, 69_000_000  # T6's seed namespace; V17b's, the dry runs' stand-in (speed2.s19.vsets)
DRY_CHUNKS = (0, 1)
PREFIX = "d17_b5_r787"                   # the development record's members
OURS = "ours_deployed"
OPPONENTS = ("pm", "pm_corr", "M1", "M4", "M1_corr", "M4_corr", "bp60", "minllr")
ARMS = ("minllr", "bp60")
HERE = os.path.dirname(os.path.abspath(__file__))


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def dump(obj, path, mode="w"):
    with open(path, mode) as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=float)
        fh.write("\n")


def parse_chunks(s):
    out = []
    for part in str(s).split(","):
        a, _, b = part.strip().partition("-")
        out += list(range(int(a), int(b or a) + 1))
    assert out and len(set(out)) == len(out), s
    return out


def load(a):
    """(freeze, record): r17.json and study 4's freeze, the freeze checked against the record."""
    rec = json.load(open(a.record))
    fz = json.load(open(a.freeze))
    assert sha_file(a.freeze) == rec["base_freeze"]["sha256"], "not freeze c500ef8e"
    return fz, rec


def s4r3_file(a, rec, rel):
    """One of S4-R3's records (committed at a31d525e), checked against r17.json."""
    p = os.path.join(a.s4r3, rel)
    assert sha_file(p) == rec["s4r3"]["files"][rel], f"{rel}: not S4-R3's record"
    return p


def tset(dry):
    """(split, {j: seed}) of T6, or with dry of its stand-in: V17b's chunks 0-1 under their own seeds."""
    if dry:
        return "test6dry", {j: DRY_BASE + 10_000 * D + j for j in DRY_CHUNKS}
    return "test6", {j: BASE + 10_000 * D + j for j in range(100)}


def name(split, j):
    return f"d{D}_p0.006_X_{split}_{j:03d}.npz"


def opponent_of(n):
    """The opponent a row counts for (scripts/s4r3/r3.py's opponent_of)."""
    if n.startswith(("M1_", "M4_")):
        return n[:2] + ("_corr" if n.endswith("_corr") else "")
    return {"pm": "pm", "pm_corr": "pm_corr", "bposd_minllr": "minllr", "bposd_bp60": "bp60"}[n]


# ---- T6 ------------------------------------------------------------------------------------------------------------------
def t6(a, rec, rehash=True):
    """T6's manifest (or the dry stand-in's) against r17.json: split, chunks and seeds, the circuit, Stim and numpy, the
    record it was drawn under; with rehash every file hashed again."""
    man = json.load(open(a.manifest))
    dry = bool(man.get("dry"))
    split, seeds = tset(dry)
    assert man["revision"] == "R17" and man["split"] == split, "not R17's T6 manifest"
    assert set(man["files"]) == {name(split, j) for j in seeds}, "the files differ from T6's chunks"
    for j, s in seeds.items():
        e = man["files"][name(split, j)]
        assert (e["j"], e["seed"], e["shots"]) == (j, s, CHUNK), f"chunk {j}: entry differs"
        if rehash:
            assert sha_file(os.path.join(a.shots, name(split, j))) == e["sha256"], f"chunk {j}: hash mismatch"
    ts = rec["test_set"]
    assert man["circuit_sha256"] == ts["circuit_sha256"] and man["versions"] == ts["versions"], "circuit or sampler"
    if not dry:
        assert man["record_sha256"] == sha_file(a.record), "T6 was drawn under another r17.json"
    return man, dict(revision="R17", t6_manifest_sha256=sha_file(a.manifest), dry=dry)


def t6_load(man, chunks, shot_dir, packed=False):
    """Chunks of T6, each loaded from the very bytes hashed against the manifest (icmp.t4.load's rule)."""
    dets, obs = [], []
    for j in chunks:
        f = name(man["split"], j)
        raw = open(os.path.join(shot_dir, f), "rb").read()
        assert sha_bytes(raw) == man["files"][f]["sha256"], f"{f}: hash mismatch"
        z = np.load(io.BytesIO(raw))
        assert int(z["seed"]) == man["files"][f]["seed"], f"{f}: seed differs from the manifest"
        dets.append(z["det"])
        obs.append(z["obs"])
    det = np.concatenate(dets)
    return (det if packed else np.unpackbits(det, axis=1)[:, :M]), np.concatenate(obs).astype(np.uint8)


def chunks_of(man):
    return sorted(e["j"] for e in man["files"].values())


# ---- every host ----------------------------------------------------------------------------------------------------------
def cmd_pins(a):
    from icmp import t4
    fz, rec = load(a)
    t4.frozen_checks(fz, {D: [0]})                      # every frozen code file, Stim and numpy, the circuit
    bad, missing = [], []
    for f, h in rec["method"]["bundle_files"].items():
        p = os.path.join(a.bundle, f)
        (missing if not os.path.exists(p) else bad if sha_file(p) != h else []).append(f)
    for f in rec["s4r3"]["files"]:
        p = os.path.join(a.s4r3, f)
        (missing if not os.path.exists(p) else bad if sha_file(p) != rec["s4r3"]["files"][f] else []).append(f)
    pol = os.path.join(HERE, os.path.basename(rec["method"]["policy"]["file"]))
    if sha_file(pol) != rec["method"]["policy"]["sha256"] or \
            json.load(open(pol))["chosen"]["tau_repr"] != rec["method"]["tau"]:
        bad.append("policy")
    out = dict(when=now(), host=t4.host(), bad=bad, missing=missing, record=sha_file(a.record),
               bundle_files=len(rec["method"]["bundle_files"]), s4r3_files=len(rec["s4r3"]["files"]))
    if a.out:
        dump(out, a.out)
    print(f"pins: frozen code, Stim, numpy and circuit equal; {len(rec['method']['bundle_files'])} bundle files, "
          f"{len(rec['s4r3']['files'])} S4-R3 records; differ {bad}; missing {missing}", flush=True)
    assert not bad, f"pinned files differ: {bad}"
    assert not missing or a.allow_missing, f"pinned files missing: {missing}"


# ---- the sampler ---------------------------------------------------------------------------------------------------------
def cmd_sample(a):
    """T6 (or with --dry the V17b stand-in) into OUT/T6 (OUT/T6dry), which must not exist, and OUT/t6_manifest.json
    (t6dry_manifest.json). The dry run's chunks must equal V17b's (G14's content sha256), so the dry run shows that
    this sampler on this host reproduces registered shots bit for bit."""
    import stim
    from icmp import circuits, t4
    fz, rec = load(a)
    t4.frozen_checks(fz, {D: [0]})
    simd = t4.host()["simd"]
    assert "avx2" in simd, f"T6 is sampled on x86-64 with AVX2 only: {simd}"
    split, seeds = tset(a.dry)
    c = t4.circuit(D)
    assert circuits.sha(c) == rec["test_set"]["circuit_sha256"] and c.num_detectors == M
    tag = "T6dry" if a.dry else "T6"
    shots = os.path.join(a.out, tag)
    os.makedirs(shots)                                   # a second draw into the same directory stops here
    started, files = now(), {}
    for j, s in sorted(seeds.items()):
        det, obs = c.compile_detector_sampler(seed=s).sample(CHUNK, separate_observables=True)
        f = name(split, j)
        dp, ob = np.packbits(det.astype(np.uint8), axis=1), obs[:, 0].astype(np.uint8)
        t4.write_npz(os.path.join(shots, f), dict(det=dp, obs=ob, seed=np.int64(s)))
        files[f] = dict(j=j, seed=s, shots=CHUNK, sha256=sha_file(os.path.join(shots, f)))
        if a.dry:                                        # the stand-in must be V17b's chunk j itself
            from speed2.s19 import vsets
            want = vsets.manifest(D)[vsets.name(D, "v17b", j)]["content_sha256"]
            assert vsets.content_sha(dp, ob) == want, f"V17b chunk {j}: this sampler does not reproduce it"
            files[f]["v17b_content_sha256"] = want
        print(f, files[f]["sha256"][:16], flush=True)
    man = dict(revision="R17", test_set=tag, dry=a.dry, split=split, d=D, seed_rule=rec["test_set"]["seed"]
               if not a.dry else "V17b's: 69,000,000 + 10,000 x 17 + j", files=files,
               circuit_sha256=circuits.sha(c), versions=dict(stim=stim.__version__, numpy=np.__version__),
               host=t4.host(), code_sha256={"r17.py": sha_file(__file__), "icmp/t4.py": sha_file(t4.__file__)},
               record_sha256=sha_file(a.record), started=started, ended=now())
    mp = os.path.join(a.out, f"{tag.lower()}_manifest.json")
    with open(mp, "x") as fh:
        json.dump(man, fh, indent=1)
        fh.write("\n")
    for f in os.listdir(shots):
        os.chmod(os.path.join(shots, f), 0o444)
    os.chmod(shots, 0o555)
    os.chmod(mp, 0o444)
    print(f"{tag}: {len(files)} chunks; manifest {sha_file(mp)}", flush=True)


def cmd_t5regen(a):
    """T5's chunks drawn again from S4-R3's manifest; each file must equal its registered sha256."""
    from icmp import t4
    fz, rec = load(a)
    t4.frozen_checks(fz, {D: [0]})
    man = json.load(open(s4r3_file(a, rec, "data/study4/r3/t5_manifest.json")))
    c = t4.circuit(D)
    os.makedirs(a.out, exist_ok=True)
    for j in parse_chunks(a.chunks):
        f = t4.name(D, j)
        e, p = man["files"][f], os.path.join(a.out, f)
        if os.path.exists(p) and sha_file(p) == e["sha256"]:
            continue
        det, obs = c.compile_detector_sampler(seed=e["seed"]).sample(CHUNK, separate_observables=True)
        t4.write_npz(p, dict(det=np.packbits(det.astype(np.uint8), axis=1), obs=obs[:, 0].astype(np.uint8),
                             seed=np.int64(e["seed"])))
        assert sha_file(p) == e["sha256"], f"{f}: this host does not reproduce T5's file"
    print(f"T5 chunks {a.chunks}: equal to S4-R3's manifest on {t4.host()['cpu']}", flush=True)


# ---- the CPU term -------------------------------------------------------------------------------------------------------
def cmd_mask(a):
    """The q = 8 % routing of V17b (200,000 shots) from the development record's c (study 4's development H100). The
    registered routing must follow from the same c at the registered tau, and the new routed set lies inside it."""
    from icmp import t4
    fz, rec = load(a)
    assert sha_file(a.devrec) == rec["method"]["bundle_files"]["devrec/d17.npz"], "not the development record"
    z = np.load(a.devrec)
    c = np.asarray(z[PREFIX + "_c"])
    assert c.dtype == np.float32 and c.shape == (200_000,) and np.isfinite(c).all()
    reg = np.abs(c.astype(np.float64)) < float(rec["method"]["tau_registered"])
    assert np.array_equal(reg, z[PREFIX + "_routed"].astype(bool)), "the record's routing is not |c| < the registered tau"
    new = np.abs(c.astype(np.float64)) < float(rec["method"]["tau"])
    assert not (new & ~reg).any()
    t4.write_npz(a.out, dict(routed=new.astype(np.uint8)))
    # the planning numbers of the preservation test on V17b: a shot routed at the registered tau carries correlated
    # PM's decision in the record, so both thresholds' decisions follow from the record alone
    from scipy.stats import norm
    dec10, obs = z[PREFIX + "_decision"].astype(np.uint8), np.asarray(z["obs"], np.uint8).reshape(-1)
    dec8 = np.where(new, dec10, (c > 0).astype(np.uint8))
    f8, f10 = dec8 != obs, dec10 != obs
    b, cc, n = int((f8 & ~f10).sum()), int((~f8 & f10).sum()), len(obs)
    delta, zq = rec["preservation"]["delta"], float(norm.ppf(1 - rec["accuracy"]["alpha"] / 2))
    sd = np.sqrt((b + cc) / n) / np.sqrt(1_000_000)      # the paired difference's sd at 1,000,000 shots
    power = {f"{k}": float(norm.cdf((delta - dt) / sd - zq)) for k, dt in (("true_diff_0", 0.0),
             ("true_diff_val", (b - cc) / n), ("true_diff_half_delta", delta / 2))}
    info = dict(stage="R17: the q = 8 % routing mask of V17b", devrec_sha256=sha_file(a.devrec), tau=rec["method"]["tau"],
                routed=int(new.sum()), routed_fraction=float(new.mean()), routed_registered=int(reg.sum()),
                mask_sha256=sha_file(a.out), created=now(),
                val_preservation=dict(failures_q08=int(f8.sum()), failures_q10=int(f10.sum()), only_q08=b, only_q10=cc,
                                      shots=n, power_at_1e6=power, rule="normal approximation of the paired difference; "
                                      "power = P(upper end of the 97.5 % interval < delta)"))
    dump(info, a.out[:-4] + ".json")
    print(f"mask: {info['routed']} of 200,000 V17b shots routed ({info['routed_fraction']:.4%}); registered tau "
          f"{info['routed_registered']}; {info['mask_sha256'][:16]}; V17b failures q 8 % {int(f8.sum())}, q 10 % "
          f"{int(f10.sum())} (only {b} / {cc}); planning power at 1e6 {power}", flush=True)


def cmd_cputerm(a):
    """One repetition of the CPU term (speed2.s11.cpu12 term's procedure): 50 untimed warm-up decodes, then one
    correlated decode per routed V17b shot in order, perf_counter around each; term = sum / 200,000. --which q08: the
    q = 8 % mask (the term that counts); q10: the registered routing (the record's), timed beside it in the same job so
    that the saving is measured on one node. Every decision must equal the development record's."""
    os.sched_setaffinity(0, {a.core})
    import platform
    import pymatching
    import stim
    from icmp import circuits, pm, s4guard
    from speed2.s11.cpu12 import loop, tails
    from speed2.s19 import vsets
    from speed2.s9.hostmon import Snap
    fz, rec = load(a)
    s4guard.check_runtime(fz, fz["route"]["versions"])
    z = np.load(a.devrec)
    assert sha_file(a.devrec) == rec["method"]["bundle_files"]["devrec/d17.npz"]
    if a.which == "q08":
        info = json.load(open(a.mask[:-4] + ".json"))
        assert sha_file(a.mask) == info["mask_sha256"] and info["tau"] == rec["method"]["tau"]
        mask, tau, q = np.load(a.mask)["routed"].astype(bool), rec["method"]["tau"], rec["method"]["q"]
    else:
        mask, tau, q = z[PREFIX + "_routed"].astype(bool), rec["method"]["tau_registered"], rec["method"]["q_registered"]
    det, obs = vsets.load(D, vsets.VAL[D], range(20))
    assert len(det) == len(mask) == 200_000 and np.array_equal(obs, np.asarray(z["obs"], np.uint8).reshape(-1))
    syn = np.ascontiguousarray(det[mask])
    m = pm.matcher(D, True, circuits.circuit(D, noise=circuits.NOISE_VERDICT))
    hs = Snap(a.core)
    t0 = time.time()
    ts, dec = loop(m, syn, True)
    wall, host = time.time() - t0, hs.delta()
    want = z[PREFIX + "_decision"].astype(np.uint8)[mask]
    res = dict(stage="R17 CPU term (speed2.s11.cpu12 term's procedure, the q = 8 % mask)", what="term", rep=a.rep,
               core=a.core, pid=os.getpid(), host=platform.node(),
               cpu=open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t"),
               job=os.environ.get("SLURM_JOB_ID"), python=sys.version.split()[0], numpy=np.__version__,
               stim=stim.__version__, pymatching=pymatching.__version__, val_set=vsets.describe(D, vsets.VAL[D], range(20)),
               cand=PREFIX, d=D, which=a.which, tau=tau, q=q, mask_sha256=sha_bytes(mask.astype(np.uint8).tobytes()),
               routed=int(mask.sum()), routed_fraction=float(mask.mean()), shots=200_000,
               term_us=float(ts.sum() / 200_000), conditional_mean_us=float(ts.mean()), routed_times=tails(ts),
               wall_s=wall, hostmon=host, decisions_equal_record=bool(np.array_equal(dec, want)),
               decisions_differ=int((dec != want).sum()), dry=a.dry, end=now())
    tag = f"term_r17_{a.which}_rep{a.rep}"
    np.savez_compressed(os.path.join(a.out, tag + ".npz"), t_us=ts.astype(np.float32))
    dump(res, os.path.join(a.out, tag + ".json"))
    print(f"{tag}: term {res['term_us']:.3f} µs per shot ({res['routed']} routed, conditional mean "
          f"{res['conditional_mean_us']:.2f}, p99 {res['routed_times']['p99']:.1f}); decisions equal record "
          f"{res['decisions_equal_record']}", flush=True)
    return 0 if res["decisions_equal_record"] else 3


def cmd_cpurecord(a):
    """The CPU job's text records as one file (s4_cpu.sbatch's format), the per-shot npz arrays left out."""
    files = {}
    for f in sorted(os.listdir(a.dir)):
        p = os.path.join(a.dir, f)
        if f == "cpu_term.json" or not os.path.isfile(p) or f.endswith(".npz"):
            continue
        t = open(p).read()
        try:
            files[f] = json.loads(t)
        except ValueError:
            files[f] = t
    dump(dict(stage="R17's CPU term (b) on one exclusive Alpine amem node (study 4's s4_cpu.sbatch procedure)",
              revision="R17", job=os.environ.get("SLURM_JOB_ID"), dry=a.dry, files=files),
         os.path.join(a.dir, "cpu_term.json"))
    print("cpu_term.json:", len(files), "files", flush=True)


def b_term(cpu):
    """(b) on the Grace basis from the R17 CPU record: max(term) x correlated PM's Grace value / median(reference)."""
    f = cpu["files"]
    reps = [f[k] for k in sorted(f) if k.startswith("term_r17_q08_rep") and k.endswith(".json")]
    refs = [f[k] for k in sorted(f) if k.startswith("ref_d17_correlated_rep") and k.endswith(".json")]
    assert len(reps) == 3 and sorted(r["rep"] for r in reps) == [1, 2, 3] and all(r["decisions_equal_record"] for r in reps)
    assert len(refs) == 3 and len({r["host"] for r in reps + refs}) == 1, "the term and the references on one node"
    beside = [f[k]["term_us"] for k in sorted(f) if k.startswith("term_r17_q10_rep") and k.endswith(".json")]
    return [r["term_us"] for r in reps], [r["per_block_us"] for r in refs], reps[0]["host"], reps[0]["cpu"], beside


# ---- the GPU node -------------------------------------------------------------------------------------------------------
_W = {}


def _pm_init(d):
    from icmp import test_read
    test_read._pm_init(d)


def _pm_job_packed(args):
    """icmp.test_read._pm_job on packed bits (one residual or detector array for both PM kinds, 612 bytes a shot)."""
    from icmp import test_read
    path, kind, packed, pre, obs, meta = args
    return test_read._pm_job((path, kind, np.unpackbits(packed, axis=1)[:, :M], pre, obs, meta))


class Pool:
    """The PM processes, at most 2 x procs jobs in flight (bounded memory, as r3.py's read)."""

    def __init__(self, procs):
        from multiprocessing import get_context
        self.procs, self.pend = procs, []
        self.pool = get_context("spawn").Pool(procs, initializer=_pm_init, initargs=(D,))

    def submit(self, job):
        self.pend.append(self.pool.apply_async(_pm_job_packed, (job,)))
        while sum(not r.ready() for r in self.pend) > 2 * self.procs:
            time.sleep(0.05)

    def close(self):
        for r in self.pend:
            r.get()
        self.pool.close()
        self.pend = []


def ising_parts(a, rec):
    """[(name, kind, spec)]: S4-R3's 8 TensorRT parts with the bundle's plan paths (same sha256), then torch. A dry
    run reads only the plans present; --standin DIR (dry runs only) puts other plan files in their place, each
    verified against its own hash, to try the TensorRT path on this GPU."""
    P = json.load(open(s4r3_file(a, rec, "data/study4/r3/t/work/r3_parts.json")))
    from icmp.test_read import trt_name, verify_artifacts
    dry, standin = getattr(a, "dry", False), getattr(a, "standin", None)
    assert not standin or dry, "--standin is for dry runs"
    parts = []
    for p in P["parts"]:
        b = os.path.basename(p["plan"])
        assert rec["method"]["bundle_files"]["plans/" + b] == p["plan_sha256"]
        q = dict(p, plan=os.path.join(standin or os.path.join(a.bundle, "plans"), b))
        if standin and os.path.exists(q["plan"]):
            q.update(plan_sha256=sha_file(q["plan"]), standin=True)
        if os.path.exists(q["plan"]) or not dry:
            parts.append((trt_name(q), "trt", q))
    assert dry or [n for n, _, _ in parts] == rec["parts"]["trt"]
    verify_artifacts(parts)
    if any(k == "trt" for _, k, _ in parts):
        import tensorrt
        assert tensorrt.__version__ == rec["method"]["versions_gpu"]["tensorrt"], tensorrt.__version__
    return parts + [(n, "torch", n[:2]) for n in rec["parts"]["torch"]]


def gpu_info():
    import torch
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,mig.mode.current",
                        "--format=csv,noheader"], capture_output=True, text=True, timeout=60).stdout.strip()
    return dict(name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()), smi=q)


def read_ising(parts, chunks, load_chunk, meta_of, root, pool, label):
    """Every part on the given chunks: its pre-decoder once per chunk, its residual packed once and decoded by plain
    and by correlated PM (rows n and n_corr); a chunk whose files already count for this read is skipped."""
    from icmp import test_read
    for n, k, p in parts:
        todo = [j for j in chunks if not all(test_read._reusable(os.path.join(root, x, f"c{j:03d}.npz"), meta_of(x, j))
                                             for x in (n, n + "_corr"))]
        if not todo:
            continue
        fn = test_read.trt_fn(p, M) if k == "trt" else test_read.torch_fn(p, D)
        t0 = time.time()
        for j in todo:
            det, obs = load_chunk(j)
            pre, res = fn(det)
            rp = np.packbits(res, axis=1)
            for x, kind in ((n, "plain"), (n + "_corr", "corr")):
                pool.submit((os.path.join(root, x, f"c{j:03d}.npz"), kind, rp, pre, obs, meta_of(x, j)))
        del fn
        print(f"{label} {n}: {len(todo)} chunks to PM ({time.time() - t0:.0f} s on the GPU)", flush=True)


def cmd_hopper(a):
    """Before any T6 read, on the GPU that reads T6 (the same job): every Ising part on T5 chunks 0-19 (200,000 shots),
    + plain and + correlated PM, against S4-R3's decisions of the same part from PB's H100 (the part digests of
    reads_ising.json), and M1 / M4 torch on the first 10,240 V17b shots, the first 10,000 against PB's residual files
    (V17b chunk 0). A part passes only with every
    decision equal (and, for torch, every residual and frame bit). Exit 5 if any part fails; hopper.json lists them."""
    from icmp import s4guard, t4, test_analyze
    from speed2.s19 import vsets
    fz, rec = load(a)
    t4.frozen_checks(fz, {D: [0]})
    reg = json.load(open(s4r3_file(a, rec, "data/study4/r3/pb/reads_ising.json")))["parts"]
    t5 = json.load(open(s4r3_file(a, rec, "data/study4/r3/t5_manifest.json")))
    parts = ising_parts(a, rec)                          # a dry run: torch and the plans present (stand-ins)
    chunks = parse_chunks(a.chunks)
    assert chunks == list(range(len(chunks))), "a prefix of T5's chunks"
    names = [x for n, _, _ in parts for x in (n, n + "_corr")]
    for x in names:                                      # S4-R3's chunk files are the registered ones
        assert test_analyze.part_digest(os.path.join(a.reg, "d17", x)) == reg[x], f"{x}: not S4-R3's read"
    out = os.path.join(a.out, "hopper")
    pool = Pool(a.procs)

    def load_chunk(j):
        return t4.load(t5, D, [j], a.t5)

    def meta_of(x, j):
        return dict(revision="R17 hopper", part=x, chunk=j, t5_manifest_sha256=rec["s4r3"]["files"][
            "data/study4/r3/t5_manifest.json"])

    read_ising(parts, chunks, load_chunk, meta_of, out, pool, "hopper")
    pool.close()
    rows = {}
    for x in names:
        mine, o1 = test_analyze.load_part(os.path.join(out, x), len(chunks))
        theirs, o2 = test_analyze.load_part(os.path.join(a.reg, "d17", x), 100)
        n = len(chunks) * CHUNK
        assert np.array_equal(o1, o2[:n]), f"{x}: T5's observables differ"
        rows[x] = dict(shots=n, differ=int((mine != theirs[:n]).sum()), failures_here=int((mine != o1).sum()),
                       failures_registered=int((theirs[:n] != o2[:n]).sum()))
        rows[x]["equal"] = rows[x]["differ"] == 0
    res = {}
    from icmp import ising
    det = vsets.load(D, vsets.VAL[D], range(2))[0][:20 * 512]   # PB's val step's first 20 batches of 512 (it forwarded
    #                                                    all of V17b in one call); the first 10,000 shots are compared
    for mdl in ("M1", "M4"):
        z = np.load(s4r3_file(a, rec, f"data/study4/r3/pb/res_d17_{mdl}_v17b000.npz"))
        pipe, _, _ = ising.load(mdl, D, "cuda")
        pre, r = ising.predecode(pipe, det)
        del pipe
        res[mdl] = dict(residual_equal=bool(np.array_equal(np.packbits(r[:CHUNK], axis=1), z["res"])),
                        frame_equal=bool(np.array_equal(pre[:CHUNK].reshape(-1), np.asarray(z["pre"]).reshape(-1))))
    s4guard.check_runtime(fz, fz["versions"])
    passed = all(r["equal"] for r in rows.values()) and all(all(v.values()) for v in res.values()) and not a.dry
    dump(dict(stage="R17 hopper check: S4-R3's H100 outputs reproduced on this GPU", gpu=gpu_info(), chunks=chunks,
              rows=rows, torch_residuals_v17b000=res, passed=passed, dry=a.dry,
              not_read=sorted(set(rec["parts"]["trt"]) - {n for n, k, _ in parts}),
              standin={n: p["plan_sha256"] for n, k, p in parts if k == "trt" and p.get("standin")}, created=now()),
         os.path.join(a.out, "hopper.json"))
    for x, r in rows.items():
        print(f"hopper {x}: {r['differ']} of {r['shots']} decisions differ from S4-R3's H100 read", flush=True)
    print(f"hopper torch residuals (V17b chunk 0): {res}; PASSED {passed}", flush=True)
    return 0 if passed or a.dry else 5                   # a dry run records the check; it never passes or gates


def read_ours(fz, rec, man, a, ident, chunks):
    """Ours through the frozen executable at the registered tau (r3.py's read_ours with T6): the val replay on V17b
    chunk 0 first, every per-chunk check, c, routing and decision of every shot."""
    from speed2.s14.t2ours import certify, do_chunk, reusable
    from speed2.s19.t4ours import PART, build4
    assert PART == OURS
    dep, ex, tau, m, info = build4(fz, D, a.root, a.lib, a.devrec)
    assert repr(tau) == rec["method"]["tau_registered"], "the executable routes at the registered tau"
    metas = {}
    for j in chunks:
        metas[j] = dict(ident, d=D, part=OURS, executable=info["executable"], chunk=j,
                        input_sha256=man["files"][name(man["split"], j)]["sha256"])
        path = os.path.join(a.out, f"d{D}", OURS, f"c{j:03d}.npz")
        if reusable(path, metas[j]):
            continue
        det, obs = t6_load(man, [j], a.shots)
        sha, _, names = do_chunk(ex, det, obs, path, metas[j], tau, m)
        print(f"ours chunk {j}: {len(names)} checks passed; {sha[:16]}", flush=True)
    certify(a.out, D, chunks, metas)
    return info


def cmd_read(a):
    from icmp import s4guard, t4, test_analyze
    fz, rec = load(a)
    man, ident = t6(a, rec)
    t4.frozen_checks(fz, {D: [0]})
    chunks = chunks_of(man)
    gi = gpu_info()
    hop = json.load(open(a.hopper))
    h100 = gi["name"] == "NVIDIA H100 80GB HBM3"
    assert h100 or (hop["passed"] and hop["gpu"]["smi"] == gi["smi"]) or (ident["dry"] and hop.get("dry")), \
        "an H100, or this very GPU after a passed hopper check"
    out = dict(stage="R17 reads on T6", start=now(), ident=ident, gpu=gi, hopper=sha_file(a.hopper), parts={})
    if "ours" in a.what:
        out["ours"] = read_ours(fz, rec, man, a, ident, chunks)
    pool = Pool(a.procs)

    def meta_of(x, j):
        return dict(ident, d=D, part=x, chunk=j, input_sha256=man["files"][name(man["split"], j)]["sha256"])

    root = os.path.join(a.out, f"d{D}")
    names = []
    if "pm" in a.what:
        from icmp import test_read
        names += ["pm", "pm_corr"]
        for j in chunks:
            todo = [(x, k) for x, k in (("pm", "plain"), ("pm_corr", "corr"))
                    if not test_read._reusable(os.path.join(root, x, f"c{j:03d}.npz"), meta_of(x, j))]
            if todo:
                det, obs = t6_load(man, [j], a.shots, packed=True)
                for x, k in todo:
                    pool.submit((os.path.join(root, x, f"c{j:03d}.npz"), k, det, None, obs, meta_of(x, j)))
    if "ising" in a.what:
        assert ident["dry"] == bool(getattr(a, "dry", False)), "--dry with the dry stand-in only"
        parts = ising_parts(a, rec)
        ok = {x for x, r in hop.get("rows", {}).items() if r["equal"]}
        assert h100 or ident["dry"] or all(n in ok and n + "_corr" in ok for n, _, _ in parts), \
            "a part without a passed hopper row on this GPU"
        names += [x for n, _, _ in parts for x in (n, n + "_corr")]
        read_ising(parts, chunks, lambda j: t6_load(man, [j], a.shots), meta_of, root, pool, "T6")
    pool.close()
    s4guard.check_runtime(fz, fz["versions"])
    from icmp import test_read
    for x in names:                                      # exactly T6's chunks, each written for this read
        assert all(test_read._reusable(os.path.join(root, x, f"c{j:03d}.npz"), meta_of(x, j)) for j in chunks), x
        out["parts"][x] = test_analyze.part_digest(os.path.join(root, x))
    out["end"] = now()
    dump(out, os.path.join(a.out, f"reads_{'_'.join(a.what)}.json"))
    print(f"reads done: {len(out['parts'])} parts", flush=True)


# ---- the A100 node -------------------------------------------------------------------------------------------------------
def cmd_bposd(a):
    """Both arms (or --arms) on T6's BP-OSD chunks (0-19) or a share of them (--chunks), study 4's decoder and settings
    (icmp.s4_bposd: check_settings, the hash-checked circuit, decode_parallel4: --procs workers, worker i on GPU
    i mod MINLLR_GPUS with its own 1/procs of the job's cores, bp_batch_size 2048). --limit (dry runs only) decodes
    the first N shots, to measure the rate. Before any shot is read, and again before each output, the frozen runtime
    check (icmp.s4guard.check_runtime: every loaded file of the frozen tree as the freeze pins it) runs with what the
    decoder workers import and every package of r17.json's versions_bposd loaded, at those versions; each worker runs
    the freeze's own worker check (s4guard.worker_runtime, route.bposd.versions). The sidecar records the versions
    found. T6's files are not hashed in advance: t6_load hashes every chunk it reads."""
    import importlib
    from icmp import s4_bposd, s4guard
    fz, rec = load(a)
    pins = rec["method"]["versions_bposd"]
    for m in ("icmp.minllr", *pins):                     # the workers' imports (icmp.minllr, cudaq_qec) and every pin
        importlib.import_module(m)
    checked = s4guard.check_runtime(fz, pins)
    env = dict(python=sys.version.split()[0], **{m: sys.modules[m].__version__ for m in pins})
    man, ident = t6(a, rec, rehash=False)
    s4_bposd.check_settings(fz)
    sha = fz["distances"][str(D)]["circuit_sha256"]
    c = s4_bposd.circuit(D, sha)
    sub = [j for j in chunks_of(man) if j in parse_chunks(rec["test_set"]["bposd_chunks"])]
    chunks = parse_chunks(a.chunks) if a.chunks else sub
    assert set(chunks) <= set(sub), f"chunks {a.chunks} are not in the BP-OSD subset"
    assert not a.limit or ident["dry"], "--limit is for dry runs"
    tag = "" if chunks == sub else f"_c{chunks[0]:03d}-{chunks[-1]:03d}"
    os.makedirs(a.out, exist_ok=True)
    for arm in a.arms:
        out = os.path.join(a.out, f"r17bposd_d{D}_{arm}{tag}.npz")
        assert not os.path.exists(out), f"{out} exists: one read per arm"
        det, obs = t6_load(man, chunks, a.shots, packed=True)
        det = np.unpackbits(det, axis=1)[:, :c.num_detectors]
        if a.limit:
            det, obs = det[:a.limit], obs[:a.limit]
        t0 = time.time()
        pred = s4_bposd.decode_parallel4(D, arm, det, a.procs, sha, os.path.abspath(a.freeze)).astype(np.uint8)
        wall = time.time() - t0
        assert pred.shape == obs.shape == (len(det),)
        s4guard.check_runtime(fz, pins)
        np.savez_compressed(out, pred=pred, obs=obs)
        dump(dict(ident, d=D, part=f"bposd_{arm}", chunks=f"{chunks[0]}-{chunks[-1]}", shots=len(obs),
                  input_sha256=hashlib.sha256("".join(man["files"][name(man["split"], j)]["sha256"]
                                                      for j in chunks).encode()).hexdigest(),
                  output_sha256=sha_file(out), written=now(), wall_s=round(wall), procs=a.procs,
                  gpus=os.environ.get("MINLLR_GPUS"), shots_per_s=len(obs) / wall, failures=int((pred != obs).sum()),
                  host=os.uname().nodename, cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), versions=env,
                  runtime_check=dict(files=checked, pins=pins, worker_check="s4guard.worker_runtime(freeze, 'bposd')")),
             out[:-4] + ".prov.json", "x")
        print(f"bposd {arm} chunks {chunks[0]}-{chunks[-1]}: {len(obs)} shots, {wall:.0f} s, "
              f"{len(obs) / wall:.2f} shots/s with {a.procs} workers", flush=True)


def cmd_bposd_merge(a):
    """The shares of each arm joined in chunk order: exactly the subset's chunks, each once, T6's observables (a dry
    run's shares, cut by --limit, against the first shots only). An untagged file is one share of the whole subset."""
    fz, rec = load(a)
    man, ident = t6(a, rec)
    sub = [j for j in chunks_of(man) if j in parse_chunks(rec["test_set"]["bposd_chunks"])]
    obs = t6_load(man, sub, a.shots, packed=True)[1]
    for arm in ARMS:
        parts = []
        for f in glob.glob(os.path.join(a.dir, f"r17bposd_d{D}_{arm}*.npz")):
            tag = os.path.basename(f)[:-4][len(f"r17bposd_d{D}_{arm}"):]      # "" (the whole subset) or _cLO-HI
            lo, hi = (int(x) for x in tag[2:].split("-")) if tag else (sub[0], sub[-1])
            prov = json.load(open(f[:-4] + ".prov.json"))
            assert prov["output_sha256"] == sha_file(f) and prov["t6_manifest_sha256"] == ident["t6_manifest_sha256"], f
            parts.append((lo, hi, f, prov))
        parts.sort()
        got = [j for lo, hi, _, _ in parts for j in range(lo, hi + 1)]
        assert got == sub, f"{arm}: the shares cover chunks {got}, not {sub} once each"
        pred = np.concatenate([np.load(f)["pred"] for _, _, f, _ in parts]).astype(np.uint8)
        o = np.concatenate([np.load(f)["obs"] for _, _, f, _ in parts]).astype(np.uint8)
        assert np.array_equal(o, obs[:len(o)] if ident["dry"] else obs), f"{arm}: the shares' observables differ from T6's"
        out = os.path.join(a.out, f"r17bposd_d{D}_{arm}.npz")
        np.savez_compressed(out, pred=pred, obs=o)
        dump(dict(ident, d=D, part=f"bposd_{arm}", chunks=f"{sub[0]}-{sub[-1]}", written=now(),
                  output_sha256=sha_file(out), shares={os.path.basename(f): dict(sha256=p["output_sha256"],
                  host=p.get("host"), wall_s=p.get("wall_s"), versions=p.get("versions")) for _, _, f, p in parts}),
             out[:-4] + ".prov.json", "x")
        print(f"bposd {arm}: {len(parts)} shares joined, {len(pred)} shots", flush=True)


# ---- the analysis -------------------------------------------------------------------------------------------------------
def inputs_of(a):
    """{name: path} of every analysis input: each file under the shots, predictions and BP-OSD directories (named
    relative to the directory's parent) and the four records by their role."""
    files = {}
    for d in (a.shots, a.preds, a.bposd):
        d = d.rstrip("/")
        for p in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)):
            if os.path.isfile(p):
                files[os.path.relpath(p, os.path.dirname(d))] = p
    files.update(manifest=a.manifest, cpu_term=a.cpu_term, hopper=a.hopper, record=a.record)
    return files


def cmd_inventory(a):
    """The pre-commit: `sha256  path` of every input (relative to IN; records by their role) and the command."""
    files = inputs_of(a)
    lines = [f"{sha_file(p)}  {r}" for r, p in sorted(files.items())]
    with open(a.out, "x") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"inventory: {len(lines)} files; {sha_file(a.out)}", flush=True)


def speed(fz, rec, a, cpu):
    """Ours = (a) + (b) + (c) and the opponents' registered times (r3.py's speed(), the same frozen arithmetic)."""
    A = json.load(open(s4r3_file(a, rec, "data/study4/r3/t/work/s4/phase_a.json")))
    sh = json.load(open(s4r3_file(a, rec, "data/study4/r3/pmshare_r3.json")))
    num = fz["route"]["speed"]["grace_numerators_us"]
    share = fz["route"]["speed"]["pm_share_grace_us"]
    v = A["cands"][PREFIX]
    sp, verdict = rec["speed"], rec["engine"]["verdict"]
    if verdict == "GO":
        g = sp["a_gpu"]["go"]
        a_us, a_src = float(g["us"]), g
    else:
        a_us, a_src = float(v["a_counted"]), sp["a_gpu"]["no_go"]
        assert a_us == sp["a_gpu"]["no_go"]["us"]
    c_us = float(sp["c_handoff"]["e3"]["us"]) if sp["c_handoff"].get("e3") else float(v["c_handoff"])
    terms, refs, host, model, beside = b_term(cpu)
    ratio = num["correlated"][str(D)] / float(np.median(refs))
    assert num["correlated"][str(D)] == sp["grace_correlated_d17_us"]
    b_us = max(terms) * ratio
    ours = dict(engine=("eng17" if verdict == "GO" else "FROZEN"), a_gpu=a_us, a_source=a_src, b_grace=b_us,
                b_node_reps=terms, ref_node_reps=refs, ratio=ratio, node=host, cpu=model, c_handoff=c_us,
                total=a_us + b_us + c_us, b_grace_registered_tau_beside=max(beside) * ratio if beside else None,
                b_node_reps_registered_tau_beside=beside)
    opp = {"pm": num["plain"][str(D)], "pm_corr": num["correlated"][str(D)]}
    for mdl in ("M1", "M4"):
        r = A["ising"][f"{D} {mdl}"]
        assert abs(r["total_grace"] - r["plan_us"] - share[mdl][str(D)]) < 1e-6, f"{mdl}: not the frozen PM share"
        opp[mdl] = r["total_grace"]
        opp[f"{mdl}_corr"] = r["plan_us"] + sh["share"][mdl]["grace_us"]
    for arm in ARMS:
        opp[arm] = A["bposd"][f"{D} {arm}"]["fastest_us"]
    return ours, opp


def cmd_analyze(a):
    from icmp import stats2, test_analyze
    from icmp.s4_analyze import check_ours
    fz, rec = load(a)
    man, ident = t6(a, rec)
    dry = ident["dry"]
    alpha = fz["route"]["alpha"]
    assert alpha == rec["accuracy"]["alpha"] == 0.025
    verdict = rec["engine"]["verdict"]
    assert verdict in ("GO", "NO-GO") or dry, "the E3 verdict is not in r17.json yet"
    if not dry or a.inventory:                           # the pre-commit: every input as inventoried
        want = dict(ln.split("  ", 1)[::-1] for ln in open(a.inventory).read().splitlines())
        have = {r: sha_file(p) for r, p in inputs_of(a).items()}
        assert have == want, f"the inputs differ from the inventory: {sorted(set(have.items()) ^ set(want.items()))[:5]}"
    os.makedirs(a.out)                                   # the analysis runs once into a new directory
    chunks = chunks_of(man)
    obs = t6_load(man, chunks, a.shots, packed=True)[1]
    parts = rec["parts"]["trt"] + rec["parts"]["torch"]
    want = sorted([OURS, "pm", "pm_corr"] + parts + [n + "_corr" for n in parts])
    have = sorted(os.listdir(os.path.join(a.preds, f"d{D}")))
    if dry:                                              # stand-in plans are not read in a dry run
        want = [x for x in want if x in have]
    assert have == want, f"the parts differ: missing {sorted(set(want) - set(have))}, extra {sorted(set(have) - set(want))}"
    pr, din = {}, {}
    for n in want:
        pdir = os.path.join(a.preds, f"d{D}", n)
        pr[n], o = test_analyze.load_part(pdir, len(chunks))
        assert np.array_equal(o, obs), f"{n}: observables differ from T6's"
        pr[n] = pr[n].astype(np.uint8)
        din[n] = test_analyze.part_digest(pdir)
    c, _ = test_analyze.load_part(os.path.join(a.preds, f"d{D}", OURS), len(chunks), key="c")
    routed, _ = test_analyze.load_part(os.path.join(a.preds, f"d{D}", OURS), len(chunks), key="routed")
    tau_reg, tau = float(rec["method"]["tau_registered"]), float(rec["method"]["tau"])
    frac10 = check_ours(pr[OURS], c, routed.astype(np.uint8), pr["pm_corr"], tau_reg)   # as executed
    r8 = (np.abs(c.astype(np.float64)) < tau)
    assert not (r8 & ~routed.astype(bool)).any(), "a shot routed at q = 8 % is not routed at the registered tau"
    q8 = np.where(r8, pr["pm_corr"], (c > 0).astype(np.uint8)).astype(np.uint8)
    frac8 = check_ours(q8, c, r8.astype(np.uint8), pr["pm_corr"], tau)
    q10 = pr.pop(OURS)
    nb = None
    for arm in ARMS:
        f = os.path.join(a.bposd, f"r17bposd_d{D}_{arm}.npz")
        z = np.load(f)
        nb = len(z["pred"]) if nb is None else nb
        assert len(z["pred"]) == nb and np.array_equal(z["obs"], obs[:nb]), f"{f}: not T6's BP-OSD subset"
        assert dry or nb == len(parse_chunks(rec["test_set"]["bposd_chunks"])) * CHUNK
        pr[f"bposd_{arm}"] = z["pred"].astype(np.uint8)
        din[os.path.basename(f)] = sha_file(f)
    fail = {n: pr[n] != (obs[:nb] if n.startswith("bposd_") else obs) for n in pr}
    f8, f10 = q8 != obs, q10 != obs
    rows, cells = [], {o: {} for o in OPPONENTS}
    for n in pr:
        k = nb if n.startswith("bposd_") else len(obs)
        r = stats2.paired(f8[:k], fail[n], alpha=alpha)
        v = "better" if r["verdict"] == "better" else "not shown"
        rows.append({**dict(d=D, part="ours_q08", vs=n, opponent=opponent_of(n)), **r, "cell_verdict": v})
        cells[opponent_of(n)][n] = dict(verdict=v, p=r["mcnemar_p"], diff=r["diff_ler"], lo=r["diff_lo"],
                                        hi=r["diff_hi"], ours=r["fail_a"], other=r["fail_b"], phi=r["phi"], shots=r["shots"])
    assert all(cells[o] for o in OPPONENTS), [o for o in OPPONENTS if not cells[o]]
    pres = stats2.paired(f8, f10, alpha=alpha)
    delta = rec["preservation"]["delta"]
    s4 = json.load(open(s4r3_file(a, rec, "data/study4/r3/final/out/r3_summary.json")))
    assert delta == -s4["cells"]["pm_corr"]["accuracy"]["rows"]["pm_corr"]["hi"] / 2, "delta is not study 4's half margin"
    pres.update(delta=delta, preserved=bool(pres["diff_hi"] < delta))
    rows.append({**dict(d=D, part="ours_q08", vs="ours_q10", opponent="preservation"), **pres,
                 "cell_verdict": "preserved" if pres["preserved"] else "not shown"})
    cpu = json.load(open(a.cpu_term))
    assert dry or not cpu.get("dry"), "a dry CPU record"
    ours, opp = speed(fz, rec, a, cpu)
    acc = {o: dict(rows=cells[o], holds=all(x["verdict"] == "better" for x in cells[o].values())) for o in OPPONENTS}
    spd = {o: dict(ours_us=ours["total"], opponent_us=opp[o], margin_us=opp[o] - ours["total"],
                   holds=bool(ours["total"] < opp[o])) for o in OPPONENTS}
    claim = all(acc[o]["holds"] and spd[o]["holds"] for o in OPPONENTS)
    inputs = dict(din, t6_manifest=ident["t6_manifest_sha256"], cpu_term=sha_file(a.cpu_term), hopper=sha_file(a.hopper),
                  record=sha_file(a.record), inventory=sha_file(a.inventory) if a.inventory else None)
    summary = dict(revision="R17", study="4", d=D, dry=dry, alpha=alpha, engine_verdict=verdict,
                   cells={o: dict(accuracy=acc[o], speed=spd[o]) for o in OPPONENTS}, ours_speed=ours,
                   claim_all_cells_hold=claim, preservation=pres, paper_row=claim and pres["preserved"],
                   failures=dict({n: int(f.sum()) for n, f in fail.items()}, ours_q08=int(f8.sum()), ours_q10=int(f10.sum())),
                   shots=dict(t6=len(obs), bposd_subset=nb, ours_q08_on_bposd_subset=int(f8[:nb].sum())),
                   routed_fraction=dict(q08=frac8, q10=frac10), tau=dict(q08=rec["method"]["tau"],
                   q10=rec["method"]["tau_registered"]), inputs=inputs, created=now(),
                   statistics=rec["accuracy"]["statistics"] + "; " + rec["preservation"]["rule"])
    dump(summary, os.path.join(a.out, "r17_summary.json"), "x")
    lead = ("d", "part", "vs", "opponent", "shots", "fail_a", "fail_b", "cell_verdict")
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in lead, lead.index(k) if k in lead else 0, k))
    with open(os.path.join(a.out, "r17_eval.csv"), "x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for o in OPPONENTS:
        print(f"{o}: accuracy {'holds' if acc[o]['holds'] else 'NOT shown'} ({len(cells[o])} rows); speed "
              f"{ours['total']:.2f} vs {opp[o]:.2f} us: {'holds' if spd[o]['holds'] else 'NOT shown'}", flush=True)
    print(f"preservation: q 8 % {int(f8.sum())} vs q 10 % {int(f10.sum())} failures in {len(obs)} shots; difference "
          f"{pres['diff_ler'] * 1e6:+.1f} per million, 97.5 % [{pres['diff_lo'] * 1e6:+.1f}, {pres['diff_hi'] * 1e6:+.1f}], "
          f"margin {delta * 1e6:.1f}: {'PRESERVED' if pres['preserved'] else 'NOT shown'}", flush=True)
    print(f"R17{' DRY' if dry else ''} claim (all {2 * len(OPPONENTS)} cells): {'HOLDS' if claim else 'NOT SHOWN'}; "
          f"ours {ours['total']:.2f} us = {ours['a_gpu']:.2f} + {ours['b_grace']:.2f} + {ours['c_handoff']:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name_, *args):
        s = sub.add_parser(name_)
        for x in args:
            s.add_argument(f"--{x}", required=True)
        return s

    common = ("freeze", "record")
    s = p("pins", *common, "bundle", "s4r3")
    s.add_argument("--out", default="")
    s.add_argument("--allow-missing", action="store_true", help="dry runs: report missing files (the plans)")
    s = p("sample", *common, "out")
    s.add_argument("--dry", action="store_true")
    p("t5regen", *common, "s4r3", "chunks", "out")
    p("mask", *common, "devrec", "out")
    s = p("cputerm", *common, "devrec", "mask", "out")
    s.add_argument("--which", default="q08", choices=("q08", "q10"))
    s.add_argument("--rep", type=int, required=True)
    s.add_argument("--core", type=int, required=True)
    s.add_argument("--dry", action="store_true")
    s = p("cpurecord", "dir")
    s.add_argument("--dry", action="store_true")
    s = p("hopper", *common, "bundle", "s4r3", "reg", "t5", "out")
    s.add_argument("--chunks", default="0-19")
    s.add_argument("--procs", type=int, default=8)
    s.add_argument("--dry", action="store_true")
    s.add_argument("--standin", default=None, help="dry runs: a directory of other plan files tried in their place")
    t6a = ("manifest", "shots")
    s = p("read", *common, *t6a, "bundle", "s4r3", "hopper", "out")
    s.add_argument("--what", nargs="+", default=["ours", "pm", "ising"], choices=["ours", "pm", "ising"])
    s.add_argument("--root")
    s.add_argument("--lib")
    s.add_argument("--devrec")
    s.add_argument("--procs", type=int, default=8)
    s.add_argument("--dry", action="store_true")
    s.add_argument("--standin", default=None)
    s = p("bposd", *common, *t6a, "out")
    s.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    s.add_argument("--chunks", default=None, help="a share of the BP-OSD subset, e.g. 0-4")
    s.add_argument("--procs", type=int, default=5)
    s.add_argument("--limit", type=int, default=0)
    p("bposd-merge", *common, *t6a, "dir", "out")
    ana = (*common, *t6a, "s4r3", "preds", "bposd", "cpu-term", "hopper")
    s = p("inventory", *ana, "out")
    s = p("analyze", *ana, "out")
    s.add_argument("--inventory", default="")
    a = ap.parse_args()
    if a.cmd == "read":
        assert "ours" not in a.what or (a.root and a.lib and a.devrec), "--root, --lib and --devrec for ours"
    rc = globals()["cmd_" + a.cmd.replace("-", "_")](a)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
