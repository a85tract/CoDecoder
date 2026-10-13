"""Dry runs only (never a result): stand-ins for what a CPU job on Alpine cannot read, so that the q rule, the freeze
and the analysis run end to end on the dry set before the GPU dry pass (E0) replaces them with real reads.

Every stand-in is labelled (standin true in its record) and the analysis accepts it only with --dry. Real reads come
first (the fp32 reference of every network with released fp32 weights, A1's S-V seed 1 exported as d9_b3_sv2s1, plain
and correlated PM); the stand-ins derive from them with a fixed seed:
  networks   a new network at d = 7, 11, 13, 15 (not trained yet): the deployed network's c x 1.05 plus Gaussian noise
             (sd 0.4); both d = 17 networks (no fp32 module in the release): +-|N(3, 1.5)| signed by correlated PM's
             answer, the sign flipped on 2 % (deployed) or 1.5 % (new) of the shots; each with a placeholder weight file
             and export record, so that the freeze can name it
  Ising      plain PM's answers (+ PM) or correlated PM's (+ correlated PM) with 0.3 % (M1) or 0.2 % (M4) flipped
  BP-OSD     correlated PM's answers with 0.4 % (min-LLR) or 0.6 % (BP 60) flipped, on the BP-OSD subset
  timing     a session (attempts.json, five GPU-term processes and the handoffs per network, one rejected process at
             d = 11, both GPU checks) and a CPU term (three repetitions per network, three refs per d), with plausible
             numbers
    python -m surf_v2.dry cal --reads READS --exports EXP              (after the real calibration-chunk reads)
    python -m surf_v2.dry test --reads READS --freeze F --session S --cpu C   (after the freeze and the real test reads)"""
import argparse
import json
import os

import numpy as np

from convlines import decoder, provenance, testset
from convlines.evaluate import KEYS, sha_arr

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "config.json")))
DRY = CFG["dry"]


def find(reads, set_, d, key, chunks):
    stem = os.path.join(reads, f"{set_}_d{d}_{key}_{chunks}")
    return (dict(np.load(stem + ".npz")), json.load(open(stem + ".json"))) if os.path.exists(stem + ".npz") else (None, None)


def write(reads, set_, d, chunks, key, arrays, meta):
    stem = os.path.join(reads, f"{set_}_d{d}_{key}_{chunks}")
    testset.write_npz(stem + ".npz", arrays)
    man = testset.sha256_bytes(open(os.path.join(testset.ROOT, testset.SETS[set_]), "rb").read())
    rec = dict(what="SURF-v2 read (surf_v2.dry stand-in)", set=set_, d=d, chunks=chunks, shots=int(len(arrays["obs"])),
               manifest_sha256=man, decoder=key, **meta, arrays={k: dict(dtype=str(v.dtype), sha256=sha_arr(v))
                                                                 for k, v in sorted(arrays.items())},
               npz=os.path.basename(stem + ".npz"), npz_sha256=testset.sha256_bytes(open(stem + ".npz", "rb").read()),
               code_digest=provenance.code_digest(), release_commit=provenance.commit(), standin=True)
    with open(stem + ".json", "x") as fh:
        json.dump(rec, fh, indent=1)


def flip(x, frac, rng):
    return (x ^ (rng.random(len(x)) < frac)).astype(np.uint8)


def net_c(kind, d, base_c, corr, rng):
    if base_c is not None:
        return (base_c * 1.05 + rng.normal(0, 0.4, len(base_c))).astype(np.float32)
    s = np.where(rng.random(len(corr)) < (0.02 if kind == "deployed" else 0.015), -1.0, 1.0)
    return (s * (2.0 * corr - 1) * np.abs(rng.normal(3, 1.5, len(corr)))).astype(np.float32)


def placeholder(exports, name):
    e = decoder.config(name)
    f = os.path.join(exports, e["file"])
    with open(f, "xb") as fh:
        fh.write(f"SURF-v2 dry stand-in for {name}: not a network\n".encode())
    sha = testset.sha256_bytes(open(f, "rb").read())
    with open(f[:-len(".safetensors")] + ".export.json", "x") as fh:
        json.dump(dict(network=name, sha256=sha, final_ema_sha256="standin", standin=True), fh, indent=1)
    return sha


def standin_nets(reads, set_, chunks, exports, taus=None):
    """The stand-in network reads of one dry split (taus None: the network alone, as a calibration read)."""
    for k in map(str, CFG["distances"]):
        d = int(k)
        corr, _ = find(reads, set_, d, "corr_pm", chunks)
        assert corr is not None, f"d{d}: the real correlated PM read of {set_} {chunks} first"
        for role in ("deployed", "new"):
            name = CFG["networks"][k][role]
            if find(reads, set_, d, name, chunks)[0] is not None:
                continue                                          # a real (fp32) read exists
            rng = np.random.default_rng(1000 * d + (1 if role == "new" else 2) + (7 if taus else 0))
            base = find(reads, set_, d, CFG["networks"][k]["deployed"], chunks)[0] if role == "new" else None
            c = net_c(role, d, base["c"] if base is not None else None, corr["dec"], rng)
            tau = None if taus is None else taus[name]
            r = (np.abs(c) < tau) if tau is not None else np.zeros(len(c), bool)
            dec = np.where(r, corr["dec"], (c > 0).astype(np.uint8)).astype(np.uint8)
            if role == "deployed":
                sha = decoder.weights_sha256(name)
            elif taus is None:                                    # calibration: the placeholder is made here
                sha = placeholder(exports, name)
            else:
                sha = testset.sha256_bytes(open(os.path.join(exports, decoder.config(name)["file"]), "rb").read())
            write(reads, set_, d, chunks, name, dict(obs=corr["obs"], dec=dec, c=c, routed=r.astype(np.uint8)),
                  dict(kind="network", network=name, network_sha256=sha, tau=tau, tau_mode="standin", routing="standin",
                       post="correlated", engine="stand-in", batch_one=False,
                       failures=int((dec != corr["obs"]).sum()), routed=int(r.sum())))


def cal(a):
    standin_nets(a.reads, DRY["set"], DRY["cal_chunks"], a.exports)
    print("stand-in calibration reads written", flush=True)


def test(a):
    fz = json.load(open(a.freeze))
    taus = {n: x["tau"] for n, x in fz["networks"].items()}
    standin_nets(a.reads, DRY["set"], DRY["test_chunks"], a.exports, taus)
    ch, bch = DRY["test_chunks"], DRY["bposd_chunks"]
    for d in CFG["distances"]:
        pm, _ = find(a.reads, DRY["set"], d, "plain_pm", ch)
        corr, _ = find(a.reads, DRY["set"], d, "corr_pm", ch)
        rng = np.random.default_rng(77 + d)
        nb = len(testset.parse_range(bch)) * testset.CHUNK
        for name, (src, frac, sub) in {"Ising M1 + PM": (pm, 0.003, None), "Ising M4 + PM": (pm, 0.002, None),
                                       "Ising M1 + correlated PM": (corr, 0.003, None),
                                       "Ising M4 + correlated PM": (corr, 0.002, None),
                                       "min-LLR BP-OSD": (corr, 0.004, nb), "BP 60 BP-OSD": (corr, 0.006, nb)}.items():
            dec = flip(src["dec"][:sub], frac, rng)
            write(a.reads, DRY["set"], d, bch if sub else ch, KEYS[name], dict(obs=src["obs"][:sub], dec=dec),
                  dict(kind="opponent", opponent=name, failures=int((dec != src["obs"][:sub]).sum())))
    # timing stand-ins: one session and one CPU term
    os.makedirs(a.session)
    os.makedirs(a.cpu)
    rng = np.random.default_rng(5)
    att = {}
    base = {7: 8.4, 9: 27.9, 11: 40.6, 13: 49.2, 15: 50.9, 17: 66.7}
    for name, n in sorted(fz["networks"].items()):
        d, rows, k = n["d"], [], 0
        reject_once = d == 11 and n["role"] == "deployed"
        for kind, count in (("gpu", 5), ("handoff", 5 if d == 7 else 1)):
            got = 0
            while got < count:
                tag = f"p{k}" if kind == "gpu" else f"handoff_p{k}"
                rec = dict(network=name, network_sha256=n["sha256"], code_digest=fz["read_code_digest"], d=d, proc=k,
                           standin=True, accepted=not (reject_once and k == 1))
                if kind == "gpu":
                    rec["per_shot_mean"] = base[d] * (1 + 0.01 * rng.random())
                else:
                    est = 0.6 + 0.8 * rng.random() if d != 7 else 2.2 + 0.05 * rng.random()
                    rec.update(handoff_estimate_us=est if d != 7 else None, handoff_us=max(1.1, est) if d != 7 else est,
                               tau=n["tau"], **({} if d == 7 else dict(routed=int(200000 * n["q"]))))
                f = f"{name}_{tag}.json"
                json.dump(rec, open(os.path.join(a.session, f), "x"), indent=1)
                rows.append(dict(kind=kind, proc=k, file=f, exit=0))
                got += rec["accepted"]
                k += 1
        att[name] = rows
        cost = {7: 35.0, 9: 70.0, 11: 160.0, 13: 290.0, 15: 480.0, 17: 760.0}[d]
        for r in (1, 2, 3):
            json.dump(dict(network=name, tau=n["tau"], term_us=float(cost * max(n["q"], 0.0) * (1 + 0.02 * r)),
                           routed=int(200000 * n["q"]), shots_sha256="standin", mask_kind="standin", slurm_job="standin",
                           code_digest=fz["read_code_digest"], standin=True),
                      open(os.path.join(a.cpu, f"term_{name}_rep{r}.json"), "x"), indent=1)
    for d in CFG["distances"]:
        for r in (1, 2, 3):
            json.dump(dict(d=d, per_block_us=float({7: 37.0, 9: 70.0, 11: 160.0, 13: 290.0, 15: 490.0, 17: 780.0}[d]
                                                   * (1 + 0.01 * r)), slurm_job="standin",
                           code_digest=fz["read_code_digest"], standin=True),
                      open(os.path.join(a.cpu, f"ref_d{d}_rep{r}.json"), "x"), indent=1)
    json.dump(att, open(os.path.join(a.session, "attempts.json"), "x"), indent=1)
    for w in ("start", "end"):
        json.dump(dict(when=w, passed=True, standin=True, time=f"2026-10-12T0{0 if w == 'start' else 1}:00:00+0000",
                       clocks=dict(gpu_uuid="GPU-standin")), open(os.path.join(a.session, f"gpucheck_{w}.json"), "x"), indent=1)
    print("stand-in test reads, timing session and CPU term written", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cal")
    p.add_argument("--reads", required=True)
    p.add_argument("--exports", required=True)
    p = sub.add_parser("test")
    p.add_argument("--reads", required=True)
    p.add_argument("--exports", required=True)
    p.add_argument("--freeze", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--cpu", required=True)
    a = ap.parse_args()
    {"cal": cal, "test": test}[a.cmd](a)


if __name__ == "__main__":
    main()
