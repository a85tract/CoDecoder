"""Accuracy on a test set: ours (convlines.decoder) against opponents on the same shots, chunk by chunk, then the
registered paired statistics (convlines.stats, alpha = 0.025) of ours against each opponent.

    python -m convlines.evaluate --set T2 --d 9 --chunks 0-99 --shots DIR [--opponents "plain PM" "correlated PM"]
        [--routing fp32 | deployed | published] [--post correlated | plain] [--build build/d7] [--device cuda]
        [--out result.json]

Ours' c and routing (--routing):
  fp32       the PyTorch reference (the trained fp32 network) and |c| < tau; runs on a CPU. Its c differs from the
             deployed executables' by rounding, so a shot whose |c| lies within rounding of tau can be routed
             differently and, rarely, decided differently.
  deployed   the deployed executable's c (timing.deployed.Logits: the Triton engine at d = 9-17 on a CUDA GPU, the
             resident kernel of timing/d7 at d = 7 on an sm_90 GPU after timing/d7/build.sh) and |c| < tau: the
             registered decisions, as the studies' test read made them. The only choice at d = 17, whose fp32 network
             is not in this release.
  published  the studies' routing mask of the test set (routing/), the reference's sign on every other shot; runs on a
             CPU and reproduces the registered decisions unless an unrouted shot's fp32 sign differs from the deployed
             one.
Each chunk also reports how many shots its routing sends differently from the published mask.
--post plain is the reverse post hoc control (README, "The two post hoc controls"): ours sends |c| < tau to plain PM
instead of correlated PM, with tau refitted for plain PM (decoder.TAU_PLAIN); it needs --routing deployed or fp32.

The registered reads: every T2 / T3 / T5 shot for ours, Ising, plain and correlated PM; the two BP-OSD arms on all of
T2 and on the first 20 chunks of T3 and T5 (run them with --chunks 0-19 at d = 15 and 17). The post hoc controls "Ising
M1 + correlated PM" and "Ising M4 + correlated PM" read every T2 / T3 shot (README, "The two post hoc controls"); at
d = 17 study 4 registered the same two decoders and read every T5 shot. DIR holds the chunk files of
`python -m convlines.testset regen`; each is checked against the manifest's sha256 before it is decoded. --out also
records the sha256 of each opponent's decisions (uint8, in chunk order) and of the observables.

SURF-v2 (surf_v2/REGISTRATION.md) reads its sets sv2cal / sv2dry / sv2test (convlines.testset) with three options:
  --net NAME   ours is another network of d: a SURF-v2 network of weights/configs_sv2.json (its weights checked against
               the frozen sha256 once the freeze has set it), or a registered one; --net none reads the opponents only
  --tau T      ours' tau: config (the network's policy, the default), none (no routing: the network alone decides
               every shot, as the calibration reads need), or a float
  --save DIR   every decoder's per-shot arrays, written once (never over a file) in testset.write_npz's format, beside
               a record: DIR/<set>_d<d>_<key>_<chunks>.npz with obs and dec (uint8; ours also c, float32, and routed,
               uint8) and .json (set, chunks, the manifest's sha256, the decoder and its settings, the network's
               weights sha256 and tau, the routing and engine, every array's sha256, the read path's code digest,
               the commit, host and versions); keys: the network's name, plain_pm, corr_pm, ising_m1_pm, ising_m4_pm,
               ising_m1_corr, ising_m4_corr, bposd_minllr, bposd_bp60
    python -m convlines.evaluate --set sv2cal --d 11 --chunks 0-19 --shots DIR --net d11_b5_sv2 --tau none
        --routing deployed --opponents --save READS
    python -m convlines.evaluate --set sv2test --d 11 --chunks 0-99 --shots DIR --net none
        --opponents "Ising M1 + PM" "Ising M4 + PM" "Ising M1 + correlated PM" "Ising M4 + correlated PM" --save READS"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

from convlines import circuits, decoder, opponents, stats, testset

try:                                                   # the CUDA-Q QEC venv (BP-OSD) has no torch: opponents only
    import torch
except ImportError:
    torch = None

KEYS = {"plain PM": "plain_pm", "correlated PM": "corr_pm", "Ising M1 + PM": "ising_m1_pm", "Ising M4 + PM": "ising_m4_pm",
        "Ising M1 + correlated PM": "ising_m1_corr", "Ising M4 + correlated PM": "ising_m4_corr",
        "min-LLR BP-OSD": "bposd_minllr", "BP 60 BP-OSD": "bposd_bp60"}


def sha_arr(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def save(a, key, arrays, meta):
    """One decoder's per-shot arrays and their record into a.save, each written once."""
    from convlines import provenance
    os.makedirs(a.save, exist_ok=True)
    stem = os.path.join(a.save, f"{a.set}_d{a.d}_{key}_{a.chunks.replace(',', '_')}")
    testset.write_npz(stem + ".npz", arrays)
    rec = dict(what="SURF-v2 read (convlines.evaluate --save)", set=a.set, d=a.d, chunks=a.chunks,
               shots=int(len(arrays["obs"])), manifest_sha256=testset.sha256_bytes(open(os.path.join(
                   testset.ROOT, testset.SETS[a.set]), "rb").read()), decoder=key, **meta,
               arrays={k: dict(dtype=str(v.dtype), sha256=sha_arr(v)) for k, v in sorted(arrays.items())},
               npz=os.path.basename(stem + ".npz"), npz_sha256=testset.sha256_bytes(open(stem + ".npz", "rb").read()),
               code_digest=provenance.code_digest(), release_commit=provenance.commit(), host=provenance.host(),
               versions=provenance.versions(), standin=False,
               gpu=torch.cuda.get_device_name() if torch and torch.cuda.is_available() else None)
    with open(stem + ".json", "x") as fh:
        json.dump(rec, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=sorted(testset.SETS))
    ap.add_argument("--d", type=int, required=True, choices=sorted(decoder.NETWORKS))
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--shots", required=True, help="directory with the chunk files")
    ap.add_argument("--opponents", nargs="*", default=["plain PM", "correlated PM"],
                    choices=opponents.NAMES + opponents.CONTROLS)
    ap.add_argument("--routing", choices=("fp32", "deployed", "published"), default="fp32",
                    help="ours' c and routing: the fp32 reference, the deployed executable, or the published mask")
    ap.add_argument("--post", choices=("correlated", "plain"), default="correlated",
                    help="ours' routed shots: correlated PM (registered), or plain PM with decoder.TAU_PLAIN's tau")
    ap.add_argument("--build", default="build/d7", help="--routing deployed at d = 7: timing/d7/build.sh's output")
    ap.add_argument("--device", default="cuda" if torch and torch.cuda.is_available() else "cpu")
    ap.add_argument("--alpha", type=float, default=0.025)
    ap.add_argument("--out", default="")
    ap.add_argument("--net", default="", help="ours: another network of d (weights/configs_sv2.json), or none")
    ap.add_argument("--tau", default="config", help="ours' tau: config, none (the network alone) or a float")
    ap.add_argument("--save", default="", help="a directory for every decoder's per-shot arrays and record")
    a = ap.parse_args()
    chunks = testset.parse_range(a.chunks)
    name = a.net or decoder.NETWORKS[a.d]
    ours = name != "none"
    assert ours or a.opponents, "--net none needs opponents"
    corr = a.post == "correlated"
    model = cfg = dep = pub = None
    tau = None
    if ours:
        cfg = decoder.config(name)
        assert cfg["d"] == a.d, f"{name} is not a network of d = {a.d}"
        if a.routing == "deployed":
            from timing import deployed
            dep = deployed.Logits(a.d, a.build, name)
        else:
            model, cfg = decoder.load_network(name, a.device)
        if not corr:                                    # the reverse control: its own tau
            assert a.routing != "published" and name == decoder.NETWORKS[a.d] and a.tau == "config", \
                "--post plain routes the registered network with its own tau: use --routing deployed or fp32"
            cfg = dict(cfg, policy=dict(cfg["policy"], tau=decoder.TAU_PLAIN[a.d]))
        assert a.tau != "config" or cfg.get("policy"), f"{name} has no policy before the freeze: --tau none"
        tau = cfg["policy"]["tau"] if a.tau == "config" else None if a.tau == "none" else float(a.tau)
        tests = decoder.routing_manifest()["test"]
        if name == decoder.NETWORKS[a.d] and a.set in tests and str(a.d) in tests[a.set]:
            pub = decoder.published_mask(tests[a.set][str(a.d)])
        assert a.routing != "published" or (pub is not None and a.tau == "config"), \
            "--routing published: the registered network on a registered test set at its tau"
    pm = circuits.matcher(a.d, correlated=corr) if ours and tau is not None else None
    opp = {n: opponents.make(n, a.d, a.device) for n in a.opponents}
    keys = (["ours"] if ours else []) + list(opp)
    fail = {n: [] for n in keys}
    arr = {n: {"dec": []} for n in keys}
    if ours:
        arr["ours"].update(c=[], routed=[])
    sha = {"obs": hashlib.sha256(), **{n: hashlib.sha256() for n in opp}}
    obs_all = []
    routed = differ = 0
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for j in chunks:
        det, obs = testset.load_chunks(a.set, a.d, [j], a.shots)
        obs_all.append(obs)
        sha["obs"].update(np.asarray(obs, np.uint8).tobytes())
        line = f"d{a.d} chunk {j}: "
        if ours:
            c = dep(det) if dep else decoder.logits(model, det, a.device)
            rp = pub[j * testset.CHUNK:(j + 1) * testset.CHUNK] if pub is not None else None
            r = rp if a.routing == "published" else (decoder.route(c, tau) if tau is not None else
                                                     np.zeros(len(c), bool))
            _, r, dec = decoder.decode(model, cfg, det, a.device, pm, c=c, routed=r, correlated=corr)
            routed += int(r.sum())
            differ += int((r != rp).sum()) if rp is not None else 0
            fail["ours"].append(dec != obs)
            arr["ours"]["c"].append(np.asarray(c, np.float32))
            arr["ours"]["routed"].append(r.astype(np.uint8))
            arr["ours"]["dec"].append(dec.astype(np.uint8))
            line += (f"ours {int(fail['ours'][-1].sum())} failures ({int(r.sum())} routed" +
                     (f", {int((r != rp).sum())} routed differently from the published mask); " if rp is not None and corr
                      else " to plain PM); " if not corr else "); "))
        for n, f in opp.items():
            dn = np.asarray(f(det), np.uint8)
            sha[n].update(dn.tobytes())
            fail[n].append(dn != obs)
            arr[n]["dec"].append(dn)
        print(line + ", ".join(f"{n} {int(fail[n][-1].sum())}" for n in opp) + f"  [{time.time() - t0:.0f} s]", flush=True)
    obs_all = np.concatenate(obs_all).astype(np.uint8)
    if a.save:
        ended = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        for n in keys:
            x = {k: np.concatenate(v) for k, v in arr[n].items()}
            x["obs"] = obs_all
            if n == "ours":
                eng = ("fp32 PyTorch reference" if a.routing == "fp32" else "the studies' published mask"
                       if a.routing == "published" else "timing/d7 resident kernel, one pass over the shots" if a.d == 7
                       else f"timing/engine.py FROZEN Triton engine, {cfg.get('engine', {}).get('stages', 3)} pipeline "
                            "stages, one shot per call")
                meta = dict(kind="network", network=name, network_sha256=decoder.weights_sha256(name), tau=tau,
                            tau_mode=a.tau, routing=a.routing, post=a.post, engine=eng,
                            batch_one=a.routing == "deployed", failures=int((x["dec"] != obs_all).sum()),
                            network_alone_failures=int(((x["c"] > 0).astype(np.uint8) != obs_all).sum()),
                            routed=int(x["routed"].sum()))
                save(a, name, x, dict(meta, started=started, ended=ended))
            else:
                meta = dict(kind="opponent", opponent=n, failures=int((x["dec"] != obs_all).sum()))
                if n in opponents.CONTROLS + ("Ising M1 + PM", "Ising M4 + PM"):
                    meta.update(ising_checkpoint_sha256=opponents.ISING[n.split()[1]][2],
                                numerics="official PyTorch (NVIDIA's runner's TF32 settings)")
                if "BP-OSD" in n:
                    meta.update(bposd=dict(opponents.BPOSD, **opponents.BPOSD_ARMS["minllr" if "min-LLR" in n else "bp60"]))
                save(a, KEYS[n], x, dict(meta, started=started, ended=ended))
    if not ours:
        return
    fa = np.concatenate(fail["ours"])
    res = dict(set=a.set, d=a.d, chunks=a.chunks, shots=len(fa), network=name, routing=a.routing, post=a.post,
               tau=tau, obs_sha256=sha["obs"].hexdigest(),
               ours=dict(failures=int(fa.sum()), routed=routed, routed_fraction=routed / len(fa),
                         routed_differ_published=differ if corr and pub is not None else None), cells={})
    for n in opp:
        res["cells"][n] = dict(stats.paired(fa, np.concatenate(fail[n]), a.alpha), decisions_sha256=sha[n].hexdigest())
        c = res["cells"][n]
        print(f"ours {c['fail_a']} vs {n} {c['fail_b']} on {c['shots']} shots: only ours {c['only_a']}, only "
              f"{n} {c['only_b']}, McNemar p {c['mcnemar_p']:.3g}, {100 * (1 - a.alpha):g} % interval of the LER "
              f"difference [{c['diff_lo']:.3g}, {c['diff_hi']:.3g}] -> {c['verdict']}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
