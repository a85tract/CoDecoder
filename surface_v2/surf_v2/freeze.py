"""The freeze record of SURF-v2 (surf_v2/REGISTRATION.md), made in a checkout of the release's surf-v2 branch once every
network's calibration read is fitted, then committed with the files it names and pushed to GitHub before any test shot
exists:
  - every network's weight file and sha256 (the new ones copied into weights/), its q and tau (repr of the float32) and
    the q rule's output (surf_v2/policies/), the deployed networks' published tau (R17's at d = 17);
  - the read path's code digest, which every calibration read carries and every test read and timing record must carry;
  - any training-failure exception (--exceptions: {d: {"why", "checks", "row_network", "ancestry"}}), labelled;
  - the config's sha256 (the analysis, the families, the registered times, the timing rules, the paper-row rule) and the
    test set's seed rule and sizes (convlines.beacon, convlines.testset).
weights/configs_sv2.json gets each new network's sha256, final_ema_sha256 and policy.

    python -m surf_v2.freeze make --policies POL --cal-reads READS --exports WEIGHTS_DIR --research-commit SHA
        [--exceptions EXC.json] [--dry --out DRY/freeze.json]"""
import argparse
import glob
import hashlib
import json
import os
import shutil

from convlines import beacon, decoder, provenance, testset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def cal_read(reads, set_, d, name):
    f = [x for x in reads if x["set"] == set_ and x["d"] == d and x["decoder"] == name]
    assert len(f) == 1, f"{name}: {len(f)} calibration reads"
    assert f[0]["tau"] is None and f[0]["kind"] == "network", f"{name}: the calibration read is not the network alone"
    return f[0]


def make(a):
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    exc = {int(k): v for k, v in (json.load(open(a.exceptions)) if a.exceptions else {}).items()}
    for d, e in exc.items():
        assert set(e) >= {"why", "checks", "row_network", "ancestry"}, f"exception at d = {d}: {sorted(e)}"
        assert e["row_network"] == cfg["networks"][str(d)]["deployed"]
    cset = cfg["dry"]["set"] if a.dry else cfg["calibration"]["set"]
    reads = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.cal_reads, "**", "*.json"), recursive=True))]
    reads = [r for r in reads if r.get("what", "").startswith("SURF-v2 read") and r["set"] == cset]
    digest = provenance.code_digest()
    nets, conf = {}, json.load(open(os.path.join(ROOT, "weights", "configs_sv2.json")))
    for k in map(str, cfg["distances"]):
        d = int(k)
        for role in ("new", "deployed"):
            name = cfg["networks"][k].get(role)
            if name is None or (role == "new" and d in exc):
                continue
            r = cal_read(reads, cset, d, name)
            assert r["code_digest"] == digest or a.dry, f"{name}: the read path changed since its calibration read"
            n = dict(d=d, role=role, cal_read=r["npz"], cal_read_sha256=r["npz_sha256"], cal_read_code_digest=r["code_digest"],
                     cal_read_commit=r["release_commit"], cal_routing=r["routing"])
            if role == "deployed":
                cf = decoder.config(name)
                n.update(file=cf.get("file") or cf["tables"], sha256=decoder.weights_sha256(name), q=cf["policy"]["q"],
                         tau=cf["policy"]["tau"], tau_repr=repr(cf["policy"]["tau"]), outcome="published tau")
                assert r["network_sha256"] == n["sha256"], f"{name}: the calibration read's weights are not the release's"
            else:
                e = conf["networks"][name]
                src = os.path.join(a.exports, e["file"])
                ex = json.load(open(src[:-len(".safetensors")] + ".export.json"))
                sha = sha_file(src)
                assert sha == ex["sha256"] == r["network_sha256"], f"{name}: the weights differ from the calibration read's"
                dst = os.path.join(ROOT, "weights", e["file"])
                if os.path.abspath(src) != os.path.abspath(dst):
                    assert not os.path.exists(dst) or sha_file(dst) == sha, f"{dst} exists and differs"
                    shutil.copyfile(src, dst)
                pf = os.path.join(a.policies, f"policy_{name}.json")
                pol = json.load(open(pf))
                assert pol["network"] == name and pol["read_sha256"] == r["npz_sha256"], f"{pf}: another read's policy"
                assert pol["config_sha256"] == sha_file(os.path.join(HERE, "config.json")), f"{pf}: another config"
                os.makedirs(os.path.join(HERE, "policies"), exist_ok=True)
                pdst = os.path.join(HERE, "policies", os.path.basename(pf))
                if os.path.abspath(pf) != os.path.abspath(pdst):
                    shutil.copyfile(pf, pdst)
                ch = pol["chosen"]
                q, tau = (ch["q"], ch["tau"]) if ch else (0.0, 0.0)
                outcome = pol["outcome"] if ch else "no admissible q"
                n.update(file=e["file"], sha256=sha, final_ema_sha256=ex["final_ema_sha256"], q=q, tau=tau, tau_repr=repr(tau),
                         outcome=outcome, policy_file=os.path.relpath(pdst, ROOT), policy_sha256=sha_file(pdst),
                         calibration_failures=ch["failures"] if ch else pol["network_failures"],
                         projected_us=ch["projected_us"] if ch else None)
                e.update(sha256=sha, policy=dict(rule=pol["rule"], q=q, tau=tau, tau_repr=repr(tau), outcome=outcome,
                                                 calibration_failures=n["calibration_failures"], shots=pol["shots"]))
                e["training"]["final_ema_sha256"] = ex["final_ema_sha256"]
            nets[name] = n
    with open(os.path.join(ROOT, "weights", "configs_sv2.json"), "w") as fh:
        json.dump(conf, fh, indent=1)
    digests = sorted({n["cal_read_code_digest"] for n in nets.values()})
    assert len(digests) == 1 or a.dry, f"the calibration reads ran more than one read path: {digests}"
    if len(digests) > 1:                                  # dry runs only: reads resumed from an earlier commit's job
        digests = [digest]
    fz = dict(what="SURF-v2's freeze record (surf_v2/REGISTRATION.md)" + (" -- DRY RUN" if a.dry else ""), dry=a.dry,
              research_commit=a.research_commit, read_code_digest=digests[0],
              read_code_files=provenance.code_files(), config_sha256=sha_file(os.path.join(HERE, "config.json")),
              configs_sv2_note="weights/configs_sv2.json holds the same sha256, q and tau", networks=nets,
              exceptions={str(d): e for d, e in sorted(exc.items())},
              test=dict(set="sv2test", seed_rule=beacon.SEED_RULE, tags={str(d): t.decode() for d, t in beacon.TAGS.items()},
                        beacon=dict(chain_hash=beacon.CHAIN["chain_hash"], delay_s=beacon.DELAY,
                                    round_rule="the first drand mainnet round at least 600 s after GitHub's push time "
                                               "of this commit"),
                        chunks=cfg["test"]["chunks"], bposd_chunks=cfg["test"]["bposd_chunks"],
                        sizes={str(d): testset.SV2["sv2test"]["chunks"][d] * testset.CHUNK for d in cfg["distances"]}),
              paper_rows=cfg["paper_rows"]["rule"], timing=cfg["timing"]["qualification"],
              args=sorted(os.path.relpath(f, ROOT) for f in glob.glob(os.path.join(HERE, "args", "*"))))
    out = a.out or os.path.join(HERE, "freeze.json")
    with open(out, "x") as fh:
        json.dump(fz, fh, indent=1)
    for name, n in nets.items():
        print(f"d = {n['d']} {n['role']:8s} {name}: q {n['q']} tau {n['tau_repr']} ({n['outcome']}) sha256 {n['sha256'][:16]}",
              flush=True)
    print(f"freeze record {out}: {sha_file(out)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make")
    p.add_argument("--policies", required=True, help="surf_v2.qrule fit's outputs")
    p.add_argument("--cal-reads", required=True, help="the calibration reads (convlines.evaluate --save)")
    p.add_argument("--exports", required=True, help="surf_v2.export's weight files and records")
    p.add_argument("--research-commit", required=True)
    p.add_argument("--exceptions", default="")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--out", default="")
    a = ap.parse_args()
    make(a)


if __name__ == "__main__":
    main()
