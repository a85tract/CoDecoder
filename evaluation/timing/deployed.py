"""The deployed executables' logits, and the validation routing masks the timing scripts read, computed once.

The studies' decisions came from the deployed executables, not from the PyTorch reference (convlines.decoder): the
Triton engine (timing/engine.py with FROZEN, deploy tables folded on the CPU in fp32) at d = 9-15, the same engine with
4 pipeline stages on study 4's deployed tables (weights/surface/d17_b5_r787_deployed.safetensors) at d = 17, and the resident
CUDA kernel (timing/d7, built by timing/d7/build.sh; sm_90 GPUs such as the H100 and H200) at d = 7, all with fp16
matrix inputs and fp32 accumulation, one shot at a time. Logits(d) gives their c here. The routing rule is the
decoder's: routed = |double(c)| < tau, any other shot decided by c > 0.

    python -m timing.deployed --d 9 --val DIR --out MASKS [--logits fp32] [--build build/d7] [--net NAME]

computes c and the routing mask of validation chunks 0-19 (200,000 shots) once per network -> MASKS/val_d9.npz (c,
routed, d, the network, its tau, the kind of logits and the shots' digest; with --net NAME, MASKS/val_NAME.npz). timing.cpu_term term reads its mask, and
timing.gpu_term compares its outputs with its c bit for bit. When the shots are the studies' own validation shots
(their digest is in routing/manifest.json), the mask is also compared with the studies' published mask (routing/val_d*,
the masks the studies' CPU term read): at d = 15 `convlines.testset val` regenerates the studies' shots on x86-64; at
d = 7-13 the studies' shots were sampled on an ARM64 machine and x86-64 gives other shots, so the mask computed here is
the one to use; at d = 17 it regenerates the study's V17b shots. --logits fp32 takes c from the PyTorch reference
instead (no GPU needed; not at d = 17, whose fp32 network is not in this release); its routing can differ from the
deployed one on the few shots whose |c| lies within rounding of tau."""
import argparse
import hashlib
import json
import os

import numpy as np


def shots_sha256(det, obs):
    """The digest of a set of shots: sha256 of np.packbits(det, axis=1)'s bytes, then the observable bytes."""
    h = hashlib.sha256(np.packbits(np.asarray(det, np.uint8), axis=1).tobytes())
    h.update(np.asarray(obs, np.uint8).reshape(-1).tobytes())
    return h.hexdigest()


def to_cuda(x):
    import torch
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_cuda(v) for v in x]
    return x


def tables(d, name=None):
    """(deploy tables on the CPU in fp32, config) of d's network (name: another network of d, e.g. a SURF-v2 one of
    weights/surface/configs_sv2.json): folded from the released weights (d = 7-15), or the deployed tables themselves where the
    release holds only those (d = 17)."""
    from convlines import decoder
    from convlines.net import deploy_tables
    name = name or decoder.NETWORKS[d]
    assert decoder.config(name)["d"] == d, f"{name} is not a network of d = {d}"
    if "tables" in decoder.config(name):
        return decoder.load_tables(name)
    model, cfg = decoder.load_network(name, "cpu")
    return deploy_tables(model), cfg


def engine(t, cfg):
    """The deployed Triton engine of tables t (on the GPU): FROZEN, with the network's own pipeline-stage count where its
    config names one (d = 17: 4, as study 4 deployed it; FROZEN's 3 at d = 9-15), PDL as engine.frozen chooses."""
    from timing import engine as E
    st = cfg.get("engine", {}).get("stages", E.FROZEN["stages"])
    if st == E.FROZEN["stages"]:
        return E.frozen(t)
    import torch
    return E.Engine(t, pdl=torch.cuda.get_device_capability() >= (9, 0), **{**E.FROZEN, "stages": st})


class Logits:
    """c of the deployed executable at d: Logits(d, build)(det (n, m) uint8) -> (n,) float32, one shot at a time as
    deployed (the Triton engine's call at batch 1; d = 7: one pass of the resident kernel over the shots, no routing)."""

    def __init__(self, d, build="build/d7", name=None):
        import torch
        self.d = d
        t, cfg = tables(d, name)
        t = to_cuda(t)                                     # folded on the CPU in fp32, as the deployed artifact
        if d == 7:
            from timing.d7.engine import Engine, Lib
            self.lib = Lib(os.path.join(build, "libs7router.so"))
            torch.zeros(1, device="cuda")                  # the primary context, current on this thread
            self.eng = Engine(t, self.lib, self.lib.ctx(), os.path.join(build, "kernel_sm90a.cubin"), body=1, fwd=1)
        else:
            self.eng = engine(t, cfg)

    def __call__(self, det):
        import torch
        det = np.ascontiguousarray(det, dtype=np.uint8)
        if self.d == 7:
            self.eng.start()
            o = self.lib.run_persistent(self.eng.pers, det, -1.0, None, len(det), warm=0)
            self.eng.stop()
            return o["z"]
        x = torch.from_numpy(det).cuda()
        c = torch.empty(len(det), dtype=torch.float32, device="cuda")
        with torch.inference_mode():
            for i in range(len(det)):
                c[i:i + 1].copy_(self.eng(x[i:i + 1])[0])
        return c.cpu().numpy()


def load(path, d, det, obs, kind=None):
    """A routing file of these validation shots: this module's output, or a studies' mask of routing/ (the manifest
    gives its shots' digest). Refused when it belongs to other shots or another d, or (kind given) to other logits.
    -> its arrays, routed as bool, with d, kind and shots_sha256."""
    from convlines import decoder
    raw = open(path, "rb").read()
    z = dict(np.load(path))
    if "shots_sha256" in z:
        meta = dict(d=int(z["d"]), kind=str(z["logits"]), shots_sha256=str(z["shots_sha256"]))
    else:
        sha = hashlib.sha256(raw).hexdigest()
        e = [e for e in decoder.routing_manifest()["val"].values() if e["sha256"] == sha]
        assert e, f"{path}: neither timing.deployed's output nor a mask of routing/manifest.json"
        meta = dict(d=e[0]["d"], kind="published", shots_sha256=e[0]["shots_sha256"])
    assert meta["d"] == d, f"{path} is a mask of d = {meta['d']}"
    assert meta["shots_sha256"] == shots_sha256(det, obs), f"{path} belongs to other validation shots"
    assert kind is None or meta["kind"] == kind, f"{path} holds {meta['kind']} logits, not {kind}"
    z.update(meta, routed=z["routed"].astype(bool))
    return z


def main():
    import torch
    from convlines import decoder, testset
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True, choices=sorted(decoder.NETWORKS))
    ap.add_argument("--val", required=True, help="the directory with validation chunks 0-19")
    ap.add_argument("--out", required=True)
    ap.add_argument("--logits", choices=("deployed", "fp32"), default="deployed")
    ap.add_argument("--build", default="build/d7", help="d = 7: timing/d7/build.sh's output directory")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="--logits fp32 only")
    ap.add_argument("--net", default="", help="another network of d (weights/surface/configs_sv2.json), at its config's tau")
    a = ap.parse_args()
    name = a.net or decoder.NETWORKS[a.d]
    assert decoder.config(name)["d"] == a.d, f"{name} is not a network of d = {a.d}"
    sh = [testset.load(os.path.join(a.val, testset.name("val", a.d, j)), a.d) for j in range(20)]
    det, obs = np.concatenate([s[0] for s in sh]), np.concatenate([s[1] for s in sh])
    digest = shots_sha256(det, obs)
    if a.logits == "deployed":
        c = Logits(a.d, a.build, name)(det)
    else:
        model, _ = decoder.load_network(name, a.device)
        c = decoder.logits(model, det, a.device)
    pol = decoder.config(name).get("policy")             # a SURF-v2 network has none before the freeze: c only
    tau = pol["tau"] if pol else None
    routed = decoder.route(c, tau) if pol else np.zeros(len(c), bool)
    rec = dict(d=a.d, network=name, tau=tau, logits=a.logits, shots=len(det), routed=int(routed.sum()), shots_sha256=digest)
    e = decoder.routing_manifest()["val"][str(a.d)]
    rec["studies_shots"] = digest == e["shots_sha256"]
    if rec["studies_shots"]:
        pub = decoder.published_mask(e)
        rec.update(routed_studies=int(pub.sum()), routed_differ_studies=int((pub != routed).sum()))
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, f"val_d{a.d}.npz" if not a.net else f"val_{name}.npz"), c=c.astype(np.float32),
             routed=routed.astype(np.uint8), d=a.d, network=name, tau=np.float64(np.nan if tau is None else tau),
             logits=a.logits, shots_sha256=digest)
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
