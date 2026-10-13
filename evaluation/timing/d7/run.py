"""Decodes chunks of d = 7 with the deployed executable, as the studies' test-set evaluation ran it: the resident kernel
gives c, the router's compiled loop sends a shot with |double(c)| < tau to correlated PyMatching (compiled into the
router library) and decides any other shot by c > 0. Per chunk it prints the failures and the routed shots, and checks
the outputs: the routed set against |c| < tau, the unrouted decisions against the sign of c, the routed decisions
against the PyMatching wheel's correlated decode, and c, the routing and the decisions against the PyTorch reference
(convlines.decoder, fp32; the kernel computes with fp16 matrix inputs and fp32 accumulation, so c differs by rounding).
--out writes c, routed and the decision of every shot.
    python -m timing.d7.run --chunks shots/T2/d7_p0.006_X_test2_000.npz [...] [--build build/d7] [--out DIR]"""
import argparse
import json
import os

import numpy as np
import torch


def to_cuda(x):
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_cuda(v) for v in x]
    return x


def main():
    from convlines import circuits, decoder, testset
    from convlines.net import deploy_tables
    from timing.d7.engine import Engine, Lib
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="+", required=True, help="chunk files of d = 7 (test or validation format)")
    ap.add_argument("--build", default="build/d7", help="build.sh's output directory")
    ap.add_argument("--out", default="", help="a directory for the per-shot outputs (one .npz per chunk)")
    a = ap.parse_args()
    name = decoder.NETWORKS[7]
    model, cfg = decoder.load_network(name, "cpu")
    tau = cfg["policy"]["tau"]
    t = to_cuda(deploy_tables(model))                  # folded on the CPU in fp32, as the deployed artifact
    ref_model = decoder.load_network(name, "cuda")[0]
    c7 = circuits.circuit(7)
    corr = circuits.matcher(7, True, c7)
    lib = Lib(os.path.join(a.build, "libs7router.so"))
    torch.zeros(1, device="cuda")                      # the primary context, current on this thread
    ctx = lib.ctx()
    pmh = lib.pm(str(circuits.dem(c7)), c7.num_detectors)
    eng = Engine(t, lib, ctx, os.path.join(a.build, "kernel_sm90a.cubin"), body=1, fwd=1)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    for path in a.chunks:
        det, obs = testset.load(path, 7)
        det = np.ascontiguousarray(det, dtype=np.uint8)
        eng.start(tau=tau)
        o = lib.run_persistent(eng.pers, det, tau, pmh, len(det), warm=1000)
        sp = eng.stop()
        c, routed, dec = o["z"], o["routed"].astype(bool), o["dec"]
        c_ref, routed_ref, dec_ref = decoder.decode(ref_model, cfg, det, "cuda", corr)
        wheel = circuits.pm_decode(corr, det[routed], correlated=True)
        row = dict(chunk=os.path.basename(path), shots=len(det), served=sp["served"], final_state=sp["state"],
                   failures=int((dec != obs).sum()), routed=int(routed.sum()),
                   routed_is_tau=bool(np.array_equal(routed, np.abs(c.astype(np.float64)) < tau)),
                   unrouted_sign=bool(np.array_equal(dec[~routed], (c[~routed] > 0).astype(np.uint8))),
                   routed_equal_wheel=bool(np.array_equal(dec[routed], wheel)),
                   decisions_equal_reference=int((dec == dec_ref).sum()),
                   routed_differ_reference=int((routed != routed_ref).sum()), failures_reference=int((dec_ref != obs).sum()),
                   max_abs_c_diff_reference=float(np.abs(c.astype(np.float64) - c_ref).max()))
        print(json.dumps(row), flush=True)
        if a.out:
            np.savez_compressed(os.path.join(a.out, os.path.basename(path).replace(".npz", "_d7exec.npz")), c=c,
                                routed=routed.astype(np.uint8), decision=dec.astype(np.uint8))


if __name__ == "__main__":
    main()
