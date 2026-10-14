"""CASCADE-TRT diagnostics: one stage per process, faulthandler on, progress printed (which stage crashes).
    python -X faulthandler diag.py trt1|trtB|trt1nograph|comp1|both NET"""
import faulthandler
import os
import sys
import time

faulthandler.enable()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trt_alone as ta  # noqa: E402
import trt_common as tc  # noqa: E402


def say(*a):
    print(f"{time.strftime('%T')}", *a, flush=True)


def main(which, net):
    import torch
    dev = ta.setup()
    PL = os.environ["CTRT_PLANS"]
    D = 432 if net == "bb" else 2184
    G = tc.TRTGraphBB if net == "bb" else tc.TRTGraphD13
    B = ta.BATCH[net]
    if which in ("comp1", "both"):
        say("compiled: building")
        c1, cB, sha = ta.paths(net, "compiled", dev, PL, os.environ.get("CTRT_CKPT", ""))
        say("compiled built", sha[:12])
        c1.replay()
        torch.cuda.synchronize()
        say("compiled b1 replay ok")
    for name, b, graph in (("trt1", 1, True), ("trtB", B, True), ("trt1nograph", 1, False)):
        if which in (name, "both") or (which == "trtall"):
            say(name, "deserialise + capture")
            g = G(tc.plan_path(PL, net, b), b, D, dev, graph=graph)
            say(name, "built", g.info())
            g.replay()
            torch.cuda.synchronize()
            out = (g.bits, g.conf) if net == "bb" else (g.logit, g.dec)
            say(name, "replay ok", [tuple(o.shape) for o in out], out[1].flatten()[:4].tolist())
    say("done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
