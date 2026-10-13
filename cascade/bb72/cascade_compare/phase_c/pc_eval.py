"""Phase C reads (PHASE_C_PLAN.md sections 2.4 and 3): bb.cascade_eval with two changes of the computation.
- The ConvLines networks (ours) run on the fused engine, the path they are timed on: bb.fused's kernels over
  pc_common.FUSED_READ_B shots per call (pc_fused.Batched), whose every decision and margin are bitwise those of the
  registered batch-1 engine (bb.engine "fused"). The class index is the decision's bits (bit j is observable j). Their
  read signature says prec "fused" and names the kernels' sources.
- The depth-rule Cascade networks (pc_common.EVAL_COMPILED, C6 and C6') run under torch.compile (gate finding 9: about
  0.36 GPU-hour less over the development blocks): each module is compiled once per process (default mode,
  dynamic=False: one graph per batch shape, 8,192 and the 1,808 left of a 10,000-shot chunk) under the same bf16
  autocast; their read signature says "compiled". With PC_EAGER=1 they run eagerly; the claim-point check (pc_speed.py
  check) compares the compiled read with an eager read of the same rows.
Every other network (C14, C14') runs exactly as in bb.cascade_eval. Every read runs on phase C's one GPU model
(pc_common.GPU, a full RTX PRO 6000; any other card is refused), and each network's signature records the driver.

  python pc_eval.py [bb.cascade_eval's arguments]"""
import os
import sys

import torch
from bb import cascade_eval as ce

import pc_common as pc

COMPILE = set() if os.environ.get("PC_EAGER") else {os.path.realpath(pc.NETS[t][0]) for t in pc.EVAL_COMPILED}
_init, _call = ce.Predictor.__init__, ce.Predictor.__call__


def _phase_c_init(self, path, dev, *args, **kw):
    _init(self, path, dev, *args, **kw)
    self.sig["driver"] = pc.gpu_env(dev)["driver"]           # refused off phase C's GPU (PHASE_C_PLAN.md 5.1)
    self.fused = None
    if self.kind == "convlines":
        import pc_fused
        self.fused = pc_fused.Batched(self.model, pc.FUSED_READ_B)
        self.prec, self.bs = "fused", pc.FUSED_READ_B
        self.sig.update(prec="fused", batch=pc.FUSED_READ_B, fused=pc_fused.signature(pc.FUSED_READ_B))
    elif os.path.realpath(path) in COMPILE:
        self.model = torch.compile(self.model, dynamic=False)
        self.sig["compiled"] = "torch.compile, default mode, dynamic=False"


@torch.no_grad()
def _phase_c_call(self, det, bs=None):
    if self.fused is None:
        return _call(self, det, bs)
    bits, margin = self.fused.run(det)
    cls = (bits.long() << torch.arange(bits.shape[1], device=bits.device)).sum(1)
    return {"bits": bits, "conf": margin, "cls": cls}


ce.Predictor.__init__, ce.Predictor.__call__ = _phase_c_init, _phase_c_call

if __name__ == "__main__":
    sys.argv[0] = ce.__file__
    ce.main()
