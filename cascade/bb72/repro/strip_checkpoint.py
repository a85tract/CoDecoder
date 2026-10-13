"""Write an inference checkpoint from a checkpoint of the Cascade comparison.

The comparison read nine checkpoints (PHASE_C_PLAN.md 2.1; every output's signature has each file's sha256). A
training run wrote each of them, and besides the network they hold that run's configuration, log and environment. The
released files keep only what inference reads:
  Cascade networks (bb.cascade_net):  {"cfg": the keys bb.cascade_net.build reads, "ema": the final EMA state dict}
  ConvLines-BB networks (bb.net):     {"cfg": the network's cfg (all of it: architecture only), "state": the final EMA
                                       state dict, "completed": True (bb.engine.load_net requires it)}
and "source_sha256", the sha256 of the checkpoint the comparison read. Every tensor is the source's, bit for bit.

  python strip_checkpoint.py SRC DST    writes DST, reloads it with torch.load(weights_only=True) and checks every
                                        tensor against SRC byte for byte; prints both sha256
"""
import hashlib
import sys

import torch

CASCADE_KEYS = ("code", "H", "L", "res", "act_ckpt", "resid_bf16", "conv_impl")
CONVLINES_KEYS = ("arch", "code", "p", "width", "bott", "heads", "mlp", "blocks", "stem_taps", "blk_taps", "tblocks",
                  "rbins")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def same_bytes(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    a, b = a.detach().cpu().reshape(-1), b.detach().cpu().reshape(-1)
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def main(src, dst):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    if cfg.get("arch") == "convlines_bb":
        extra = set(cfg) - set(CONVLINES_KEYS)
        if extra:
            raise SystemExit(f"{src}: cfg has keys outside the architecture: {sorted(extra)}")
        key, out = "state", {"cfg": dict(cfg), "state": ck["state"], "completed": True}
    else:
        key, out = "ema", {"cfg": {k: cfg[k] for k in CASCADE_KEYS if k in cfg}, "ema": ck["ema"]}
    out["source_sha256"] = sha256_file(src)
    torch.save(out, dst)
    back = torch.load(dst, map_location="cpu", weights_only=True)
    sd0, sd1 = ck[key], back[key]
    assert list(sd0) == list(sd1), "state dict keys differ"
    bad = [k for k in sd0 if not same_bytes(sd0[k], sd1[k])]
    assert not bad, f"tensors differ: {bad[:5]}"
    assert back["cfg"] == out["cfg"]
    n = sum(v.numel() for k, v in sd1.items() if v.is_floating_point())
    print(f"{dst}: {len(sd1)} tensors equal byte for byte, {n:,} floating-point values; cfg {back['cfg']}; "
          f"source {out['source_sha256']}; released {sha256_file(dst)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
