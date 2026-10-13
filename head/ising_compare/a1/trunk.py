"""A1's trunk initialisations (revision/REVISION_PLAN.md 2.1, "Initialization"). For training seed s: torch.manual_seed(s),
then the network is built on the CPU with the deployed trunk (arch.net.ConvLines at d = 9, width 128, 3 blocks;
bb.net.ConvLinesBB on BB72, 3 blocks of width 128), so its trunk (stem and blocks; on BB also the per-layer bias) holds
the first draws after the seed, before any head; that state is saved with the trunk configuration, the seed and the
torch version, and its sha256 printed (and registered in the job arguments). Every arm of seed s loads it with --init
(load_trunk: exactly the trunk's keys, the same trunk configuration and seed). The tool also builds each A1 head kind
the way the trainers do (seed s, head seed s + 1,000) and checks that its trunk equals the saved one.
    python -m a1.trunk surface --seed 1 --out DIR
    python -m a1.trunk bb72 --seed 1 --design data/bb/design_bb72.json --out DIR"""
import argparse
import hashlib
import os

import torch

from a1 import seeds as a1seeds

TRUNK_CFG = {"convlines": ("d", "width", "bott", "blocks", "stem_k", "block_k"),
             "convlines_bb": ("code", "width", "bott", "blocks", "stem_taps", "blk_taps")}


def load_trunk(model, ck, seed, keys_fn):
    """A trunk file into model: its seed must be the run's --seed, its configuration the model's trunk, its keys exactly
    the model's trunk keys; the head keys keep their initialisation."""
    assert ck.get("kind") == "trunk" and ck["seed"] == seed, f"trunk file of seed {ck.get('seed')}, run seed {seed}"
    arch = model.cfg["arch"]
    assert ck["cfg"]["arch"] == arch and all(ck["cfg"][k] == model.cfg[k] for k in TRUNK_CFG[arch]), \
        "trunk configuration differs"
    keys = keys_fn(model)
    assert sorted(ck["state"]) == sorted(keys), "the file's keys are not this network's trunk"
    res = model.load_state_dict(ck["state"], strict=False)
    assert not res.unexpected_keys and set(res.missing_keys) == set(model.state_dict()) - set(keys), res


def sha_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build(code, seed, head, design=None, aux=False, head_seed=True):
    """The network as the trainer builds it for an arm (torch seed s, head seed s + 1,000), on the CPU."""
    torch.manual_seed(seed)
    hs = a1seeds.head_seed(seed) if head_seed else None
    if code == "surface":
        from arch import net
        return net.ConvLines(9, 128, 32, 3, 4, head=head, aux_lines=aux, head_seed=hs), net.trunk_keys
    from bb import net
    stem, blk = net.taps_from_design(design)
    return net.ConvLinesBB("bb72", 0.003, stem, blk, 128, 32, 3, 4, head=head, aux_translates=aux,
                           head_seed=hs), net.trunk_keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("code", choices=["surface", "bb72"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--design", default="data/bb/design_bb72.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    deployed = "vote" if a.code == "surface" else "joint"
    model, keys_fn = build(a.code, a.seed, deployed, a.design, head_seed=False)
    keys = keys_fn(model)
    sd = model.state_dict()
    ck = {"kind": "trunk", "seed": a.seed, "code": a.code, "cfg": dict(model.cfg),
          "state": {k: sd[k].detach().clone() for k in keys}, "torch": torch.__version__}
    heads = [("vote", False), ("line", False), ("pooled", False), ("pooled", True)] if a.code == "surface" else \
        [("joint", False), ("pooled", False), ("pooled", True)]
    for head, aux in heads:                      # the trainers' build: the trunk does not depend on the head
        m2, _ = build(a.code, a.seed, head, a.design, aux)
        s2 = m2.state_dict()
        assert all(torch.equal(s2[k], ck["state"][k]) for k in keys), f"head {head} changes the trunk draw"
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"trunk_{a.code}_s{a.seed}.pt")
    torch.save(ck, path)
    print(f"TRUNK {path} sha256 {sha_file(path)} keys {len(keys)} params {sum(v.numel() for v in ck['state'].values())}",
          flush=True)


if __name__ == "__main__":
    main()
