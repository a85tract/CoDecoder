"""Phase C of the BB72 Cascade comparison (cascade_compare/PHASE_C_PLAN.md): the fixed inputs and the shared helpers.

Everything the plan fixes in advance lives here: the networks and their files, the two settings (circuit families of
bb.cascade_data), the seed kinds of phase C, the p grids and block sizes, the routing candidates and the speed protocol's
constants. The other scripts import it; nothing here imports torch, so the Relay-BP venv can import it too.

Seeds: bb.cascade_data.eval_seed(kind, p, chunk, family), the study's namespace (2^55 + ...), kinds never used before phase C:
  10  curves and routing development (every network alone; the development block of the routing rule)
  11  routing confirmation (read once, after locked_<setting>.json is committed)
  12  the speed stream (inputs of the end-to-end pipeline timing and of the agreement checks)
  13  dry runs (development slices; never reported)"""
import hashlib
import json
import math
import os

W = "<cluster-scratch>"
W3, BB, W4 = W + "/cascade_w3", W + "/bb", W + "/cascade_w4"
CODE, K, R = "bb72", 12, 6

# tag -> (checkpoint, the setting it was trained in, kind)
NETS = {
    "C14": (W3 + "/ckpt/C14/final_ema.pt", "uniform", "cascade"),
    "C6": (W3 + "/ckpt/C6/final_ema.pt", "uniform", "cascade"),
    "S55": (W3 + "/ckpt/S55/final_ema.pt", "uniform", "convlines"),
    "C14p": (W3 + "/ckpt/C14_dfm/final_ema.pt", "depspam_fm", "cascade"),
    "C6p": (W4 + "/ckpt/C6_dfm/final_ema.pt", "depspam_fm", "cascade"),          # from the Windows workstation, after it ends
    "S55p": (W3 + "/ckpt/S55_dfm/final_ema.pt", "depspam_fm", "convlines"),
    "S55p_deep": (W3 + "/ckpt/S55_dfm_deep/final_ema.pt", "depspam_fm", "convlines"),
    "r11": (BB + "/ckpt/r11/final_ema.pt", "uniform", "convlines"),
    "r12": (BB + "/ckpt/r12/final_ema.pt", "uniform", "convlines"),
}
LATE = ("C6p",)                                    # evaluated by the add-on jobs when its weights reach Alpine
SETTINGS = ("uniform", "depspam_fm")
BIG = {"uniform": "C14", "depspam_fm": "C14p"}     # "Cascade" in each setting (question 3's large, deep network)
DEPTH = {"uniform": "C6", "depspam_fm": "C6p"}     # the depth-rule variant, a secondary opponent
SMALL = {"uniform": ("S55",), "depspam_fm": ("S55p", "S55p_deep")}
REGISTERED = "r12"                                 # question 1's network with routing (its accurate mode)
OURS = ("S55", "S55p", "S55p_deep", "r11", "r12")  # our networks (ConvLines-BB); the others are Cascade networks
EVAL_COMPILED = ("C6", "C6p")                      # read through torch.compile (pc_eval.py; gate finding 9)

KIND_DEV, KIND_CONF, KIND_SPEED, KIND_DRY = 10, 11, 12, 13
CHUNK = 10_000                                     # shots per chunk (cascade_eval)
PS = (0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0055)
DEV_CHUNKS = {0.001: 1000, 0.0015: 300, 0.002: 100, 0.0025: 50, 0.003: 50,
              0.0035: 30, 0.004: 30, 0.0045: 30, 0.005: 30, 0.0055: 30}      # x 10,000 shots, kind 10
CONF_CAP = {0.001: 2000, 0.0015: 1000, 0.002: 500, 0.0025: 200, 0.003: 200,
            0.0035: 200, 0.004: 200, 0.0045: 200, 0.005: 200, 0.0055: 200}   # chunks: the confirmation's ceiling
RELAY_P = (0.002, 0.003)                           # the p of every configuration with Relay-BP

# routing candidates (the rule: PHASE_C_PLAN.md section 4)
Q1 = (0.01, 0.02, 0.05, 0.10, 0.20, 0.40)          # fraction of shots S hands to B (tau1 = the q1-quantile of S's margin)
R2 = (0.001, 0.0025, 0.005, 0.01)                  # fraction of all shots that reach Relay-BP in S -> B -> R
SCORES = ("minabs", "softplus")                    # B's confidence: min |logit|, or -sum softplus(-|logit|)
BASE_Q = {0.002: 0.0025, 0.003: 0.02}              # the registered accurate modes' q (BB72-2F at 0.2 %, BB72-3F at 0.3 %)
ROUTE_Q = tuple(sorted(set(Q1) | set(R2) | set(BASE_Q.values())))   # the routed fractions the speed screen tests
SPEED_MARGIN = 0.9                                 # feasible: predicted latency <= 0.9 x Cascade's, at both metrics
NCPU_RELAY = 16                                    # Relay-BP workers of the speed job (the amortized model's cores)
DELTA_REL, POWER, ALPHA = 0.10, 0.90, 0.05         # confirmation sizing: 10 % fewer block failures; power; the one alpha
# The headline family (PHASE_C_PLAN.md 6): every comparison with Cascade that can make or veto a headline, in both
# settings, judged by one Holm procedure at ALPHA: our networks alone at every p (the curves) and every locked
# configuration (S-B at every p; S-B-R, S-R and r12 -> R at RELAY_P). The confirmation is sized at ALPHA / M_HEADLINE.
M_HEADLINE = len(SETTINGS) * (len(OURS) * len(PS) + len(PS) + 3 * len(RELAY_P))
M_SECONDARY = len(SETTINGS) * (len(PS) + 3 * len(RELAY_P))   # non-inferiority statements, one per locked configuration
BATCHES = (1, 16, 64, 256, 1024, 4096)             # amortized-latency sweep; the verdict's batch size is the largest
VERDICT_B = 4096
FUSED_READ_B = VERDICT_B                           # shots per call of the ConvLines networks' fused engine in the reads
                                                   # and the screen's reference (pc_fused.Batched; bitwise the timed
                                                   # batch-1 path at any size)
REPS = 3                                           # fresh processes per network and per pipeline; the slowest counts
AGREE_REL, AGREE_Z = 0.01, 3.0                     # the accuracy tolerance of a timed path (agrees)
CHECK_CHUNKS = 10                                  # the claim-point check: 100,000 shots per p in RELAY_P and setting,
                                                   # chunks 10-19 (the screen used chunks 0-9 of the same kind and p)
PRECISION = {"graph": "BF16 (autocast)", "compiled": "BF16 (autocast, torch.compile)",
             "fused": "FP16 matmul inputs, FP32 accumulation (bb.engine fused)"}
# The card of every GPU step (PHASE_C_PLAN.md 5.1): a full RTX PRO 6000. A MIG slice's name adds its profile ("... MIG
# 1g.24gb"); gpu_env refuses any other card.
GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"


def gpu_env(dev):
    """The card of a GPU step, refused unless it is GPU; with its driver (gpu.sbatch exports it from nvidia-smi) and the
    venv's torch build. Every GPU step of phase C runs on one GPU model in one venv; the lock names them."""
    import torch
    name = torch.cuda.get_device_name(dev)
    if name != GPU:
        raise SystemExit(f"GPU {name!r}: every GPU step of phase C runs on a full {GPU}")
    return {"gpu": name, "driver": os.environ.get("PC_GPU_DRIVER", ""), "torch": torch.__version__}


def precision(path, kind):
    """The label of a timed path's precision (ConvLines' top-2 vote of the BF16 paths runs in FP32)."""
    return PRECISION[path] + ("; top-2 vote in FP32" if kind == "convlines" and path != "fused" else "")


def agrees(only_t, only_e, n_e):
    """The accuracy tolerance of a timed path (PHASE_C_PLAN.md 2.4). On the same shots, only_t counts the shots that only
    the timed computation fails (or routes), only_e those that only the evaluated computation fails (or routes), and n_e
    is the evaluated computation's count. They agree when |only_t - only_e| <= AGREE_REL n_e + AGREE_Z sqrt(only_t +
    only_e): the counts differ by at most 1 % of the evaluated count, beyond three standard deviations of the symmetric
    flips that rounding alone produces (bf16 Triton tiles depend on the batch size and flip a few near-zero logits)."""
    return abs(only_t - only_e) <= AGREE_REL * n_e + AGREE_Z * math.sqrt(only_t + only_e)


def frozen_paths(v):
    """A network's frozen timed paths (speed_components.json entry v): mode -> (path, batch). single: the single-shot
    latency and the single-mode pipelines; amortized: the network's amortized latency; batched: the batched pipelines."""
    pa, ba = v["a_best"].split("@")
    return {"single": (v["t1_path"], 1), "amortized": (pa, int(ba)), "batched": (v["a4096_path"], VERDICT_B)}


def decoder_key(c):
    """The routed decoder a locked configuration belongs to (its configurations over p form one decoder)."""
    if c["type"] == "S-R" and c["S"] == REGISTERED:
        return "r12 -> R"
    return {"S-B": "S -> B", "S-B-R": "S -> B -> R", "S-R": "S -> R"}[c["type"]]


def p_tag(p):
    return f"p{p:.5f}"


def p_block_to_pl(pb, k=K, r=R):
    """The paper's Eq. (1): block failure rate -> P_L per logical qubit per round (never applied to a difference)."""
    x = 2 * (1 - pb) ** (1 / k) - 1
    return (1 - x ** (1 / r)) / 2 if x > 0 else float("nan")


def wilson(f, n, z=1.959964):
    if n == 0:
        return 0.0, 1.0
    ph = f / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def point(f, n):
    lo, hi = wilson(f, n)
    return {"shots": int(n), "fail": int(f), "p_block": f / n if n else float("nan"), "pb_lo": lo, "pb_hi": hi,
            "pl": p_block_to_pl(f / n) if n else float("nan"), "pl_lo": p_block_to_pl(lo), "pl_hi": p_block_to_pl(hi)}


def holm(pvals, alpha=ALPHA):
    """Holm's step-down thresholds: index -> the alpha at which that test is judged (None once a rejection fails:
    it and every later test are not rejected)."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out, m, go = {}, len(pvals), True
    for rank, i in enumerate(order):
        a = alpha / (m - rank)
        out[i] = a if go else None
        if pvals[i] >= a:
            go = False
            out[i] = None
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Read(dict):
    """A read's arrays, each loaded from its npz once, and the confidences computed from them (gate finding 9: the rule
    sweeps every candidate over the same arrays)."""

    def __init__(self, npz):
        super().__init__()
        self.npz, self.conf = npz, {}

    def __missing__(self, k):
        self[k] = self.npz[k]
        return self[k]

    def __contains__(self, k):
        return dict.__contains__(self, k) or k in self.npz.files


def load_read(d, p):
    """One cascade_eval read (--save DIR) at p -> (arrays, json, {tag: model index}); the tags are matched through the
    checkpoints' paths (NETS)."""
    import numpy as np
    z = Read(np.load(os.path.join(d, f"{p_tag(p)}.npz")))
    meta = json.load(open(os.path.join(d, f"{p_tag(p)}.json")))
    by_path = {os.path.realpath(v[0]): t for t, v in NETS.items()}
    idx = {}
    for i, m in enumerate(meta["models"]):
        t = by_path.get(os.path.realpath(m["path"]))
        if t is None:
            raise SystemExit(f"{d} at {p}: model {m['path']} is not one of NETS")
        idx[t] = i
    return z, meta, idx


def confidence(z, meta, idx, tag, score="minabs"):
    """Per-shot confidence of a network in a read (lower = less sure): ConvLines the margin, Cascade min |logit| or
    -sum softplus(-|logit|)."""
    import numpy as np
    i = idx[tag]
    if score != "minabs" and meta["models"][i]["kind"] != "cascade":
        score = "minabs"
    if (i, score) not in z.conf:
        if score == "minabs":
            z.conf[(i, score)] = z[f"conf{i}"].astype(np.float64)
        else:
            z.conf[(i, score)] = -np.logaddexp(0.0, -np.abs(z[f"z{i}"].astype(np.float64))).sum(1)
    return z.conf[(i, score)]


def det_digests(meta):
    return {c["chunk"]: c["det_sha256"] for c in meta["chunk_meta"]}


def main():
    """Shell helpers for the job files (jobs/*.sh):
      groups dev               'p,p,...@0:N' per chunk count of the development block
      groups conf LOCKED.json  'p@0:N' per p of a lock (its confirmation block)
      groups check             'p,p@N:2N' the claim-point check's read (RELAY_P, CHECK_CHUNKS after the screen's)
      tags [late|compiled]     the networks whose checkpoint exists (LATE ones only with 'late'; with 'compiled' the
                               EVAL_COMPILED ones, late or not)
      paths TAG ...            their checkpoints, comma-separated
      need LOCKED.json         the networks a lock reads (B, its small networks, r12)"""
    import sys
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "groups" and args[0] == "dev":
        by = {}
        for p in PS:
            by.setdefault(DEV_CHUNKS[p], []).append(str(p))
        print(" ".join(f"{','.join(v)}@0:{n}" for n, v in by.items()))
    elif cmd == "groups" and args[0] == "conf":
        lk = json.load(open(args[1]))
        print(" ".join(f"{p}@0:{n}" for p, n in lk["conf_chunks"].items()))
    elif cmd == "groups" and args[0] == "check":
        print(f"{','.join(str(p) for p in RELAY_P)}@{CHECK_CHUNKS}:{2 * CHECK_CHUNKS}")
    elif cmd == "tags" and args[:1] == ["compiled"]:
        print(" ".join(t for t in EVAL_COMPILED if os.path.exists(NETS[t][0])))
    elif cmd == "tags":
        late = args[:1] == ["late"]
        print(" ".join(t for t, v in NETS.items() if os.path.exists(v[0]) and ((t in LATE) == late)))
    elif cmd == "paths":
        print(",".join(NETS[t][0] for t in args))
    elif cmd == "need":
        lk = json.load(open(args[0]))
        need = [lk["B"]] + sorted({c["S"] for c in lk["configs"]} - {lk["B"]})
        print(" ".join(need))
    else:
        raise SystemExit(main.__doc__)


if __name__ == "__main__":
    main()
