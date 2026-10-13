"""X2 BB72: where S55'-deep (ConvLines-BB, six blocks) and C14' (our re-implementation of Cascade) differ on the D1
test's shots (cascade_compare/phase_d/D1_RESULT.md on ising-cascade-d1): p = 0.2 % (11,000,000 shots, q1 = 2 %) and
0.3 % (17,000,000 shots, q1 = 10 %), setting depspam_fm.

Inputs, read-only: D1's eager reads in the Alpine archive (cascade_d1/out/read/p*.npz and .json, not published), which hold every shot's detector rows, observables and both networks' outputs, and for
the detector layout the circuit, rebuilt with the code of D1's freeze tree (F = 310ce8e0) and checked against the
read's circuit sha256. The rows are checked against the read's per-chunk detector digests before use.

  python bb_d1.py layout                 # the circuit's detector coordinates, per p
  python bb_d1.py run --procs 8          # per-shot features -> $X2_OUT/bb/p0.00{2,3}00.npz (resumable per p)"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ARCH = os.environ.get("X2_D1_READS", "<archive>/cascade_d1/out")    # D1's per-shot reads (not published)
D1_TREE = os.environ.get("X2_D1_TREE", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "cascade", "bb72"))           # ising_compare and qldpc_neural; the circuit sha256 is checked
RELAY = os.environ.get("QND_RELAY", "<relay-bp>")                # relay-bp's repository at d185194 (the circuits)
OUT = os.path.join(os.environ.get("X2_OUT", "out"), "bb")
K, NDET, CHUNK = 12, 432, 10_000
NPZ_SHA = {0.002: "ceb3ab405dc490dea9599ae02aa5095f923d066a9649c79e2517babe59d779a3",
           0.003: "360e9fb38c0df4c49d219aecad47a1ba52f417992393c7e6f0a0cfe6db001321"}
TAU1 = {0.002: 7.3331888389587405, 0.003: 7.60439167022705}
# D1_RESULT.md, eager path: routed; S55'-deep alone and C14' failures; kept breaks / repairs / both fail
REG = {0.002: dict(routed=220058, S=10348, C=7922, breaks=19, repairs=86, both=238),
       0.003: dict(routed=1701544, S=112636, C=76730, breaks=81, repairs=342, both=1977)}
POP8 = np.array([bin(x).count("1") for x in range(256)], np.uint8)
REF_EVERY = 1000


def tag(p):
    return f"p{p:.5f}"


def circuit(p, want):
    sys.path[:0] = [f"{D1_TREE}/ising_compare", f"{D1_TREE}/qldpc_neural"]
    os.environ["QND_RELAY"] = RELAY
    from bb.cascade_data import circuit_at
    c = circuit_at("bb72", p, "depspam_fm")
    assert hashlib.sha256(str(c).encode()).hexdigest() == want, f"p = {p}: circuit is not the read's"
    return c


def layout(c):
    """(layer, site, is_x) per detector. The circuit's coordinates are one index each, so the layout is read from the
    measurement records, as scripts/make_circuits.py of the release does for the surface code: a detector's site is
    the ancilla of its latest ancilla record and its layer that ancilla's measurement count (detectors that also
    compare final data measurements get one more); X type = that ancilla is measured with MX."""
    import stim
    meas = []                                                    # (qubit, kind, count of that qubit so far)
    cnt = {}
    for inst in c.flattened():
        if stim.gate_data(inst.name).produces_measurements:
            for t in inst.targets_copy():
                meas.append((t.value, inst.name, cnt.get(t.value, 0)))
                cnt[t.value] = cnt.get(t.value, 0) + 1
    anc = {q for q, k in cnt.items() if k > 1}
    r, site, is_x = [], [], []
    seen = 0
    for inst in c.flattened():
        if stim.gate_data(inst.name).produces_measurements:
            seen += len(inst.targets_copy())
        elif inst.name == "DETECTOR":
            recs = [seen + t.value for t in inst.targets_copy()]
            a = max(k for k in recs if meas[k][0] in anc)
            r.append(meas[a][2] + int(any(meas[k][0] not in anc for k in recs)))
            site.append(meas[a][0])
            is_x.append(meas[a][1] in ("MX", "MRX"))
    assert len(r) == NDET
    return np.array(r), np.array(site), np.array(is_x, bool)


def load(p):
    path = f"{ARCH}/read/{tag(p)}.npz"
    raw_sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert raw_sha == NPZ_SHA[p], f"{path}: not D1's recorded file"
    meta = json.load(open(f"{ARCH}/read/{tag(p)}.json"))
    z = np.load(path)
    names = [os.path.basename(os.path.dirname(m)) for m in z["models"]]
    iC, iS = names.index("C14_dfm"), names.index("S55_dfm_deep")
    return z, meta, iC, iS


def det_digests_ok(det, chunks, meta):
    """d1_common.chunk_digests: every chunk's rows (packbits little-endian) against the read's chunk_meta."""
    want = {c["chunk"]: c["det_sha256"] for c in meta["chunk_meta"]}
    order = [c["chunk"] for c in meta["chunk_meta"]]
    lo = 0
    for j in order:
        hi = lo + CHUNK
        assert np.all(chunks[lo:hi] == j), f"rows of chunk {j} are not contiguous"
        d = np.unpackbits(det[lo:hi], axis=1, count=NDET, bitorder="little")
        assert hashlib.sha256(np.packbits(d, axis=1).tobytes()).hexdigest()[:16] == want[j], f"chunk {j} digest"
        lo = hi
    assert lo == len(det)
    return len(order)


def run_point(p, lay):
    os.makedirs(OUT, exist_ok=True)
    out = f"{OUT}/{tag(p)}.npz"
    if os.path.exists(out):
        print(f"{out} exists, skipped")
        return
    t0 = time.time()
    z, meta, iC, iS = load(p)
    det, chunks = z["det"], z["chunk"]
    n_chunks = det_digests_ok(det, chunks, meta)
    print(f"p = {p}: {len(det):,} rows, {n_chunks} chunk digests ok ({time.time() - t0:.0f} s)", flush=True)
    obs = z["obs"]
    fC, fS = z[f"fail{iC}"].astype(bool), z[f"fail{iS}"].astype(bool)
    confS = z[f"conf{iS}"].astype(np.float64)
    confC = z[f"conf{iC}"].astype(np.float64)
    routed = confS < TAU1[p]
    keep = ~routed
    cj = json.load(open(f"{ARCH}/compiled/{tag(p)}.json"))
    assert cj["tau1"] == TAU1[p], "tau1 differs from the compiled read's"
    reg = REG[p]
    got = dict(routed=int(routed.sum()), S=int(fS.sum()), C=int(fC.sum()), breaks=int((fS & ~fC & keep).sum()),
               repairs=int((~fS & fC & keep).sum()), both=int((fS & fC & keep).sum()))
    assert got == reg, f"p = {p}: counts {got} are not D1's {reg}"
    # the compiled single and batched C14' paths (the timed computation), for the kept repairs and breaks
    cz = np.load(f"{ARCH}/compiled/{tag(p)}.npz")
    comp = {}
    for path, key in (("single", "bits1"), ("batched", "bits4")):
        fCc = (cz[key] != obs).any(1)
        comp[path] = dict(C=int(fCc.sum()), breaks=int((fS & ~fCc & keep).sum()),
                          repairs=int((~fS & fCc & keep).sum()), both=int((fS & fCc & keep).sum()))
    n_det = POP8[det].sum(1, dtype=np.int32)
    wrongC = POP8[z[f"pred{iC}"] ^ obs].sum(1, dtype=np.int8)
    wrongS = POP8[z[f"pred{iS}"] ^ obs].sum(1, dtype=np.int8)
    n_flip = POP8[obs].sum(1, dtype=np.int8)
    # every failure of either network, and every 1000th shot as the reference sample
    sel = np.nonzero(fS | fC | (np.arange(len(obs)) % REF_EVERY == 0))[0]
    r, site, is_x = lay
    nr = r.max() + 1
    bits = np.unpackbits(det[sel], axis=1, count=NDET, bitorder="little")
    per_round = np.stack([bits[:, (r == k)].sum(1) for k in range(nr)], 1).astype(np.int16)
    per_round_x = np.stack([bits[:, (r == k) & is_x].sum(1) for k in range(nr)], 1).astype(np.int16)
    # time-like pairs: the same check fired in consecutive detector layers
    nxt = {}
    for i in range(NDET):
        nxt[(site[i], r[i])] = i
    tp = [(i, nxt[(site[i], r[i] + 1)]) for i in range(NDET) if (site[i], r[i] + 1) in nxt]
    tp = np.array(tp, np.int64)
    tpairs = (bits[:, tp[:, 0]] & bits[:, tp[:, 1]]).sum(1).astype(np.int16)
    zC = z[f"z{iC}"][sel].astype(np.float32)                    # C14''s 12 logits on the selected shots
    errC = np.unpackbits(z[f"pred{iC}"][sel] ^ obs[sel], axis=1, count=K)  # logicals C14' got wrong (big-endian)
    errS = np.unpackbits(z[f"pred{iS}"][sel] ^ obs[sel], axis=1, count=K)
    np.savez_compressed(out + ".tmp.npz", p=p, n=len(obs), routed=routed, fS=fS, fC=fC, confS=confS.astype(np.float32),
                        confC=confC.astype(np.float32), n_det=n_det, wrongC=wrongC, wrongS=wrongS, n_flip=n_flip,
                        sel=sel, per_round=per_round, per_round_x=per_round_x, tpairs=tpairs, zC=zC, errC=errC,
                        errS=errS, obs_sel=np.unpackbits(obs[sel], axis=1, count=K), clsS=z[f"cls{iS}"][sel],
                        comp=json.dumps(comp), counts=json.dumps(got), n_tp=len(tp))
    os.replace(out + ".tmp.npz", out)
    print(f"p = {p}: counts as D1 {got}; compiled {comp}; {len(sel):,} selected; {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("layout", "run"))
    a = ap.parse_args()
    for p in (0.002, 0.003):
        meta = json.load(open(f"{ARCH}/read/{tag(p)}.json"))
        c = circuit(p, meta["circuit_sha256"])
        if a.what == "layout":
            co = c.get_detector_coordinates()
            arr = [co[i] for i in range(c.num_detectors)]
            dims = sorted({len(v) for v in arr})
            print(f"p = {p}: {c.num_detectors} detectors, {c.num_observables} observables; coordinate lengths {dims}")
            if dims and dims[0] > 0:
                A = np.array([v[:min(dims)] for v in arr], float)
                for k in range(A.shape[1]):
                    u = np.unique(A[:, k])
                    print(f"  dim {k}: {len(u)} values {u[:12].tolist()}{' ...' if len(u) > 12 else ''}")
                print("  first:", A[:4].tolist(), "last:", A[-4:].tolist())
            r, site, is_x = layout(c)
            print(f"  layers {r.max() + 1}: detectors per layer {np.bincount(r).tolist()}, X-type per layer "
                  f"{np.bincount(r[is_x], minlength=r.max() + 1).tolist()}; sites {len(np.unique(site))}")
            continue
        run_point(p, layout(c))


if __name__ == "__main__":
    main()
