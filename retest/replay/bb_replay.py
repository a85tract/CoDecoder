"""Replay checks of the BB shim (RETEST.md section 6) on the published records of stage 3 (bb72 tree) and stage 4
(gross_stage4 tree). Run with the tree's ising_compare and qldpc_neural on PYTHONPATH, BB_SHOTS and BB_PREDS set.
  unpack OUT_JSON       the published per-shot outputs (test_outputs_*.npz, test4_outputs_G-4RF.npz) -> one npz per
                        decoder and chunk in BB_PREDS, as the reads wrote them; the network reads carry the engine
                        signature they were made with (stage 3: the cell's policy engine with the analysed GPU,
                        cells.json "gpu"; stage 4: cell.json "engine"), and the secs arrays are zeros (the analyses
                        read predictions only; times come from the rows files)
  shots OUT_JSON        the regenerated test chunks in BB_SHOTS against the published array hashes and seeds
  platform TAGS OUT_JSON  the network reads in BB_PREDS (made again on the re-test's GPU) against the published
                        predictions and margins, bit for bit
  compare A B OUT_JSON  two analysis outputs (results*.json, RESULTS*.md): equal, or the differing keys
  rows A B OUT_JSON     an original rows file and its derived copy: every column of every row equal (value_ms as text),
                        the note's original keys unchanged"""
import hashlib
import json
import os
import sys

import numpy as np

from bb import opp, route, shots

D = os.path.join(shots.HERE, "data", "bb")
GROSS = os.path.exists(os.path.join(D, "stage4_gross", "cell.json"))


def published():
    """[(code, p, split, chunks, npz arrays, index entry, {tag: engine signature})] of the tree's published outputs."""
    if GROSS:
        s = os.path.join(D, "stage4_gross")
        c, ix = json.load(open(os.path.join(s, "cell.json"))), json.load(open(os.path.join(s, "outputs_index_G-4RF.json")))
        e = ix["test4"]
        return [(c["code"], c["p"], c["split"], [x["chunk"] for x in e["chunks"]],
                 _load(os.path.join(s, e["file"]), e["sha256"]), e, {c["tag"]: c["engine"]})]
    s = os.path.join(D, "stage3")
    c, ix = json.load(open(os.path.join(s, "cells.json"))), json.load(open(os.path.join(s, "test_outputs_index.json")))
    out = []
    for name, e in ix.items():
        sp = c["splits"][name]
        sigs = {}
        for x in c["cells"].values():
            if x["split"] == name:
                pol = json.load(open(os.path.join(route.POLICIES, x["policy"])))["engine"]
                sigs[x["tag"]] = dict(pol, gpu=c["gpu"])
        out.append((sp["code"], sp["p"], "test", [x["chunk"] for x in e["chunks"]],
                    _load(os.path.join(s, e["file"]), e["sha256"]), e, sigs))
    return out


def _load(path, sha):
    """The published npz, its file sha256 checked against the index."""
    if hashlib.sha256(open(path, "rb").read()).hexdigest() != sha:
        raise SystemExit(f"{path}: sha256 differs from the published index")
    return np.load(path)


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def unpack(out):
    rep = {}
    os.makedirs(opp.PREDS, exist_ok=True)
    for code, p, split, chunks, z, e, sigs in published():
        n = shots.CHUNK
        obs = np.unpackbits(z["obs"], axis=1, count=12)
        for k, j in enumerate(chunks):
            sl = slice(k * n, (k + 1) * n)
            for d in ("relay", "bposd", "gpu_bposd", "relay_full"):
                pred = np.unpackbits(z[f"{d}_pred"][sl], axis=1, count=12)
                extra = {"read": f"published {e['file']}"} if d == "gpu_bposd" else {}
                np.savez(opp.pred_path(code, p, split, j, d), pred=pred, fail=(pred != obs[sl]).any(1),
                         secs=np.zeros(len(pred), np.float32), conv=np.ones(len(pred), bool), **extra)
            for tag, sig in sigs.items():
                pred = np.unpackbits(z[f"net-{tag}_pred"][sl], axis=1, count=12)
                np.savez(route.net_path(code, p, split, j, tag), pred=pred, margin=z[f"net-{tag}_margin"][sl],
                         fail=(pred != obs[sl]).any(1), sig=json.dumps(sig, sort_keys=True),
                         meta=json.dumps({"replay": f"unpacked from {e['file']}"}))
        rep[f"{code}_p{p}_{split}"] = {"file": e["file"], "chunks": len(chunks), "tags": sorted(sigs)}
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(rep), flush=True)


def check_shots(out):
    rep, ok = {}, True
    for code, p, split, chunks, z, e, _ in published():
        bad = []
        for x in e["chunks"]:
            det, obs = shots.load(code, p, split, x["chunk"])
            if (_sha(np.packbits(det, axis=1)), _sha(np.packbits(obs, axis=1)), shots.seed_of(split, code, p, x["chunk"])) \
                    != (x["det_packed_sha256"], x["obs_packed_sha256"], x["seed"]):
                bad.append(x["chunk"])
        rep[f"{code}_p{p}_{split}"] = {"chunks": len(e["chunks"]), "equal": len(e["chunks"]) - len(bad), "differ": bad}
        ok &= not bad
    rep["ok"] = ok
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(rep), flush=True)
    sys.exit(0 if ok else 1)


def platform(tags, out):
    rep, ok = {}, True
    for code, p, split, chunks, z, e, _ in published():
        for tag in tags.split(","):
            if f"net-{tag}_pred" not in z.files:
                continue
            pred = np.concatenate([np.load(route.net_path(code, p, split, j, tag))["pred"] for j in chunks])
            margin = np.concatenate([np.load(route.net_path(code, p, split, j, tag))["margin"] for j in chunks])
            sig = json.loads(str(np.load(route.net_path(code, p, split, chunks[0], tag))["sig"]))
            ref_pred = np.unpackbits(z[f"net-{tag}_pred"], axis=1, count=12)
            ref_margin = z[f"net-{tag}_margin"]
            r = {"shots": len(pred), "decisions_differ": int((pred != ref_pred).any(1).sum()),
                 "margins_bitwise_equal": int((margin.view(np.uint32) == ref_margin.view(np.uint32)).sum()),
                 "margin_absdiff_max": float(np.abs(margin.astype(np.float64) - ref_margin).max()), "gpu": sig["gpu"],
                 "engine": sig}
            r["ok"] = r["decisions_differ"] == 0 and r["margins_bitwise_equal"] == len(pred)
            ok &= r["ok"]
            rep[f"{tag}_{code}_p{p}_{split}"] = r
    rep["ok"] = ok
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps({k: (v if k == "ok" else {x: v[x] for x in ("shots", "decisions_differ", "margins_bitwise_equal",
                                                                  "gpu", "ok")}) for k, v in rep.items()}), flush=True)
    sys.exit(0 if ok else 1)


def _diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [x for k in sorted(set(a) | set(b)) for x in (_diff(a.get(k), b.get(k), f"{path}/{k}")
                                                              if k in a and k in b else [f"{path}/{k}"])]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in _diff(u, v, f"{path}[{i}]")]
    return [] if a == b else [path]


def _get(x, path):
    for part in path.strip("/").split("/"):
        k, _, i = part.partition("[")
        x = x[k] if k else x
        if i:
            x = x[int(i[:-1])]
    return x


def compare(a, b, out):
    """Equal, or the differing keys; a float that differs only beyond 1e-12 relative is listed with both values (the
    last digits of a p value computed with another host's libm), and "equal_to_1e-12" says whether that is all."""
    near = []
    if a.endswith(".json"):
        ja, jb = json.load(open(a)), json.load(open(b))
        diff = _diff(ja, jb)
        for k in diff:
            try:
                x, y = _get(ja, k), _get(jb, k)
            except (KeyError, IndexError, TypeError):
                continue
            if isinstance(x, float) and isinstance(y, float) and abs(x - y) <= 1e-12 * max(abs(x), abs(y)):
                near.append({"key": k, "a": repr(x), "b": repr(y)})
    else:
        diff = [] if open(a).read() == open(b).read() else ["text differs"]
    rep = {"a": a, "b": b, "equal": not diff, "differ": diff[:50], "within_1e-12": near,
           "equal_to_1e-12": len(near) == len(diff)}
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(rep), flush=True)
    sys.exit(0 if rep["equal_to_1e-12"] else 1)


def rows(a, b, out):
    import csv
    ra, rb = list(csv.DictReader(open(a))), list(csv.DictReader(open(b)))
    bad = []
    for i, (x, y) in enumerate(zip(ra, rb)):
        nx, ny = (json.loads(r["note"]) if r.get("note") else {} for r in (x, y))
        if any(x[k] != y[k] for k in x if k != "note") or any(ny.get(k) != v for k, v in nx.items()):
            bad.append(i)
    rep = {"a": a, "b": b, "rows": len(ra), "equal": len(ra) == len(rb) and not bad, "differ": bad[:50]}
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(rep), flush=True)
    sys.exit(0 if rep["equal"] else 1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "unpack":
        unpack(sys.argv[2])
    elif cmd == "shots":
        check_shots(sys.argv[2])
    elif cmd == "platform":
        platform(sys.argv[2], sys.argv[3])
    elif cmd == "compare":
        compare(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "rows":
        rows(sys.argv[2], sys.argv[3], sys.argv[4])
