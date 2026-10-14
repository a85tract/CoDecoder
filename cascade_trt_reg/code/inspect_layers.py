"""CASCADE-TRT: summary of the TensorRT build rows and of each engine's layers (type, tactic, input format).
    python inspect_layers.py PLANS_DIR [N]"""
import collections
import json
import os
import sys

D = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
for ln in open(os.path.join(D, "build.jsonl")):
    r = json.loads(ln)
    print(r["net"], r["batch"], r["prec"], "ws", r.get("ws_gb", 24), "layers", r["engine_layers"], r["out_formats"],
          r["layer_types"], "build_s", r["tried"][0]["build_s"])
for f in sorted(os.listdir(D)):
    if not f.endswith(".layers.json"):
        continue
    L = json.load(open(os.path.join(D, f))).get("Layers", [])
    c = collections.Counter()
    for ly in L:
        if isinstance(ly, dict):
            ins = [i.get("Format/Datatype", "?")[:34] for i in ly.get("Inputs", [])]
            c[(ly.get("LayerType"), (ly.get("TacticName") or "")[:70], ins[0] if ins else "")] += 1
    print("==", f, len(L))
    for k, v in c.most_common(N):
        print(" ", v, k)
