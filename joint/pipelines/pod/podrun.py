"""ARB3's pod launcher for the BB72 scripts (checks and smoke timings only, never a registered run):

    python podrun.py SCRIPT.py ARGS...

It changes, from outside the tree, in this process and in every spawned worker (which re-runs this file as
__mp_main__), three things and nothing else:
  - paths: pc_common's cluster placeholders become the pod's. NETS[C14p] and NETS[S55p_deep] are the release's
    weights/cascade_bb72/C14p.pt and S55p_deep.pt; W4 is POD_MAP_W4, which holds the release's speed_components.json
    (its sha256 is the D1 lock's speed_sha256) and C4b's read of depspam_fm regenerated on the pod (kind 12, chunks
    10-19, both p: m2.py's in-process S check reads it).
  - the identity of the release's weights: bb.cascade_eval.sha256_file returns, for those two files, the source_sha256
    each one carries (the sha256 of the checkpoint the studies read and the D1 lock names: the release re-saved them
    as inference copies with no tensor changed, SANITIZE.md), after checking the file's own sha256 against the
    release's weights/MANIFEST.sha256; sv_common.load_B does the same for CS13.
  - the card: phase C's guard (pc_common.GPU, a full RTX PRO 6000) admits the pod's card, POD_GPU or nvidia-smi's
    name (here an H100 80GB HBM3: the RTX PRO 6000 was sold out; checks only), and every row records it.
Environment: POD_RELEASE (the release root), POD_MAP_W4."""
import os
import re
import runpy
import subprocess
import sys
import zipfile

REL = os.environ.get("POD_RELEASE", "/workspace/arb3/release")
WTS = os.path.join(REL, "weights", "cascade_bb72")
FILES = {"C14p": "C14p.pt", "S55p_deep": "S55p_deep.pt"}


def manifest():
    out = {}
    for ln in open(os.path.join(REL, "weights", "MANIFEST.sha256")):
        h, f = ln.split()
        out[os.path.realpath(os.path.join(REL, "weights", f))] = h
    return out


def source_sha(path):
    z = zipfile.ZipFile(path)
    raw = z.read(next(n for n in z.namelist() if n.endswith("data.pkl")))
    m = re.search(rb"source_sha256.{1,24}?([0-9a-f]{64})", raw, re.S)
    if not m:
        raise SystemExit(f"podrun: {path} carries no source_sha256")
    return m.group(1).decode()


def install():
    import pc_common as pc
    from bb import cascade_eval as ce
    if getattr(ce.sha256_file, "_pod", False):
        return
    for t, f in FILES.items():
        v = pc.NETS[t]
        pc.NETS[t] = (os.path.join(WTS, f),) + tuple(v[1:])
    pc.W4 = os.environ.get("POD_MAP_W4", "/workspace/arb3/w4")
    gpu = os.environ.get("POD_GPU") or subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                                      capture_output=True, text=True).stdout.strip().splitlines()[0]
    pc.GPU = gpu
    man = manifest()
    mapped = {}
    orig = ce.sha256_file

    def sha256_file(path):
        rp = os.path.realpath(path)
        if rp in {os.path.realpath(os.path.join(WTS, f)) for f in FILES.values()}:
            if rp not in mapped:
                own = orig(path)
                if man.get(rp) != own:
                    raise SystemExit(f"podrun: {path} sha256 {own} is not the release manifest's")
                mapped[rp] = source_sha(path)
            return mapped[rp]
        return orig(path)
    sha256_file._pod = True
    ce.sha256_file = sha256_file


install()

if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1].endswith(".py"):
        raise SystemExit("usage: podrun.py SCRIPT.py ARGS...")
    script = os.path.abspath(sys.argv[1])
    sys.argv = [script] + sys.argv[2:]
    sys.path.insert(0, os.path.dirname(script))
    runpy.run_path(script, run_name="__main__")
