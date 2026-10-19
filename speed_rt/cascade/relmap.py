"""relmap.py: ARB3's unregistered timing of the joint decision (2026-10-19) runs the release's tree (convlines_public at
94aaaa7 with arb-prod), because the pinned cluster trees and checkpoints were removed from Alpine's scratch. srt_cas.py
imports it when SRT_RELMAP=1, in the main process and in every spawned worker, before the harness. Like ARB3's pod
launcher (pod/podrun.py), it changes from outside the tree only what the release's sanitized tree cannot hold:
  - pc_common's checkpoint paths of C14p and S55p_deep become the release's weights/cascade_bb72/{C14p,S55p_deep}.pt,
    and W4 becomes SRT_RELMAP_W4. That directory holds the release's speed_components.json (its sha256 is the D1 lock's
    speed_sha256) and C4b's read, regenerated on the card for m2.py's in-process S check.
  - bb.cascade_eval.sha256_file returns, for those two files, the source_sha256 each one carries: the sha256 of the
    checkpoint that the D1 lock names. The release re-saved them without changing any tensor. The file is first checked
    against the release's weights/MANIFEST.sha256.
The card guard stays srt_cas.py's: the RTX PRO 6000 by the harness, and the H100 with SRT_CARD=h100.
Environment: SRT_RELMAP_RELEASE (the release root), SRT_RELMAP_W4."""
import os
import re
import zipfile

FILES = {"C14p": "C14p.pt", "S55p_deep": "S55p_deep.pt"}


def manifest(rel):
    out = {}
    for ln in open(os.path.join(rel, "weights", "MANIFEST.sha256")):
        h, f = ln.split()
        out[os.path.realpath(os.path.join(rel, "weights", f))] = h
    return out


def source_sha(path):
    z = zipfile.ZipFile(path)
    raw = z.read(next(n for n in z.namelist() if n.endswith("data.pkl")))
    m = re.search(rb"source_sha256.{1,24}?([0-9a-f]{64})", raw, re.S)
    if not m:
        raise SystemExit(f"relmap: {path} carries no source_sha256")
    return m.group(1).decode()


def install():
    import pc_common as pc
    from bb import cascade_eval as ce
    if getattr(ce.sha256_file, "_relmap", False):
        return
    rel = os.environ["SRT_RELMAP_RELEASE"]
    wts = os.path.join(rel, "weights", "cascade_bb72")
    for t, f in FILES.items():
        pc.NETS[t] = (os.path.join(wts, f),) + tuple(pc.NETS[t][1:])
    pc.W4 = os.environ["SRT_RELMAP_W4"]
    man, mapped, orig = manifest(rel), {}, ce.sha256_file
    ours = {os.path.realpath(os.path.join(wts, f)) for f in FILES.values()}

    def sha256_file(path):
        rp = os.path.realpath(path)
        if rp in ours:
            if rp not in mapped:
                own = orig(path)
                if man.get(rp) != own:
                    raise SystemExit(f"relmap: {path} sha256 {own} is not the release manifest's")
                mapped[rp] = source_sha(path)
            return mapped[rp]
        return orig(path)
    sha256_file._relmap = True
    ce.sha256_file = sha256_file
