"""Phase D1's speed wrapper (PHASE_D1_PLAN.md 4.1 and 5.2): milestone 2's m2.py (cascade_compare/batchopt, 48150303),
unchanged, with D1's lock in place of Amendment 1's. m2.frozen_speed names Amendment 1's locks; here it returns phase
C's frozen speed components and {"depspam_fm": D1's lock}, after checking that the lock froze those components, as m2.py
itself replaces pc.gpu_env for the H100. Then m2.main() runs the subcommand (D1-S uses only pipeline):

  python d1_m2.py pipeline --setting depspam_fm --config I --p P --mode single|batched --engine base|new --tail-b 0 \
      --rep R --n N --out rows.jsonl          (--config -1: Cascade alone at --p)

D1_LOCK names another lock (the dry runs' dry lock)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "batchopt"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase_c"))
import m2  # noqa: E402
import pc_common as pc  # noqa: E402

LOCK = os.environ.get("D1_LOCK") or os.path.join(HERE, "d1_locked_depspam_fm.json")


def frozen_speed():
    """Phase C's frozen speed components and D1's lock (read only); the lock names the components' sha256."""
    sp = os.path.join(pc.W4, "out", "speed", "speed_components.json")
    if pc.sha256_file(sp) != json.load(open(LOCK))["speed_sha256"]:
        raise SystemExit(f"{sp}: not the speed components frozen in {LOCK}")
    return json.load(open(sp)), {"depspam_fm": LOCK}


m2.frozen_speed = frozen_speed

if __name__ == "__main__":
    m2.main()
