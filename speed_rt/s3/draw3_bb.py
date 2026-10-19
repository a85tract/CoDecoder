"""SPEED-RT stage 3's BB draw (STAGE3_batched.md section 4): speed_rt/bb/srt_bb.py unchanged, with this stage's sets file
(speed_rt/s3/sets_s3.json) in place of stage 2's, so that chunk j of s3-bb72-p2, s3-bb72-p3 and s3-gross-p4 gets
speed_rt.sets.seed(R_3, set, j). Run as srt_bb.py is run (SRT_BEACON = stage 3's beacon record, in the BB tree):

    python speed_rt/s3/draw3_bb.py bb.shots --code bb72 --p 0.003 --split test --chunks 0-9 --rows M.csv
    python speed_rt/s3/draw3_bb.py --check-manifest M.csv... --out OUT.json"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bb"))
import srt_bb  # noqa: E402

srt_bb.SETS_FILE = os.path.join(HERE, "sets_s3.json")

if __name__ == "__main__":
    sys.exit(srt_bb.main())
