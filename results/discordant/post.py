"""X2: the steps after the per-chunk and BB runs, in order: join the surface chunks and the BB72 files, write the
tables, draw the figures.

  python post.py"""
import sys

import bb_join
import figures
import surface_join
import tables

if __name__ == "__main__":
    sys.argv = ["surface_join.py"]
    surface_join.main()
    bb_join.main()
    tables.main()
    figures.main()
