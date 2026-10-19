"""Put the tables arb5_analysis.py wrote (out/arb5_tables.md, blank-line separated: the main table, the comparison with
the paper's row, the fit half, the timing) into RESULTS.md's placeholders, so no number is copied by hand.

    python fill_results.py RESULTS.md out/arb5_tables.md"""
import sys

doc, tables = sys.argv[1], sys.argv[2]
parts = [p.strip() for p in open(tables).read().split("\n\n") if p.strip()]
assert len(parts) == 4, f"{len(parts)} tables, 4 expected"
s = open(doc).read()
for key, t in zip(("MAIN_TABLE", "REG_TABLE", "FIT_TABLE", "TIME_TABLE"), parts):
    assert s.count(key) == 1, key
    s = s.replace(key, t)
open(doc, "w").write(s)
