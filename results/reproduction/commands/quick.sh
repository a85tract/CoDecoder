#!/bin/bash
cd /workspace/repo && . .venv/bin/activate && export ISING_DECODING_DIR=/workspace/Ising-Decoding
mkdir -p out
{ echo "== weights manifest"; (cd weights && sha256sum -c MANIFEST.sha256); } > out/manifest.log 2>&1
{ echo "== circuits sha check"; python -c "
from convlines import circuits
for d in (7,9,11,13,15):
    c = circuits.circuit(d); print(d, 'ok', c.num_detectors, c.num_observables)
"; } > out/circuits.log 2>&1
{ echo "== make_circuits"; sha256sum circuits/*.stim weights/grid_d*.npz > out/circuits_before.sha; mkdir -p out/mc && python scripts/make_circuits.py --help; } > out/make_circuits.log 2>&1
{ echo "== composite"; python -m timing.composite; } > out/composite.log 2>&1
{ echo "== figure"; sha256sum figure/*.png; mkdir -p out/fig_committed && cp figure/*.png out/fig_committed/; python figure/acc_speed.py; sha256sum figure/*.png; cmp figure/accuracy_vs_speed_grace.png out/fig_committed/accuracy_vs_speed_grace.png && echo GRACE PNG IDENTICAL; cmp figure/accuracy_vs_speed_m4.png out/fig_committed/accuracy_vs_speed_m4.png && echo M4 PNG IDENTICAL; } > out/figure.log 2>&1
echo QUICK DONE
