#!/bin/bash
cd /workspace/repo && . .venv/bin/activate
mkdir -p logs shots/T2 shots/T3 shots/val
( python -m convlines.testset regen --set T2 --d 7 --chunks 0-199 --out shots/T2; python -m convlines.testset val --d 7 --out shots/val ) > logs/regen_d7.log 2>&1 &
for d in 9 11 13; do ( python -m convlines.testset regen --set T2 --d $d --chunks 0-99 --out shots/T2; python -m convlines.testset val --d $d --out shots/val ) > logs/regen_d$d.log 2>&1 & done
( python -m convlines.testset regen --set T3 --d 15 --chunks 0-99 --out shots/T3; python -m convlines.testset val --d 15 --out shots/val ) > logs/regen_d15.log 2>&1 &
wait
echo REGEN ALL DONE
