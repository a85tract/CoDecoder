# The sanity read (seed kind 13 only, which no development, test or confirmation block uses): CS13's
# final_ema.pt and budget_ema.pt load (sha256 checked) and decode, beside S on the release's engine and correlated PM, on
# 200,000 shots at 0.6 % (kind 13, chunks 0-19) and 1,000,000 at 0.35 % (chunks 0-99, the earlier dry runs' shots, on which
# correlated PM failed 89 times). B on its graph path (no compile) here; the screen times every path. Sourced by
# dry_full.sh (DRY=1, RUN set).
source $T/surface/eval/jobs/common.sh
O=$E/sanity
run sha256sum $R/ckpt/CS13/final_ema.pt $R/ckpt/CS13/budget_ema.pt
RD="$PY -m surface.sv_read --dry --stage dev --paths graph@1,graph@4096"
run $RD --p 0.006 --chunks 0-20 --comp S1,SB,B1,Bb,MS,MB --ref-s 2000 --out $O/final
run $RD --p 0.0035 --chunks 0-100 --comp SB,Bb,MB --out $O/final
run $RD --ckpt budget --p 0.006 --chunks 0-20 --comp Bb --out $O/budget
run $RD --ckpt budget --p 0.0035 --chunks 0-100 --comp Bb --out $O/budget
