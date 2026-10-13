# Phase C Relay-BP (PHASE_C_PLAN.md section 4): pc_relay.py on one read's routed union. Sourced with MODE (dev|conf),
# READ, OUTR and, for conf, LOCKED; optional PS (default RELAY_P) and EXTRA (e.g. --max-shots N, dry runs only).
$PYR $PC/pc_relay.py --read $READ --mode $MODE ${LOCKED:+--locked $LOCKED} ${PS:+--p $PS} --procs $SLURM_CPUS_ON_NODE \
    --out $OUTR ${EXTRA:-}
