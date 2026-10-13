# Phase C, the analysis (PHASE_C_PLAN.md section 6): every table of the final report and the mechanical verdicts.
# Optional: O (outputs, default $R/out), ADDON=1 (the C6' add-on reads), SP, SPL (a later timing of C6', descriptive),
# PAPER (the paper's digitised timings, default $O/paper_timing.json when it exists: C8 digitises them after the
# deployment, as the runbook set it), RO (report directory).
O=${O:-$R/out}; SPL=${SPL:-$O/speed_late}; PAPER=${PAPER:-$O/paper_timing.json}
$PY $PC/pc_report.py --curves uniform=$O/dev/uniform${ADDON:+,$O/dev_late/uniform} \
    --curves depspam_fm=$O/dev/depspam_fm${ADDON:+,$O/dev_late/depspam_fm} \
    $(for S in uniform depspam_fm; do [ -f $O/confirm_$S.json ] && echo --confirm $O/confirm_$S.json; done) \
    $(for S in uniform depspam_fm; do [ -f $O/check_$S.json ] && echo --check $O/check_$S.json; done) \
    $(for S in uniform depspam_fm; do [ -f $O/locked_$S.json ] && echo --locked $O/locked_$S.json; done) \
    $([ -f $SPL/speed_components.json ] && echo --speed-late $SPL/speed_components.json) \
    $([ -f $PAPER ] && echo --paper-timing $PAPER) \
    --speed ${SP:-$O/speed}/speed_components.json --speed-rows ${SP:-$O/speed}/rows.jsonl --out ${RO:-$O/report}
