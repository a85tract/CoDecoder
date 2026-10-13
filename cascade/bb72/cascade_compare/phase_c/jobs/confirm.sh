# Phase C, the confirmation read (PHASE_C_PLAN.md section 4), once per setting, after locked_$SETTING.json is committed:
# the networks the lock needs on fresh rows of seed kind 11 (KIND), the chunk counts the lock fixed per p.
LOCKED=${LOCKED:-$R/out/locked_$SETTING.json}
KIND=${KIND:-11}; OUT=${OUT:-$R/out/conf/$SETTING}
PGROUPS=${PGROUPS:-$($PY $PC/pc_common.py groups conf $LOCKED)}
TAGS=${TAGS:-$($PY $PC/pc_common.py need $LOCKED)}
echo "locked $(sha256sum $LOCKED)"
source $PC/jobs/curves.sh
