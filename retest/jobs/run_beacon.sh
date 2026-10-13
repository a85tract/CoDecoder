#!/bin/bash
# After GitHub's push time is known (RETEST.md section 1, steps 2-3), on Alpine's login node (the compute nodes have
# no internet): the beacon (waits for the round, then fetches every endpoint) and the seeds, each written once.
#   bash retest/jobs/run_beacon.sh PUSH_TIME FREEZE_COMMIT      PUSH_TIME = GitHub's, ISO 8601 UTC
set -euo pipefail
T=${T:-<cluster-scratch>/retest/code/freeze}; RUN=${RUN:-<cluster-scratch>/retest/run}
mkdir -p $RUN && cd $T
/usr/bin/python3 retest/beacon.py --push-time "$1" --commit "$2" --out $RUN/beacon.json
/usr/bin/python3 retest/seeds.py $RUN/beacon.json $RUN/seeds.json
chmod a-w $RUN/beacon.json $RUN/seeds.json
