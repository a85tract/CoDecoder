#!/bin/bash
# Run on a compute host: clone the Relay-BP repository at the pinned revision (reference circuits, testdata helpers).
set -e
DIR=${QND_RELAY:-$HOME/work/relay}
[ -d "$DIR" ] || git clone -q https://github.com/trmue/relay.git "$DIR"
cd "$DIR" && git fetch -q && git checkout -q d185194ba0cb4101ced4340d82b2ee6d42f225f0 && git log -1 --format='%H %ad'
