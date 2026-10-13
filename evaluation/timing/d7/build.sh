#!/bin/bash
# Builds the d = 7 executable into OUT (default build/d7):
#   kernel_sm90a.cubin  kernel.cu for sm_90a (H100 / H200): nvcc -O3 -std=c++17 -arch=sm_90a -cubin
#   libs7router.so      router.cc + PyMatching v2.4.0's C++ sources (the wheel's SOURCE_FILES_NO_MAIN) + Stim at
#                       1320ad7e (the commit PyMatching 2.4.0's CMake pins), each fetched from GitHub at that commit and
#                       compiled with the wheel's flags (CMake Release on x86_64: C++20, -O3 -mno-avx2, NDEBUG; Stim also
#                       -fno-strict-aliasing -msse2) and -ffp-contract=off, linked into one shared library: libstdc++ /
#                       libgcc static and hidden, only the s7_* / s9_* C API exported, internal references bound locally
#                       (-Bsymbolic), so it loads beside the wheel's own copy of PyMatching and any host libstdc++.
#   bash timing/d7/build.sh [OUT] [JOBS]        needs git, g++ >= 10, nvcc >= 12.0; CXX and NVCC override the compilers
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); mkdir -p "${1:-build/d7}"; O=$(cd "${1:-build/d7}" && pwd); J=${2:-8}
CXX=${CXX:-g++}; NVCC=${NVCC:-nvcc}
PM=$O/src/PyMatching; ST=$O/src/Stim
if [ ! -f "$PM/COMMIT" ]; then
  git clone -q --depth 1 --branch v2.4.0 https://github.com/oscarhiggott/PyMatching "$PM"
  git -C "$PM" rev-parse HEAD > "$PM/COMMIT"
fi
if [ ! -f "$ST/COMMIT" ]; then
  mkdir -p "$ST" && git -C "$ST" init -q && git -C "$ST" remote add origin https://github.com/quantumlib/Stim
  git -C "$ST" fetch -q --depth 1 origin 1320ad7eac7de34d2e9c70daa44fbc6d84174450
  git -C "$ST" checkout -q FETCH_HEAD && git -C "$ST" rev-parse HEAD > "$ST/COMMIT"
fi
[ "$(cat "$PM/COMMIT")" = 6f63b2b9474ba0fa7e511fe52bffdce858a06984 ] || { echo "PyMatching is not v2.4.0"; exit 1; }
[ "$(cat "$ST/COMMIT")" = 1320ad7eac7de34d2e9c70daa44fbc6d84174450 ] || { echo "Stim is not 1320ad7e"; exit 1; }

"$NVCC" -O3 -std=c++17 -arch=sm_90a -cubin -o "$O/kernel_sm90a.cubin" "$H/kernel.cu"

mkdir -p "$O/obj/stim" "$O/obj/pm"
COMMON=(-std=c++20 -O3 -DNDEBUG -fPIC -mno-avx2 -ffp-contract=off -fvisibility=hidden -fvisibility-inlines-hidden)
SFLAGS=("${COMMON[@]}" -fno-strict-aliasing -msse2 -w "-I$ST/src")
PFLAGS=("${COMMON[@]}" "-I$PM/src" "-I$ST/src")
PM_SRC="driver/namespaced_main.cc driver/mwpm_decoding.cc flooder/graph.cc flooder/detector_node.cc
  flooder_matcher_interop/compressed_edge.cc flooder/graph_fill_region.cc flooder/match.cc flooder/graph_flooder.cc
  matcher/alternating_tree.cc matcher/mwpm.cc flooder_matcher_interop/region_edge.cc
  flooder_matcher_interop/mwpm_event.cc tracker/flood_check_event.cc diagram/animation_main.cc diagram/mwpm_diagram.cc
  search/search_graph.cc search/search_detector_node.cc search/search_flooder.cc driver/user_graph.cc"
cc() {   # cc OBJDIR ROOT FILE FLAGS...: ROOT/FILE -> OBJDIR/FILE (with / as _).o in the background, at most J at a time
  while [ "$(jobs -rp | wc -l)" -ge "$J" ]; do wait -n; done
  $CXX "${@:4}" -c "$2/$3" -o "$1/${3//\//_}.o" &
}
while read -r f; do cc "$O/obj/stim" "$ST" "$f" "${SFLAGS[@]}"; done < <(grep -v '^\s*$' "$ST/file_lists/source_files_no_main")
for f in $PM_SRC; do cc "$O/obj/pm" "$PM/src/pymatching" "sparse_blossom/$f" "${PFLAGS[@]}"; done
cc "$O/obj/pm" "$PM/src/pymatching" rand/rand_gen.cc "${PFLAGS[@]}"
for p in $(jobs -p); do wait "$p"; done                         # a failed compile stops the script here (set -e)
$CXX "${PFLAGS[@]}" -c "$H/router.cc" -o "$O/obj/router.o"
printf '{ global: s7_*; s9_*; local: *; };\n' > "$O/exports.map"
$CXX -shared -o "$O/libs7router.so" "$O/obj/router.o" "$O"/obj/pm/*.o "$O"/obj/stim/*.o -static-libstdc++ -static-libgcc \
  -Wl,--exclude-libs,ALL -Wl,-Bsymbolic -Wl,--version-script="$O/exports.map" -ldl -pthread
"$NVCC" --version | tail -1; $CXX --version | head -1
sha256sum "$O/kernel_sm90a.cubin" "$O/libs7router.so"
