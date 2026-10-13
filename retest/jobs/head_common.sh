# Sourced by the head_*.sh args files (jobs/head_job.sbatch has set A, RT, RT_*, OUT, PIN, PY, PYS, SHIM and the
# environment). The test sets' chunk lists (the dry run draws, reads and decodes one chunk per code), the original
# calibration chunks and reads, the frozen lists, and their checks.
H=$RT/retest/head                  # networks.sha256, cal_reads.sha256, taus.csv, check_taus.py (frozen with the shim)
if [ -n "${RT_DRY:-}" ]; then CH_surface=0-0; CH_bb72=0-0; else CH_surface=0-99; CH_bb72=0-49; fi
# cal_links: the original calibration chunks (surface cal 0-19, BB72 a1cal 0-19; their rows, with the sha256 that a1.sets.load
# checks on every load, are head/cal_chunks.csv: A1's rows.csv without its test rows; the pinned tree has no manifest) and the 23 original calibration reads (cal_reads.sha256)
# are symlinked, never copied or rewritten; the originals are not touched. Run once, by the draw job.
cal_links() {
  mkdir -p $OUT/sets $OUT/reads/surface $OUT/reads/bb72
  for j in $(seq 0 19); do
    for f in a1_surface_cal_$(printf %03d $j).npz a1_bb72_a1cal_$(printf %03d $j).npz; do
      [ -e $A/sets/$f ] || { echo "no $A/sets/$f"; return 1; }
      ln -sfn $A/sets/$f $OUT/sets/$f
    done
  done
  while read -r sha f; do
    [ -e $A/reads/$f ] || { echo "no $A/reads/$f"; return 1; }
    ln -sfn $A/reads/$f $OUT/reads/$f
  done < $H/cal_reads.sha256
}
# cal_check: every calibration chunk and read is there and the reads have the frozen sha256
cal_check() {
  [ "$(ls $OUT/sets | grep -c -E '^a1_(surface_cal|bb72_a1cal)_[0-9]{3}\.npz$')" = 40 ] || { echo "CAL CHECK: not the 40 calibration chunks"; return 1; }
  awk -F, 'NR > 1 {print $2 "  " $1}' $H/cal_chunks.csv | (cd $OUT/sets && sha256sum --quiet --strict -c -) || { echo "CAL CHECK: a calibration chunk differs from the original"; return 1; }
  (cd $OUT/reads && sha256sum --quiet --strict -c $H/cal_reads.sha256) || { echo "CAL CHECK: a calibration read differs from the original"; return 1; }
  echo "CAL CHECK: 40 calibration chunks and 23 calibration reads equal the original sha256"
}
# net_check: the 23 networks have the sha256 the freeze lists
net_check() {
  sha256sum --quiet --strict -c $H/networks.sha256 || { echo "NET CHECK: a network differs from the freeze's list"; return 1; }
  echo "NET CHECK: 23 networks equal the freeze's sha256"
}
