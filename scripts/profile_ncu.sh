#!/usr/bin/env bash
# Nsight Compute counter capture for the fused LK kernel vs OpenCV's GPU LK.
#
# ncu serializes and replays each kernel to gather counters, so the durations
# it reports are inflated and meaningless -- speedup lives in the wall-clock
# table printed by lk_gpu on a normal (non-ncu) run. This script produces the
# hardware-counter evidence for that speedup: DRAM traffic, achieved throughput,
# cache reuse, and occupancy for a single steady-state launch of each arm.
#
# The lk_gpu binary must be built with the `--profile` mode (see cuda/main.cpp),
# which warms both arms, then wraps exactly one custom-kernel launch and one
# OpenCV calc() between cudaProfilerStart/Stop. `--profile-from-start off` makes
# ncu capture only that region, so the KITTI load, grayscale, uploads, and the
# cold first frame are all excluded.
#
# Clocks are locked once (nvidia-smi -lgc) so rate/percent metrics describe one
# consistent operating point; byte counts are frequency-invariant regardless.
# Locking needs root and is reset on exit. A host that forbids clock control can
# skip the lock (LOCK_CLOCKS=0) at the cost of comparability across runs.
#
# Usage:
#   scripts/profile_ncu.sh [BINARY] [OUTDIR]
#     BINARY  path to the built lk_gpu (default: cuda/build/lk_gpu)
#     OUTDIR  where the report + table are written (default: bench/results)
# Env:
#   CLOCK=1200        graphics clock (MHz) to lock; sustainable on a 1650 Ti
#   LOCK_CLOCKS=1     set 0 to skip nvidia-smi clock locking (no root)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="${1:-$REPO_ROOT/cuda/build/lk_gpu}"
OUTDIR="${2:-$REPO_ROOT/bench/results}"
CLOCK="${CLOCK:-1200}"
LOCK_CLOCKS="${LOCK_CLOCKS:-1}"

REPORT="$OUTDIR/lk_profile"
CSV="$OUTDIR/lk_counters.csv"
TABLE="$OUTDIR/lk_counters.md"

# Memory-bound counter set for sm_75. Byte sums show the traffic fusion cuts;
# the two throughput percents are the speed-of-light read (memory-bound if DRAM
# is high while SM is low); hit rates show cache reuse; occupancy + registers
# explain how much of the machine each kernel keeps busy.
METRICS="dram__bytes_read.sum,\
dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread"

if [[ ! -x "$BINARY" ]]; then
    echo "error: lk_gpu binary not found or not executable: $BINARY" >&2
    echo "build it first (cmake --build cuda/build) or pass the path as arg 1." >&2
    exit 1
fi

mkdir -p "$OUTDIR"

cleanup() {
    if [[ "$LOCK_CLOCKS" == "1" ]]; then
        sudo nvidia-smi -rgc >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if [[ "$LOCK_CLOCKS" == "1" ]]; then
    echo "locking graphics clock to ${CLOCK} MHz (sudo; reset on exit)"
    sudo nvidia-smi -pm 1 >/dev/null
    sudo nvidia-smi -lgc "${CLOCK},${CLOCK}" >/dev/null
fi

# ncu runs the binary from its own directory so lk_gpu's FlowSet("../../data")
# resolves to the same KITTI root the normal eval uses.
echo "capturing counters -> ${REPORT}.ncu-rep"
( cd "$(dirname "$BINARY")" && \
  ncu --profile-from-start off \
      --clock-control none \
      --target-processes all \
      --metrics "$METRICS" \
      --force-overwrite -o "$REPORT" \
      "./$(basename "$BINARY")" --profile )

echo "exporting csv -> ${CSV}"
ncu --import "${REPORT}.ncu-rep" --csv --page details --print-units base > "$CSV"

echo "building table -> ${TABLE}"
python3 "$REPO_ROOT/scripts/parse_ncu.py" "$CSV" > "$TABLE"

echo
cat "$TABLE"
echo
echo "open the full report with: ncu-ui ${REPORT}.ncu-rep"
