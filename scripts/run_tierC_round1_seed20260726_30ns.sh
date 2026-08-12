#!/usr/bin/env bash

set -u

PROJECT="$HOME/autodl-tmp/huang_refcba_repro"
PARAM_PY="/root/miniconda3/envs/huang-md/bin/python"
HOOMD_PY="/root/miniconda3/envs/huang-hoomd710/bin/python"

ROOT="$PROJECT/results/tierC_adaptive_30ns"
MANIFEST="$ROOT/round1_manifest.tsv"
MASTER_LOG="$ROOT/master_round1.log"
STATUS_FILE="$ROOT/round1_status.tsv"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$PROJECT" || exit 1

mkdir -p \
  "$ROOT/inputs" \
  "$ROOT/runs" \
  "$ROOT/logs"

if [ ! -f "$STATUS_FILE" ]; then
    printf \
      "run_token\tstatus\texit_code\telapsed_seconds\tfinished_at\n" \
      > "$STATUS_FILE"
fi

is_complete() {
    local run_dir="$1"

    "$HOOMD_PY" - "$run_dir" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])

required = [
    root / "metadata.json",
    root / "trajectory_positions.npz",
    root / "production_thermo.csv",
    root / "final_state_hoomd.npz",
]

if not all(path.is_file() for path in required):
    raise SystemExit(1)

try:
    metadata = json.loads(
        (root / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
except Exception:
    raise SystemExit(1)

raise SystemExit(
    0
    if int(metadata.get("prod_steps", -1)) >= 30_000_000
    else 1
)
PY
}

{
echo "================================================================"
echo "TIER-C ADAPTIVE ROUND 1"
echo "Nine representative unresolved states"
echo "Started: $(date --iso-8601=seconds)"
echo "================================================================"
} | tee -a "$MASTER_LOG"

tail -n +2 "$MANIFEST" |
while IFS=$'\t' read -r \
    PH NACL CONC SEED RUN_TOKEN
do
    [ -z "$RUN_TOKEN" ] && continue

    INPUT_DIR="$ROOT/inputs/$RUN_TOKEN"
    OUTPUT_DIR="$ROOT/runs/$RUN_TOKEN"
    RUN_LOG="$ROOT/logs/${RUN_TOKEN}.log"

    {
    echo
    echo "================================================================"
    echo "RUN: $RUN_TOKEN"
    echo "Started: $(date --iso-8601=seconds)"
    echo "================================================================"
    } | tee -a "$MASTER_LOG"

    if is_complete "$OUTPUT_DIR"; then
        echo "Already complete; skipping." |
            tee -a "$MASTER_LOG"

        printf \
          "%s\tcomplete_existing\t0\t0\t%s\n" \
          "$RUN_TOKEN" \
          "$(date --iso-8601=seconds)" \
          >> "$STATUS_FILE"

        continue
    fi

    if [ -d "$OUTPUT_DIR" ]; then
        BACKUP="${OUTPUT_DIR}_incomplete_$(date +%Y%m%d_%H%M%S)"

        echo "Moving incomplete output to $BACKUP" |
            tee -a "$MASTER_LOG"

        mv "$OUTPUT_DIR" "$BACKUP"
    fi

    rm -rf "$INPUT_DIR"

    "$PARAM_PY" \
      scripts/export_hoomd_state_input.py \
      --ph "$PH" \
      --nacl-mM "$NACL" \
      --concentration-mg-ml "$CONC" \
      --seed "$SEED" \
      --output-dir "$INPUT_DIR" \
      > "$RUN_LOG" 2>&1

    EXPORT_STATUS=$?

    if [ "$EXPORT_STATUS" -ne 0 ]; then
        echo "EXPORT FAILED: exit=$EXPORT_STATUS" |
            tee -a "$MASTER_LOG"

        printf \
          "%s\texport_failed\t%s\t0\t%s\n" \
          "$RUN_TOKEN" \
          "$EXPORT_STATUS" \
          "$(date --iso-8601=seconds)" \
          >> "$STATUS_FILE"

        continue
    fi

    echo "Starting 30 ns production..." |
        tee -a "$MASTER_LOG"

    START=$(date +%s)

    "$HOOMD_PY" \
      scripts/run_single_state_hoomd.py \
      --input-dir "$INPUT_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --minimize-max-steps 500 \
      --equil-steps 50000 \
      --prod-steps 30000000 \
      --report-interval 100000 \
      >> "$RUN_LOG" 2>&1

    RUN_STATUS=$?
    ELAPSED=$(($(date +%s) - START))

    if [ "$RUN_STATUS" -eq 0 ] &&
       is_complete "$OUTPUT_DIR"
    then
        echo \
          "RUN COMPLETE: $RUN_TOKEN elapsed_seconds=$ELAPSED" |
          tee -a "$MASTER_LOG"

        printf \
          "%s\tcomplete\t0\t%s\t%s\n" \
          "$RUN_TOKEN" \
          "$ELAPSED" \
          "$(date --iso-8601=seconds)" \
          >> "$STATUS_FILE"
    else
        echo \
          "RUN FAILED OR INCOMPLETE: $RUN_TOKEN exit=$RUN_STATUS" |
          tee -a "$MASTER_LOG"

        printf \
          "%s\trun_failed_or_incomplete\t%s\t%s\t%s\n" \
          "$RUN_TOKEN" \
          "$RUN_STATUS" \
          "$ELAPSED" \
          "$(date --iso-8601=seconds)" \
          >> "$STATUS_FILE"
    fi
done

{
echo
echo "================================================================"
echo "TIER-C ADAPTIVE ROUND 1 FINISHED"
echo "Finished: $(date --iso-8601=seconds)"
echo "================================================================"
} | tee -a "$MASTER_LOG"
