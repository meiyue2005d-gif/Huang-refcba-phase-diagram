#!/usr/bin/env bash

set -u

PROJECT=~/autodl-tmp/huang_refcba_repro
HUANG_PY=/root/miniconda3/envs/huang-md/bin/python
HOOMD_PY=/root/miniconda3/envs/huang-hoomd710/bin/python

QUEUE="$PROJECT/results/tierA_llps_30ns/tierA_seed20260723_missing.tsv"
ROOT="$PROJECT/results/tierA_llps_30ns"
MASTER_LOG="$ROOT/master_seed20260723.log"
STATUS_FILE="$ROOT/status_seed20260723.tsv"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$PROJECT" || {
    echo "Could not enter project directory."
    return 1 2>/dev/null || true
}

mkdir -p \
    "$ROOT/inputs" \
    "$ROOT/runs" \
    "$ROOT/logs"

if [ ! -f "$STATUS_FILE" ]; then
    printf \
      "run_token\tstatus\texit_code\tfinished_at\n" \
      > "$STATUS_FILE"
fi

is_complete() {
    local output_dir="$1"

    "$HOOMD_PY" - "$output_dir" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])

required = [
    root / "metadata.json",
    root / "final_state_hoomd.npz",
    root / "trajectory_positions.npz",
    root / "production_thermo.csv",
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

if int(metadata.get("prod_steps", -1)) < 30_000_000:
    raise SystemExit(1)

raise SystemExit(0)
PY
}

{
echo "================================================================"
echo "TIER-A LLPS 30 NS BATCH"
echo "Started: $(date --iso-8601=seconds)"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Queue: $QUEUE"
echo "================================================================"
} | tee -a "$MASTER_LOG"

tail -n +2 "$QUEUE" |
while IFS=$'\t' read -r \
    PH \
    NACL \
    CONC \
    SEED \
    BASE_TOKEN \
    RUN_TOKEN
do
    INPUT_DIR="$ROOT/inputs/$RUN_TOKEN"
    OUTPUT_DIR="$ROOT/runs/$RUN_TOKEN"
    RUN_LOG="$ROOT/logs/${RUN_TOKEN}.log"

    echo | tee -a "$MASTER_LOG"
    echo "================================================================" \
        | tee -a "$MASTER_LOG"
    echo "RUN: $RUN_TOKEN" \
        | tee -a "$MASTER_LOG"
    echo "Started: $(date --iso-8601=seconds)" \
        | tee -a "$MASTER_LOG"
    echo "================================================================" \
        | tee -a "$MASTER_LOG"

    if is_complete "$OUTPUT_DIR"; then
        echo "Already complete; skipping." \
            | tee -a "$MASTER_LOG"

        printf \
          "%s\tcomplete_existing\t0\t%s\n" \
          "$RUN_TOKEN" \
          "$(date --iso-8601=seconds)" \
          >> "$STATUS_FILE"

        continue
    fi

    if [ -d "$OUTPUT_DIR" ]; then
        BACKUP_DIR="${
            OUTPUT_DIR
        }_incomplete_$(
            date +%Y%m%d_%H%M%S
        )"

        echo \
          "Moving incomplete output to: $BACKUP_DIR" \
          | tee -a "$MASTER_LOG"

        mv "$OUTPUT_DIR" "$BACKUP_DIR"
    fi

    rm -rf "$INPUT_DIR"

    echo "Exporting input..." \
        | tee -a "$MASTER_LOG"

    "$HUANG_PY" \
        scripts/export_hoomd_state_input.py \
        --ph "$PH" \
        --nacl-mM "$NACL" \
        --concentration-mg-ml "$CONC" \
        --seed "$SEED" \
        --output-dir "$INPUT_DIR" \
        >> "$RUN_LOG" 2>&1

    EXPORT_STATUS=$?

    if [ "$EXPORT_STATUS" -ne 0 ]; then
        echo \
          "EXPORT FAILED: status=$EXPORT_STATUS" \
          | tee -a "$MASTER_LOG"

        printf \
          "%s\texport_failed\t%s\t%s\n" \
          "$RUN_TOKEN" \
          "$EXPORT_STATUS" \
          "$(date --iso-8601=seconds)" \
          >> "$STATUS_FILE"

        continue
    fi

    echo "Starting 30 ns production..." \
        | tee -a "$MASTER_LOG"

    START_SECONDS=$(date +%s)

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
    END_SECONDS=$(date +%s)
    ELAPSED=$((END_SECONDS - START_SECONDS))

    echo \
      "Runner exit=$RUN_STATUS elapsed_seconds=$ELAPSED" \
      | tee -a "$MASTER_LOG"

    if [ "$RUN_STATUS" -eq 0 ] \
       && is_complete "$OUTPUT_DIR"
    then
        echo "RUN COMPLETE: $RUN_TOKEN" \
            | tee -a "$MASTER_LOG"

        printf \
          "%s\tcomplete\t0\t%s\n" \
          "$RUN_TOKEN" \
          "$(date --iso-8601=seconds)" \
          >> "$STATUS_FILE"
    else
        echo "RUN FAILED OR INCOMPLETE: $RUN_TOKEN" \
            | tee -a "$MASTER_LOG"

        printf \
          "%s\trun_failed_or_incomplete\t%s\t%s\n" \
          "$RUN_TOKEN" \
          "$RUN_STATUS" \
          "$(date --iso-8601=seconds)" \
          >> "$STATUS_FILE"
    fi
done

{
echo
echo "================================================================"
echo "TIER-A SEED-20260723 BATCH FINISHED"
echo "Finished: $(date --iso-8601=seconds)"
echo "================================================================"
} | tee -a "$MASTER_LOG"
