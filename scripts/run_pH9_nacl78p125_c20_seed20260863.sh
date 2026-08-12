#!/usr/bin/env bash
set -eu

ROOT="$HOME/autodl-tmp/huang_refcba_repro"
cd "$ROOT"

MANIFEST="$ROOT/manifests/pH9_nacl78p125_c20_seed20260863.tsv"
RESULT_ROOT="$ROOT/results/pH9_nacl78p125_c20_seed20260863"

EXPORTER="$ROOT/scripts/export_hoomd_state_input.py"
RUNNER="$ROOT/scripts/run_single_state_hoomd.py"

HUANG_PY="/root/miniconda3/envs/huang-md/bin/python"
HOOMD_PY="/root/miniconda3/envs/huang-hoomd710/bin/python"

SEED=20260863

mkdir -p "$RESULT_ROOT"

for path in "$HUANG_PY" "$HOOMD_PY" "$EXPORTER" "$RUNNER" "$MANIFEST"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: missing $path"
        exit 1
    fi
done

echo "============================================================"
echo "HUANG_PY  : $HUANG_PY"
echo "HOOMD_PY  : $HOOMD_PY"
echo "EXPORTER  : $EXPORTER"
echo "RUNNER    : $RUNNER"
echo "MANIFEST  : $MANIFEST"
echo "RESULTS   : $RESULT_ROOT"
echo "============================================================"

"$HOOMD_PY" - <<'PY'
import hoomd
print("HOOMD version:", hoomd.version.version)
print("GPU enabled :", hoomd.version.gpu_enabled)
PY

total=$(
    awk 'NF >= 3 && $1 !~ /^#/ {n++} END {print n+0}' "$MANIFEST"
)

index=0

while read -r ph salt concentration extra; do
    [[ -z "${ph:-}" ]] && continue
    [[ "$ph" == \#* ]] && continue

    index=$((index + 1))

    ph_tag="${ph//./p}"
    salt_tag="${salt//./p}"
    c_tag="${concentration//./p}"

    state_id="pH${ph_tag}_nacl${salt_tag}_c${c_tag}_seed${SEED}"
    out="$RESULT_ROOT/$state_id"
    input_dir="$out/hoomd_input"

    echo
    echo "================================================================"
    echo "[$index/$total] $state_id"
    echo "pH=$ph  NaCl=$salt mM  concentration=$concentration mg/mL"
    echo "================================================================"

    if [[ -f "$out/run.log" ]] &&
       grep -q "RUN_SINGLE_STATE_HOOMD: PASS" "$out/run.log" &&
       [[ -f "$out/trajectory_positions.npz" ]] &&
       [[ -f "$out/metadata.json" ]]; then
        echo "SKIP: already complete"
        continue
    fi

    if [[ -d "$out" ]]; then
        backup="${out}_partial_$(date +%Y%m%d_%H%M%S)"
        echo "Moving incomplete directory to:"
        echo "$backup"
        mv "$out" "$backup"
    fi

    mkdir -p "$input_dir"

    echo
    echo "===== EXPORT INPUT ====="

    set +e
    "$HUANG_PY" "$EXPORTER" \
        --ph "$ph" \
        --nacl-mM "$salt" \
        --concentration-mg-ml "$concentration" \
        --seed "$SEED" \
        --output-dir "$input_dir" \
        2>&1 | tee "$out/export.log"

    export_code=${PIPESTATUS[0]}
    set -e

    if [[ "$export_code" -ne 0 ]]; then
        echo "FAILED EXPORT: $state_id"
        exit "$export_code"
    fi

    echo
    echo "Generated input files:"
    find "$input_dir" -maxdepth 2 -type f -printf '%P\n' | sort

    echo
    echo "===== RUN HOOMD ====="

    set +e
    "$HOOMD_PY" "$RUNNER" \
        --input-dir "$input_dir" \
        --output-dir "$out" \
        --equil-steps 50000 \
        --prod-steps 30000000 \
        --report-interval 100000 \
        --minimize-max-steps 500 \
        2>&1 | tee "$out/run.log"

    run_code=${PIPESTATUS[0]}
    set -e

    if [[ "$run_code" -ne 0 ]]; then
        echo "FAILED RUN: $state_id exit_code=$run_code"
        exit "$run_code"
    fi

    if ! grep -q "RUN_SINGLE_STATE_HOOMD: PASS" "$out/run.log"; then
        echo "FAILED: PASS marker not found"
        exit 1
    fi

    if [[ ! -f "$out/trajectory_positions.npz" ]]; then
        echo "FAILED: missing trajectory_positions.npz"
        exit 1
    fi

    if [[ ! -f "$out/metadata.json" ]]; then
        echo "FAILED: missing metadata.json"
        exit 1
    fi

    echo "COMPLETE: $state_id"

done < "$MANIFEST"

echo
echo "================================================================"
echo "THREE_PHASE_NACL78P125_20260863: COMPLETE"
echo "================================================================"
