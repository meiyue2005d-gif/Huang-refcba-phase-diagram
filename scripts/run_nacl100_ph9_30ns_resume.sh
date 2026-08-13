#!/usr/bin/env bash

cd ~/autodl-tmp/huang_refcba_repro || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate huang-md

for seed in 20260718 20260719 20260720
do
  out="results/direct_coexistence_long30ns_nacl100/pH9p0_c0p5_seed${seed}"
  mkdir -p "$out"

  echo
  echo "============================================================"
  echo "NaCl=100 mM, pH=9.0, 30 ns, seed=$seed"
  echo "Start time: $(date)"
  echo "============================================================"

  if [ ! -f "$out/trajectory_positions.npz" ]; then
    echo "[1/4] Running 30 ns simulation"

    python -u scripts/run_direct_coexistence.py \
      --ph 9.0 \
      --nacl-mM 100 \
      --global-concentration-mg-ml 0.5 \
      --initial-slab-concentration-mg-ml 5.0 \
      --z-aspect-ratio 3.0 \
      --equil-steps 200000 \
      --prod-steps 30000000 \
      --report-interval 20000 \
      --minimize-max-iterations 500 \
      --seed "$seed" \
      --output-dir "$out" \
      2>&1 | tee "$out/run.log"

    status=${PIPESTATUS[0]}

    if [ "$status" -ne 0 ]; then
      echo "FAILED simulation: $out"
      continue
    fi
  else
    echo "[1/4] SKIP existing trajectory"
  fi

  if [ ! -f "$out/direct_coexistence_summary.json" ]; then
    echo "[2/4] Running axial analysis"

    python scripts/analyze_direct_coexistence.py \
      --input-dir "$out" \
      --bins 60 \
      --analysis-fraction 0.5 \
      --smoothing-bins 3 || {
        echo "FAILED axial analysis: $out"
        continue
      }
  else
    echo "[2/4] SKIP existing axial analysis"
  fi

  if [ ! -f "$out/slab_dynamics_summary.json" ]; then
    echo "[3/4] Running slab dynamics"

    python scripts/analyze_slab_dynamics.py \
      --input-dir "$out" \
      --analysis-fraction 0.5 \
      --persistent-fraction 0.8 \
      --contact-cutoff-nm 5.35 || {
        echo "FAILED slab dynamics: $out"
        continue
      }
  else
    echo "[3/4] SKIP existing slab dynamics"
  fi

  if [ ! -f "$out/slab_contact_continuity_summary.json" ]; then
    echo "[4/4] Running contact audit"

    python scripts/audit_slab_contact_continuity.py \
      --input-dir "$out" \
      --analysis-fraction 0.5 \
      --persistent-fraction 0.8 \
      --contact-cutoff-nm 5.35 || {
        echo "FAILED contact audit: $out"
        continue
      }
  else
    echo "[4/4] SKIP existing contact audit"
  fi

  echo "COMPLETED: $out"
  echo "End time: $(date)"
done

echo
echo "ALL_30NS_JOBS_FINISHED"
echo "Finish time: $(date)"
