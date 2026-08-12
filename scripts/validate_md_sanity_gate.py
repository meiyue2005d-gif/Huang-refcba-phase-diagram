#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from huang_md.md_phase_classifier import MDPhaseMetrics, classify_md_phase


CONTROL = ROOT / "results/analysis/md_control_sanity_inputs.csv"
PILOT = ROOT / "results/pilot_scan_nacl0/pilot_summary.csv"
OUT = ROOT / "results/analysis/md_sanity_gate"


def classify(row: pd.Series) -> dict[str, object]:
    result = classify_md_phase(
        MDPhaseMetrics(
            mean_clustered_fraction=float(row.mean_clustered_fraction),
            mean_largest_cluster_fraction=float(
                row.mean_largest_cluster_fraction
            ),
            percolation_fraction=float(row.percolation_fraction),
            final_msd_nm2=float(row.final_msd_nm2),
            frozen_finite_clusters=bool(row.frozen_finite_clusters),
        )
    )
    return {
        "gate_phase": result.final_phase,
        "gate_internal_state": result.internal_state,
        "gate_confidence": result.confidence,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    controls = pd.read_csv(CONTROL)
    pilot = pd.read_csv(PILOT)

    expected = {
        "control_soluble_pH3": "soluble",
        "control_soluble_pH9": "soluble",
        "diagnostic_pI_c20": "aggregate",
    }

    control_rows = []
    for _, row in controls.iterrows():
        result = classify(row)
        target = expected[str(row.control_group)]
        control_rows.append(
            {
                **row.to_dict(),
                **result,
                "expected_phase": target,
                "sanity_pass": result["gate_phase"] == target,
            }
        )

    pilot_rows = []
    for _, row in pilot.iterrows():
        result = classify(row)
        existing = str(row.provisional_phase)
        pilot_rows.append(
            {
                **row.to_dict(),
                **result,
                "existing_phase": existing,
                "matches_existing": result["gate_phase"] == existing,
            }
        )

    control_table = pd.DataFrame(control_rows)
    pilot_table = pd.DataFrame(pilot_rows)

    controls_pass = bool(control_table.sanity_pass.all())
    matches = int(pilot_table.matches_existing.sum())
    total = len(pilot_table)
    no_llps = bool((pilot_table.gate_phase != "llps").all())

    control_table.to_csv(OUT / "control_gate_validation.csv", index=False)
    pilot_table.to_csv(OUT / "pilot_gate_validation.csv", index=False)

    summary = {
        "controls_pass": controls_pass,
        "pilot_matches_existing": matches,
        "pilot_rows": total,
        "phase_counts": {
            str(k): int(v)
            for k, v in pilot_table.gate_phase.value_counts().items()
        },
        "llps_states_from_homogeneous_pilot": int(
            (pilot_table.gate_phase == "llps").sum()
        ),
    }
    (OUT / "md_sanity_gate_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 94)
    print("Control sanity gate")
    print("=" * 94)
    print(
        control_table[
            [
                "control_group",
                "expected_phase",
                "gate_phase",
                "gate_internal_state",
                "sanity_pass",
            ]
        ].to_string(index=False)
    )

    print("\nPilot matches existing:", f"{matches}/{total}")
    print("\nFinal phase counts:")
    print(pilot_table.gate_phase.value_counts().to_string())

    checks = {
        "all controls pass": controls_pass,
        "all pilot rows reproduced": matches == total,
        "no homogeneous state called LLPS": no_llps,
    }

    print("\nLogical checks:")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL':<4s} {name}")

    if not all(checks.values()):
        raise RuntimeError("MD sanity gate validation failed.")

    print("\nMD sanity gate validation completed.")


if __name__ == "__main__":
    main()
