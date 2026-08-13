#!/usr/bin/env python3
"""Merge replicated long/slab runs into a conservative final state table."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huang_md.phase_scan import (  # noqa: E402
    PhaseScanConfig,
    classify_run_directory,
    extract_boundary_intervals,
)


def consensus(labels: list[str], minimum_seeds: int) -> tuple[str, str]:
    resolved = [label for label in labels if label in {"soluble", "llps", "aggregate"}]
    counts = Counter(resolved)
    if not counts:
        return "unresolved", "no_resolved_replicates"
    phase, count = counts.most_common(1)[0]
    if count < minimum_seeds:
        return "unresolved", f"only_{count}_supporting_seed"
    conflicting = sum(value for key, value in counts.items() if key != phase)
    if conflicting:
        return "unresolved", "conflicting_replicates"
    return phase, f"{count}_seed_consensus"


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase_scan.yaml")
    parser.add_argument("--summary-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--minimum-seeds", type=int, default=2)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    config = PhaseScanConfig.from_yaml(args.config)
    summary = (args.summary_dir or config.output_root / "summary").resolve()
    output = (args.output_dir or config.output_root / "final").resolve()
    output.mkdir(parents=True, exist_ok=True)

    states = pd.read_csv(summary / "phase_state_table.csv")
    long_manifest = read_manifest(summary / "long_run_manifest.tsv")
    direct_manifest = read_manifest(summary / "direct_coexistence_manifest.tsv")

    for index, state in states.iterrows():
        source_id = str(state["state_id"])
        long_labels: list[str] = []
        if not long_manifest.empty:
            selected = long_manifest[long_manifest["source_state_id"].astype(str) == source_id]
            long_labels = [
                classify_run_directory(
                    path, config.classification, expected_model_id=config.state_model_id
                )["phase_conservative"]
                for path in selected["output_dir"]
            ]
        slab_labels: list[str] = []
        if not direct_manifest.empty:
            selected = direct_manifest[direct_manifest["source_state_id"].astype(str) == source_id]
            slab_labels = [
                classify_run_directory(
                    state["output_dir"], config.classification, direct_dir=path,
                    expected_model_id=config.state_model_id,
                )["phase_conservative"]
                for path in selected["output_dir"]
            ]

        final_phase = str(state["phase_conservative"])
        final_reason = "coarse_classification_no_targeted_refinement"
        if long_labels:
            phase, reason = consensus(long_labels, args.minimum_seeds)
            final_phase, final_reason = phase, f"long:{reason}"
        if slab_labels:
            phase, reason = consensus(slab_labels, args.minimum_seeds)
            # Direct coexistence is the only route to confirmed LLPS and takes
            # precedence only when replicated and conflict-free.
            if phase == "llps" or final_phase == "unresolved":
                final_phase, final_reason = phase, f"slab:{reason}"
        if (
            str(state.get("phase_operational")) == "llps_candidate"
            and (not slab_labels or set(slab_labels) <= {"unresolved"})
        ):
            final_phase = "unresolved"
            final_reason = "awaiting_replicated_direct_coexistence"
        states.at[index, "long_seed_labels"] = ";".join(long_labels)
        states.at[index, "slab_seed_labels"] = ";".join(slab_labels)
        states.at[index, "phase_final"] = final_phase
        states.at[index, "final_consensus_reason"] = final_reason

    boundaries = extract_boundary_intervals(
        states, phase_col="phase_final", include_unresolved=False
    )
    states.to_csv(output / "final_phase_state_table.csv", index=False)
    boundaries.to_csv(output / "final_phase_boundary_intervals.csv", index=False)
    print(states["phase_final"].value_counts(dropna=False).to_string())
    print(f"Boundary intervals: {len(boundaries)}")
    print(f"Output: {output}")
    if not args.no_plots:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "plot_phase_diagrams.py"),
                "--input", str(output / "final_phase_state_table.csv"),
                "--output-dir", str(output),
                "--phase-column", "phase_final",
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
