#!/usr/bin/env python3
"""Conservatively classify a manifest and produce boundary/refinement tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huang_md.phase_scan import (  # noqa: E402
    PhaseScanConfig,
    classify_run_directory,
    build_direct_coexistence_manifest,
    expand_refinement_manifest,
    extract_boundary_intervals,
    refinement_states,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase_scan.yaml")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = PhaseScanConfig.from_yaml(args.config)
    manifest = (args.manifest or config.output_root / "coarse_manifest.tsv").resolve()
    output = (args.output_dir or config.output_root / "summary").resolve()
    output.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(manifest, sep="\t")
    evidence = pd.DataFrame(
        [
            classify_run_directory(
                path, config.classification, expected_model_id=config.state_model_id
            )
            for path in table["output_dir"]
        ]
    )
    combined = pd.concat([table.reset_index(drop=True), evidence], axis=1)
    boundaries = extract_boundary_intervals(combined)
    refine = refinement_states(combined, boundaries)
    combined.to_csv(output / "phase_state_table.csv", index=False)
    boundaries.to_csv(output / "phase_boundary_intervals.csv", index=False)
    refine.to_csv(output / "targeted_refinement_manifest.csv", index=False)
    coexist = refine[refine["phase_operational"] == "llps_candidate"].copy()
    coexist.to_csv(output / "direct_coexistence_candidates.csv", index=False)
    seeds = tuple(int(value) for value in config.refinement.get("seeds", []))
    long_manifest = expand_refinement_manifest(
        refine, config.output_root, seeds, config.state_config, config.md_config
    )
    direct_manifest = build_direct_coexistence_manifest(
        coexist, config.output_root, seeds, config.state_config, config.md_config
    )
    long_columns = [
        "task_id", "source_state_id", "state_id", "ph", "nacl_mM",
        "concentration_mg_ml", "seed", "output_dir", "state_config",
        "md_config", "selection_reason",
        "expected_state_model_id",
    ]
    direct_columns = [
        "task_id", "source_state_id", "state_id", "ph", "nacl_mM",
        "global_concentration_mg_ml", "initial_slab_concentration_mg_ml",
        "z_aspect_ratio", "seed", "output_dir", "state_config", "md_config",
        "expected_state_model_id",
    ]
    long_manifest.reindex(columns=long_columns).to_csv(
        output / "long_run_manifest.tsv", sep="\t", index=False
    )
    direct_manifest.reindex(columns=direct_columns).to_csv(
        output / "direct_coexistence_manifest.tsv", sep="\t", index=False
    )
    print(combined["phase_conservative"].value_counts(dropna=False).to_string())
    print(f"Boundary intervals: {len(boundaries)}")
    print(f"Long-run candidates: {len(refine)}")
    print(f"Direct-coexistence candidates: {len(coexist)}")
    print(f"Replicated long runs: {len(long_manifest)}")
    print(f"Replicated slab runs: {len(direct_manifest)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
