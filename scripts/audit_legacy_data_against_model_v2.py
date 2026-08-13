#!/usr/bin/env python3
"""Audit legacy simulation runs against the versioned state models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huang_md.state_model import (  # noqa: E402
    RefCBAStateModel,
    calculate_K2_kBT,
    calculate_Z2,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--results-root", type=Path, default=ROOT / "results")
    result.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "results" / "model_v2_data_audit.csv",
    )
    return result


def scalar(function, pH: float, salt: float, model: RefCBAStateModel) -> float:
    if function is calculate_Z2:
        return float(function([salt], model)[0])
    return float(function([pH], [salt], model)[0])


def main() -> None:
    args = parser().parse_args()
    legacy = RefCBAStateModel.from_yaml(
        ROOT / "configs" / "refcba_state_model_legacy.yaml"
    )
    revised = RefCBAStateModel.from_yaml(
        ROOT / "configs" / "refcba_state_model.yaml"
    )

    rows: list[dict[str, object]] = []
    for metadata_path in args.results_root.rglob("metadata.json"):
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        required = {"pH", "added_nacl_mM", "K2_kBT", "Z2"}
        if not required.issubset(data):
            continue

        pH = float(data["pH"])
        salt = float(data["added_nacl_mM"])
        stored_K2 = float(data["K2_kBT"])
        stored_Z2 = float(data["Z2"])
        legacy_K2 = scalar(calculate_K2_kBT, pH, salt, legacy)
        revised_K2 = scalar(calculate_K2_kBT, pH, salt, revised)
        revised_Z2 = scalar(calculate_Z2, pH, salt, revised)
        matches_legacy = bool(
            np.isclose(stored_K2, legacy_K2, rtol=1e-8, atol=1e-8)
            and np.isclose(stored_Z2, revised_Z2, rtol=1e-8, atol=1e-8)
        )
        matches_revised = bool(
            np.isclose(stored_K2, revised_K2, rtol=1e-8, atol=1e-8)
            and np.isclose(stored_Z2, revised_Z2, rtol=1e-8, atol=1e-8)
        )
        run_dir = metadata_path.parent
        has_trajectory = (run_dir / "trajectory_positions.npz").exists()
        has_final_state = (run_dir / "final_state_hoomd.npz").exists()

        if matches_revised:
            disposition = "directly_reusable_revised_model"
        elif matches_legacy and has_trajectory:
            disposition = "legacy_only_raw_trajectory_reanalyzable"
        elif matches_legacy:
            disposition = "legacy_only_summary_available"
        else:
            disposition = "unknown_or_other_model_review_required"

        rows.append(
            {
                "run_directory": str(run_dir),
                "pH": pH,
                "added_nacl_mM": salt,
                "concentration_mg_ml": data.get("concentration_mg_ml"),
                "seed": data.get("seed"),
                "stored_K2_kBT": stored_K2,
                "legacy_K2_kBT": legacy_K2,
                "revised_K2_kBT": revised_K2,
                "relative_K2_change_revised_vs_stored": (
                    (revised_K2 - stored_K2) / max(abs(stored_K2), 1e-12)
                ),
                "stored_Z2": stored_Z2,
                "revised_Z2": revised_Z2,
                "matches_legacy_model": matches_legacy,
                "matches_revised_model": matches_revised,
                "has_trajectory_positions": has_trajectory,
                "has_final_state": has_final_state,
                "disposition": disposition,
            }
        )

    frame = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)

    print(f"Audited runs: {len(frame)}")
    if not frame.empty:
        print(frame["disposition"].value_counts().to_string())
    print(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()
