from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from huang_md.phase_scan import (
    PhaseScanConfig,
    classify_run_directory,
    extract_boundary_intervals,
)


ROOT = Path(__file__).resolve().parents[1]


def test_default_grid_has_448_unique_states() -> None:
    config = PhaseScanConfig.from_yaml(ROOT / "configs" / "phase_scan.yaml")
    rows = config.manifest_rows()
    assert config.state_count == 448
    assert len({row["state_id"] for row in rows}) == 448
    assert {row["ph"] for row in rows} >= {3.0, 9.0}
    assert {row["nacl_mM"] for row in rows} >= {0.0, 500.0}
    assert {row["concentration_mg_ml"] for row in rows} >= {0.1, 20.0}
    assert all(not Path(row["output_dir"]).is_absolute() for row in rows)


def write_dynamics(run: Path, **overrides: object) -> None:
    values = {
        "mean_clustered_fraction": 0.95,
        "mean_largest_cluster_fraction": 0.15,
        "percolation_fraction": 0.0,
        "final_msd_nm2": 1.0,
        "frozen_finite_clusters": False,
        "final_initial_bond_survival": 0.2,
        "mean_consecutive_bond_retention": 0.5,
        "late_largest_cluster_fraction_slope_per_ns": 0.0,
    }
    values.update(overrides)
    target = run / "dynamics_analysis"
    target.mkdir(parents=True)
    (target / "dynamics_summary.json").write_text(json.dumps(values), encoding="utf-8")


def test_mobile_finite_clusters_are_soluble(tmp_path: Path) -> None:
    write_dynamics(tmp_path)
    result = classify_run_directory(tmp_path)
    assert result["phase_conservative"] == "soluble"
    assert result["phase_operational"] == "soluble"
    assert result["internal_state"] == "mobile_finite_cluster_fluid"


def test_homogeneous_coarsening_is_only_an_llps_candidate(tmp_path: Path) -> None:
    write_dynamics(tmp_path, late_largest_cluster_fraction_slope_per_ns=0.1)
    result = classify_run_directory(tmp_path)
    assert result["phase_conservative"] == "unresolved"
    assert result["phase_operational"] == "llps_candidate"


def test_llps_requires_profile_and_exchange(tmp_path: Path) -> None:
    write_dynamics(tmp_path)
    (tmp_path / "direct_coexistence_summary.json").write_text(
        json.dumps({
            "profile_state": "persistent_axial_inhomogeneity",
            "late_mean_density_contrast": 0.7,
            "late_mean_axial_cv": 1.0,
            "exchange_proxy_detected": True,
        }), encoding="utf-8"
    )
    (tmp_path / "slab_dynamics_summary.json").write_text(
        json.dumps({
            "mean_late_contact_turnover_fraction": 0.1,
            "final_core_relative_msd_nm2": 0.2,
        }), encoding="utf-8"
    )
    assert classify_run_directory(tmp_path)["phase_conservative"] == "llps"


def test_missing_slab_output_cannot_count_as_soluble_evidence(tmp_path: Path) -> None:
    homogeneous = tmp_path / "homogeneous"
    slab = tmp_path / "unfinished_slab"
    write_dynamics(homogeneous)
    slab.mkdir()
    result = classify_run_directory(homogeneous, direct_dir=slab)
    assert result["phase_conservative"] == "unresolved"
    assert not result["complete"]


def test_dissolved_slab_resolves_candidate_as_soluble(tmp_path: Path) -> None:
    homogeneous = tmp_path / "homogeneous"
    slab = tmp_path / "slab"
    write_dynamics(homogeneous, late_largest_cluster_fraction_slope_per_ns=0.1)
    slab.mkdir()
    (slab / "direct_coexistence_summary.json").write_text(
        json.dumps({
            "profile_state": "weak_or_dissolved_profile",
            "late_mean_density_contrast": 0.1,
            "late_mean_axial_cv": 0.2,
            "exchange_proxy_detected": False,
        }), encoding="utf-8"
    )
    (slab / "slab_dynamics_summary.json").write_text(
        json.dumps({"analysis_frames": 10}), encoding="utf-8"
    )
    result = classify_run_directory(homogeneous, direct_dir=slab)
    assert result["phase_conservative"] == "soluble"
    assert result["internal_state"] == "slab_dissolved_to_single_phase"


def test_model_mismatch_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text(
        json.dumps({"state_model_id": "legacy_refcba_abs_charge_v1"}),
        encoding="utf-8",
    )
    write_dynamics(tmp_path)
    result = classify_run_directory(tmp_path, expected_model_id="current_v3")
    assert result["phase_conservative"] == "unresolved"
    assert result["internal_state"] == "state_model_mismatch"


def test_boundaries_are_reported_as_sampled_intervals() -> None:
    table = pd.DataFrame([
        {"state_id": "a", "ph": 4.0, "nacl_mM": 0.0, "concentration_mg_ml": 1.0, "phase_final": "soluble"},
        {"state_id": "b", "ph": 4.0, "nacl_mM": 0.0, "concentration_mg_ml": 2.0, "phase_final": "llps"},
    ])
    boundaries = extract_boundary_intervals(table, "phase_final")
    row = boundaries[boundaries["scan_axis"] == "concentration_mg_ml"].iloc[0]
    assert row["lower_sample"] == 1.0
    assert row["upper_sample"] == 2.0
    assert bool(row["boundary_is_interval"])


def test_final_boundaries_exclude_unresolved_transitions() -> None:
    table = pd.DataFrame([
        {"state_id": "a", "ph": 4.0, "nacl_mM": 0.0, "concentration_mg_ml": 1.0, "phase_final": "soluble"},
        {"state_id": "b", "ph": 4.0, "nacl_mM": 0.0, "concentration_mg_ml": 2.0, "phase_final": "unresolved"},
    ])
    assert extract_boundary_intervals(
        table, "phase_final", include_unresolved=False
    ).empty
