"""Configuration, evidence aggregation, and boundaries for phase scans."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from huang_md.md_phase_classifier import (
    MDPhaseMetrics,
    classify_md_phase,
)


AXES = ("ph", "nacl_mM", "concentration_mg_ml")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def tag_number(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class PhaseScanConfig:
    scan_id: str
    state_config: Path
    md_config: Path
    output_root: Path
    pH_values: tuple[float, ...]
    nacl_mM_values: tuple[float, ...]
    concentration_mg_ml_values: tuple[float, ...]
    seeds: tuple[int, ...]
    coarse: dict[str, int]
    refinement: dict[str, Any]
    classification: dict[str, float]
    state_model_id: str

    @classmethod
    def from_yaml(cls, filename: str | Path) -> "PhaseScanConfig":
        path = Path(filename).resolve()
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))["phase_scan"]
        root = path.parent.parent

        def resolved(value: str) -> Path:
            candidate = Path(value)
            return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

        state_config = resolved(raw["state_config"])
        state_raw = yaml.safe_load(state_config.read_text(encoding="utf-8"))
        config = cls(
            scan_id=str(raw["scan_id"]),
            state_config=state_config,
            md_config=resolved(raw["md_config"]),
            output_root=resolved(raw["output_root"]),
            pH_values=tuple(float(v) for v in raw["pH_values"]),
            nacl_mM_values=tuple(float(v) for v in raw["nacl_mM_values"]),
            concentration_mg_ml_values=tuple(
                float(v) for v in raw["concentration_mg_ml_values"]
            ),
            seeds=tuple(int(v) for v in raw["seeds"]),
            coarse={k: int(v) for k, v in raw["coarse"].items()},
            refinement=dict(raw.get("refinement", {})),
            classification={
                k: float(v) for k, v in raw.get("classification", {}).items()
            },
            state_model_id=str(state_raw["state_model"]["model_id"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        for name, values in (
            ("pH", self.pH_values),
            ("NaCl", self.nacl_mM_values),
            ("concentration", self.concentration_mg_ml_values),
        ):
            if not values or tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} values must be unique and increasing.")
        if self.pH_values[0] != 3.0 or self.pH_values[-1] != 9.0:
            raise ValueError("pH grid must include endpoints 3 and 9.")
        if self.nacl_mM_values[0] != 0.0 or self.nacl_mM_values[-1] != 500.0:
            raise ValueError("NaCl grid must include endpoints 0 and 500 mM.")
        if (
            self.concentration_mg_ml_values[0] != 0.1
            or self.concentration_mg_ml_values[-1] != 20.0
        ):
            raise ValueError("Concentration grid must include 0.1 and 20 mg/mL.")
        if not self.seeds:
            raise ValueError("At least one seed is required.")

    @property
    def state_count(self) -> int:
        return (
            len(self.pH_values)
            * len(self.nacl_mM_values)
            * len(self.concentration_mg_ml_values)
            * len(self.seeds)
        )

    def manifest_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for ph, salt, concentration, seed in itertools.product(
            self.pH_values,
            self.nacl_mM_values,
            self.concentration_mg_ml_values,
            self.seeds,
        ):
            state_id = (
                f"pH{tag_number(ph)}_nacl{tag_number(salt)}"
                f"_c{tag_number(concentration)}_seed{seed}"
            )
            rows.append(
                {
                    "task_id": len(rows),
                    "state_id": state_id,
                    "ph": ph,
                    "nacl_mM": salt,
                    "concentration_mg_ml": concentration,
                    "seed": seed,
                    "output_dir": portable_path(self.output_root / "coarse" / state_id),
                    "state_config": portable_path(self.state_config),
                    "md_config": portable_path(self.md_config),
                    "expected_state_model_id": self.state_model_id,
                }
            )
        return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def classify_run_directory(
    run_dir: str | Path,
    thresholds: dict[str, float] | None = None,
    direct_dir: str | Path | None = None,
    expected_model_id: str | None = None,
) -> dict[str, Any]:
    """Combine homogeneous and optional slab evidence for one state."""
    run = Path(run_dir)
    thresholds = thresholds or {}
    metadata = read_json(run / "metadata.json") or read_json(
        run / "hoomd_input_metadata.json"
    )
    observed_model_id = metadata.get("state_model_id")
    if expected_model_id is not None and observed_model_id != expected_model_id:
        return {
            "phase_conservative": "unresolved",
            "phase_operational": "unresolved",
            "internal_state": "state_model_mismatch",
            "confidence": "none",
            "rationale": (
                f"Expected state_model_id={expected_model_id!r}, observed "
                f"{observed_model_id!r}. Historical Hamiltonians cannot be mixed."
            ),
            "complete": False,
            "metadata_state_model_id": observed_model_id,
        }
    dynamics = read_json(run / "dynamics_analysis" / "dynamics_summary.json")
    if not dynamics:
        return {
            "phase_conservative": "unresolved",
            "phase_operational": "unresolved",
            "internal_state": "missing_dynamics_summary",
            "confidence": "none",
            "rationale": "The homogeneous-state dynamics analysis is missing.",
            "complete": False,
            **{f"metadata_{k}": v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))},
        }

    direct_run = Path(direct_dir) if direct_dir else run
    if direct_dir is not None and expected_model_id is not None:
        direct_metadata = read_json(direct_run / "metadata.json") or read_json(
            direct_run / "hoomd_input_metadata.json"
        )
        direct_model_id = direct_metadata.get("state_model_id")
        if direct_model_id != expected_model_id:
            return {
                "phase_conservative": "unresolved",
                "phase_operational": "unresolved",
                "internal_state": "direct_state_model_mismatch",
                "confidence": "none",
                "rationale": (
                    f"Expected slab state_model_id={expected_model_id!r}, "
                    f"observed {direct_model_id!r}."
                ),
                "complete": False,
                "metadata_state_model_id": direct_model_id,
            }
    direct = read_json(direct_run / "direct_coexistence_summary.json")
    slab = read_json(direct_run / "slab_dynamics_summary.json")
    if direct_dir is not None and (not direct or not slab):
        return {
            "phase_conservative": "unresolved",
            "phase_operational": "unresolved",
            "internal_state": "incomplete_direct_coexistence_evidence",
            "confidence": "none",
            "rationale": (
                "Both the axial-profile and slab-dynamics summaries are "
                "required before a direct-coexistence replicate is counted."
            ),
            "complete": False,
            "direct_coexistence_confirmed": False,
            "dynamic_exchange_confirmed": False,
        }
    profile_confirmed = bool(
        direct.get("profile_state") == "persistent_axial_inhomogeneity"
        and float(direct.get("late_mean_density_contrast", 0.0))
        >= thresholds.get("direct_profile_min_contrast", 0.50)
        and float(direct.get("late_mean_axial_cv", 0.0))
        >= thresholds.get("direct_profile_min_axial_cv", 0.75)
    )
    profile_dissolved = bool(
        direct.get("profile_state") == "weak_or_dissolved_profile"
        and float(direct.get("late_mean_density_contrast", 1.0)) <= 0.20
        and float(direct.get("late_mean_axial_cv", 1.0)) <= 0.35
    )
    exchange_confirmed = bool(
        direct.get("exchange_proxy_detected", False)
        and slab
        and (
            float(slab.get("mean_late_contact_turnover_fraction", 0.0))
            >= thresholds.get("slab_min_contact_turnover", 0.02)
            or float(slab.get("final_core_relative_msd_nm2", 0.0))
            >= thresholds.get("slab_min_core_msd_nm2", 0.05)
        )
    )

    metrics = MDPhaseMetrics(
        mean_clustered_fraction=float(dynamics["mean_clustered_fraction"]),
        mean_largest_cluster_fraction=float(
            dynamics["mean_largest_cluster_fraction"]
        ),
        percolation_fraction=float(dynamics["percolation_fraction"]),
        final_msd_nm2=float(dynamics["final_msd_nm2"]),
        frozen_finite_clusters=bool(dynamics["frozen_finite_clusters"]),
        direct_coexistence_confirmed=profile_confirmed,
        dynamic_exchange_confirmed=exchange_confirmed,
    )
    result = classify_md_phase(metrics)
    conservative = result.final_phase
    if conservative == "needs_review":
        conservative = "unresolved"

    # Operational labels guide expensive validation; they are never presented
    # as confirmed equilibrium LLPS.
    coarsening_slope = float(
        dynamics.get("late_largest_cluster_fraction_slope_per_ns", 0.0)
    )
    if direct_dir is not None and profile_dissolved and conservative == "soluble":
        operational = "soluble"
        internal_state = "slab_dissolved_to_single_phase"
        confidence = "high"
        rationale = "The initialized dense slab dissolved to a weak axial profile."
    elif direct_dir is not None and direct.get("profile_state") == "intermediate_profile":
        conservative = "unresolved"
        operational = "unresolved"
        internal_state = "intermediate_direct_coexistence_profile"
        confidence = "low"
        rationale = "The slab neither clearly persisted nor dissolved."
    elif conservative in {"llps", "aggregate"}:
        operational = conservative
        internal_state = result.internal_state
        confidence = result.confidence
        rationale = result.rationale
    elif (
        float(dynamics["mean_clustered_fraction"]) >= 0.80
        and coarsening_slope
        >= thresholds.get("homogeneous_coarsening_min_slope_per_ns", 0.02)
    ):
        operational = "llps_candidate"
        conservative = "unresolved"
        internal_state = result.internal_state
        confidence = result.confidence
        rationale = result.rationale
    else:
        operational = "soluble"
        internal_state = result.internal_state
        confidence = result.confidence
        rationale = result.rationale

    return {
        "phase_conservative": conservative,
        "phase_operational": operational,
        "internal_state": internal_state,
        "confidence": confidence,
        "rationale": rationale,
        "complete": True,
        "direct_coexistence_confirmed": profile_confirmed,
        "dynamic_exchange_confirmed": exchange_confirmed,
        **{k: dynamics.get(k) for k in (
            "mean_clustered_fraction",
            "mean_largest_cluster_fraction",
            "percolation_fraction",
            "final_initial_bond_survival",
            "mean_consecutive_bond_retention",
            "final_msd_nm2",
            "frozen_finite_clusters",
            "late_largest_cluster_fraction_slope_per_ns",
        )},
        **{f"metadata_{k}": metadata.get(k) for k in (
            "state_model_id",
            "charge_mapping",
            "protein_id",
            "added_salt_is_extrapolation",
            "charge_sign_reversal_is_extrapolation",
            "outside_strict_salr_regime",
        )},
    }


def extract_boundary_intervals(
    table: pd.DataFrame,
    phase_col: str = "phase_conservative",
    include_unresolved: bool = True,
) -> pd.DataFrame:
    """Return sampled brackets, not interpolated exact phase boundaries."""
    rows: list[dict[str, Any]] = []
    for axis in AXES:
        fixed = [name for name in AXES if name != axis]
        for fixed_values, group in table.groupby(fixed, dropna=False):
            fixed_values = (
                fixed_values if isinstance(fixed_values, tuple) else (fixed_values,)
            )
            ordered = group.sort_values(axis).reset_index(drop=True)
            for index in range(len(ordered) - 1):
                low, high = ordered.iloc[index], ordered.iloc[index + 1]
                if low[phase_col] == high[phase_col]:
                    continue
                if not include_unresolved and "unresolved" in {
                    str(low[phase_col]), str(high[phase_col])
                }:
                    continue
                low_value, high_value = float(low[axis]), float(high[axis])
                row = {
                    "scan_axis": axis,
                    "lower_sample": low_value,
                    "upper_sample": high_value,
                    "lower_phase": low[phase_col],
                    "upper_phase": high[phase_col],
                    "boundary_midpoint": (
                        math.sqrt(low_value * high_value)
                        if axis == "concentration_mg_ml" and low_value > 0
                        else 0.5 * (low_value + high_value)
                    ),
                    "boundary_is_interval": True,
                    "lower_state_id": low.get("state_id", ""),
                    "upper_state_id": high.get("state_id", ""),
                }
                row.update(dict(zip(fixed, fixed_values)))
                rows.append(row)
    return pd.DataFrame(rows)


def refinement_states(
    table: pd.DataFrame,
    boundaries: pd.DataFrame,
) -> pd.DataFrame:
    selected: dict[str, set[str]] = {}

    complete_ids = set(
        table.loc[table.get("complete", True).astype(bool), "state_id"].astype(str)
        if "complete" in table.columns
        else table["state_id"].astype(str)
    )

    def add(state_id: str, reason: str) -> None:
        value = str(state_id)
        if value in complete_ids:
            selected.setdefault(value, set()).add(reason)

    for _, row in table.iterrows():
        if not bool(row.get("complete", True)):
            continue
        if row.get("phase_conservative") == "unresolved":
            add(row["state_id"], "unresolved")
        if row.get("phase_operational") == "llps_candidate":
            add(row["state_id"], "homogeneous_llps_candidate")
        if row.get("confidence") in {"none", "low"}:
            add(row["state_id"], "low_confidence")
    for _, row in boundaries.iterrows():
        add(row["lower_state_id"], f"{row['scan_axis']}_boundary_neighbor")
        add(row["upper_state_id"], f"{row['scan_axis']}_boundary_neighbor")

    output = table[table["state_id"].astype(str).isin(selected)].copy()
    output["selection_reason"] = output["state_id"].astype(str).map(
        lambda value: ";".join(sorted(selected[value]))
    )
    return output


def expand_refinement_manifest(
    selected: pd.DataFrame,
    output_root: Path,
    seeds: Iterable[int],
    state_config: Path,
    md_config: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, source in selected.iterrows():
        for seed in seeds:
            base_id = str(source["state_id"]).rsplit("_seed", 1)[0]
            state_id = f"{base_id}_seed{int(seed)}"
            rows.append({
                "task_id": len(rows),
                "source_state_id": source["state_id"],
                "state_id": state_id,
                "ph": float(source["ph"]),
                "nacl_mM": float(source["nacl_mM"]),
                "concentration_mg_ml": float(source["concentration_mg_ml"]),
                "seed": int(seed),
                "output_dir": portable_path(output_root / "refinement" / state_id),
                "state_config": portable_path(state_config),
                "md_config": portable_path(md_config),
                "expected_state_model_id": source.get("expected_state_model_id", ""),
                "selection_reason": source.get("selection_reason", ""),
            })
    return pd.DataFrame(rows)


def build_direct_coexistence_manifest(
    selected: pd.DataFrame,
    output_root: Path,
    seeds: Iterable[int],
    state_config: Path,
    md_config: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, source in selected.iterrows():
        global_concentration = float(source["concentration_mg_ml"])
        initial_concentration = max(4.0 * global_concentration, 0.4)
        for seed in seeds:
            base_id = str(source["state_id"]).rsplit("_seed", 1)[0]
            state_id = f"{base_id}_slab_seed{int(seed)}"
            rows.append({
                "task_id": len(rows),
                "source_state_id": source["state_id"],
                "state_id": state_id,
                "ph": float(source["ph"]),
                "nacl_mM": float(source["nacl_mM"]),
                "global_concentration_mg_ml": global_concentration,
                "initial_slab_concentration_mg_ml": initial_concentration,
                "z_aspect_ratio": 3.0,
                "seed": int(seed),
                "output_dir": portable_path(output_root / "direct_coexistence" / state_id),
                "state_config": portable_path(state_config),
                "md_config": portable_path(md_config),
                "expected_state_model_id": source.get("expected_state_model_id", ""),
            })
    return pd.DataFrame(rows)
