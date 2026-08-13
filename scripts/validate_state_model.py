#!/usr/bin/env python3
"""Validate pH- and salt-dependent refCBA potential parameters."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from huang_md.electrostatics import (
    estimate_isoelectric_point,
    net_charge,
)
from huang_md.parameters import HuangPotentialParameters
from huang_md.potential import (
    reduced_to_distance_nm,
    total_potential_reduced,
)
from huang_md.state_model import (
    RefCBAStateModel,
    calculate_K2_kBT,
    calculate_Z2,
    ionic_strength_mM,
    parameters_for_state,
)


BASELINE_CONFIG = ROOT / "configs" / "huang_baseline.yaml"
STATE_CONFIG = ROOT / "configs" / "refcba_state_model.yaml"
OUTPUT_DIR = ROOT / "results" / "state_model_validation"


def potential_landmarks(
    x: np.ndarray,
    potential: np.ndarray,
) -> tuple[float, float, float, float]:
    well_mask = (x >= 0.80) & (x <= 1.15)
    barrier_mask = (x >= 1.00) & (x <= 3.00)

    well_indices = np.where(well_mask)[0]
    barrier_indices = np.where(barrier_mask)[0]

    well_index = well_indices[
        np.argmin(potential[well_indices])
    ]
    barrier_index = barrier_indices[
        np.argmax(potential[barrier_indices])
    ]

    return (
        float(x[well_index]),
        float(potential[well_index]),
        float(x[barrier_index]),
        float(potential[barrier_index]),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = HuangPotentialParameters.from_yaml(
        BASELINE_CONFIG
    )
    baseline.validate()

    model = RefCBAStateModel.from_yaml(STATE_CONFIG)

    pH_grid = np.linspace(3.0, 9.0, 121)
    salt_grid = np.arange(0.0, 500.0 + 1.0e-9, 10.0)

    records: list[dict[str, float]] = []

    for salt in salt_grid:
        for pH in pH_grid:
            state = parameters_for_state(
                baseline,
                model,
                pH=float(pH),
                added_nacl_mM=float(salt),
            )

            records.append(
                {
                    "pH": float(pH),
                    "added_NaCl_mM": float(salt),
                    "total_ionic_strength_mM": float(
                        ionic_strength_mM(
                            np.array([salt]),
                            model,
                        )[0]
                    ),
                    "net_charge_e": float(
                        net_charge(np.array([pH]))[0]
                    ),
                    "absolute_charge_e": float(
                        abs(net_charge(np.array([pH]))[0])
                    ),
                    "K2_kBT": state.K2_kBT,
                    "Z2": state.Z2,
                }
            )

    parameter_table = pd.DataFrame.from_records(records)
    parameter_table.to_csv(
        OUTPUT_DIR / "state_parameter_grid.csv",
        index=False,
    )

    selected_pH = [
        3.0,
        4.0,
        4.5,
        estimate_isoelectric_point(),
        5.5,
        7.0,
        9.0,
    ]
    selected_salt = [0.0, 100.0, 500.0]

    x = np.linspace(
        0.65,
        baseline.cutoff_reduced,
        12000,
    )
    distance_nm = reduced_to_distance_nm(x, baseline)

    summary_records: list[dict[str, float]] = []

    figure, axis = plt.subplots(figsize=(9.0, 6.0))

    for salt in selected_salt:
        for pH in selected_pH:
            state = parameters_for_state(
                baseline,
                model,
                pH=float(pH),
                added_nacl_mM=float(salt),
            )

            potential = total_potential_reduced(x, state)

            well_x, well_u, barrier_x, barrier_u = (
                potential_landmarks(x, potential)
            )

            summary_records.append(
                {
                    "pH": float(pH),
                    "added_NaCl_mM": float(salt),
                    "net_charge_e": float(
                        net_charge(np.array([pH]))[0]
                    ),
                    "K2_kBT": state.K2_kBT,
                    "Z2": state.Z2,
                    "well_x": well_x,
                    "well_r_nm": (
                        well_x * baseline.diameter_nm
                    ),
                    "well_U_kBT": well_u,
                    "barrier_x": barrier_x,
                    "barrier_r_nm": (
                        barrier_x * baseline.diameter_nm
                    ),
                    "barrier_U_kBT": barrier_u,
                }
            )

            if salt in (0.0, 500.0) and pH in (
                3.0,
                4.5,
                5.5,
                9.0,
            ):
                axis.plot(
                    distance_nm,
                    potential,
                    linewidth=1.5,
                    label=(
                        f"pH {pH:.2f}, "
                        f"NaCl {salt:.0f} mM"
                    ),
                )

    axis.axhline(0.0, linewidth=0.8)
    axis.set_xlim(2.8, 14.0)
    axis.set_ylim(-65.0, 120.0)
    axis.set_xlabel("Center-to-center distance r (nm)")
    axis.set_ylabel("Interaction potential U(r) / kBT")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "selected_state_potentials.png",
        dpi=240,
    )
    plt.close(figure)

    summary_table = pd.DataFrame.from_records(
        summary_records
    )
    summary_table.to_csv(
        OUTPUT_DIR / "selected_state_summary.csv",
        index=False,
    )

    figure, axis = plt.subplots(figsize=(8.0, 5.5))

    for salt in selected_salt:
        K2_values = calculate_K2_kBT(
            pH_grid,
            np.full_like(pH_grid, salt),
            model,
        )

        axis.plot(
            pH_grid,
            K2_values,
            linewidth=2.0,
            label=f"NaCl {salt:.0f} mM",
        )

    axis.axvline(
        estimate_isoelectric_point(),
        linestyle="--",
        linewidth=1.0,
        label="Estimated pI",
    )
    axis.set_xlabel("pH")
    axis.set_ylabel("Repulsive amplitude K2 / kBT")
    axis.set_xlim(3.0, 9.0)
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "K2_vs_pH.png",
        dpi=240,
    )
    plt.close(figure)

    z2_values = calculate_Z2(salt_grid, model)

    figure, axis = plt.subplots(figsize=(8.0, 5.5))
    axis.plot(salt_grid, z2_values, linewidth=2.0)
    axis.set_xlabel("Added NaCl (mM)")
    axis.set_ylabel("Calibrated screening parameter Z2")
    axis.set_xlim(0.0, 500.0)
    axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "calibrated_Z2_vs_NaCl.png",
        dpi=240,
    )
    plt.close(figure)

    reference_state = parameters_for_state(
        baseline,
        model,
        pH=4.5,
        added_nacl_mM=0.0,
    )

    pI = estimate_isoelectric_point()

    pI_state = parameters_for_state(
        baseline,
        model,
        pH=pI,
        added_nacl_mM=0.0,
    )

    high_salt_state = parameters_for_state(
        baseline,
        model,
        pH=4.5,
        added_nacl_mM=500.0,
    )

    checks = {
        "reference K2": abs(
            reference_state.K2_kBT - 53.056
        ) < 1.0e-8,
        "reference Z2": abs(
            reference_state.Z2 - 1.483
        ) < 1.0e-8,
        "K2 near pI": pI_state.K2_kBT < 0.05,
        "salt raises Z2": (
            high_salt_state.Z2 > reference_state.Z2
        ),
        "salt lowers K2": (
            high_salt_state.K2_kBT
            < reference_state.K2_kBT
        ),
    }

    print("=" * 76)
    print("refCBA pH-salt state model validation")
    print("=" * 76)

    print(f"Estimated pI              : {pI:.6f}")

    print("\nReference state: pH 4.5, added NaCl 0 mM")
    print(
        f"  K2                      : "
        f"{reference_state.K2_kBT:.6f} kBT"
    )
    print(
        f"  Z2                      : "
        f"{reference_state.Z2:.6f}"
    )

    print("\nAt estimated pI, added NaCl 0 mM")
    print(
        f"  K2                      : "
        f"{pI_state.K2_kBT:.8f} kBT"
    )
    print(
        f"  Z2                      : "
        f"{pI_state.Z2:.6f}"
    )

    print("\nAt pH 4.5, added NaCl 500 mM")
    print(
        f"  K2                      : "
        f"{high_salt_state.K2_kBT:.6f} kBT"
    )
    print(
        f"  Z2                      : "
        f"{high_salt_state.Z2:.6f}"
    )

    print("\nValidation checks:")

    failed: list[str] = []

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status:<4s}  {name}")

        if not passed:
            failed.append(name)

    print("\nGenerated files:")

    for output in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {output}")

    if failed:
        raise RuntimeError(
            "State model validation failed: "
            + ", ".join(failed)
        )

    print("\nAll state-model checks passed.")


if __name__ == "__main__":
    main()
