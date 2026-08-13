#!/usr/bin/env python3
"""Validate the pH 4.5 Huang baseline potential."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from huang_md.parameters import HuangPotentialParameters
from huang_md.potential import (
    attractive_yukawa_reduced,
    force_reduced,
    reduced_to_distance_nm,
    repulsive_yukawa_reduced,
    total_potential_reduced,
)


CONFIG = ROOT / "configs" / "huang_baseline.yaml"
OUTPUT_DIR = ROOT / "results" / "potential_validation"


def find_extrema(
    x: np.ndarray,
    potential: np.ndarray,
) -> tuple[int, int]:
    well_mask = (x >= 0.80) & (x <= 1.10)
    barrier_mask = (x >= 1.00) & (x <= 2.50)

    well_indices = np.where(well_mask)[0]
    barrier_indices = np.where(barrier_mask)[0]

    well_index = well_indices[
        np.argmin(potential[well_indices])
    ]

    barrier_index = barrier_indices[
        np.argmax(potential[barrier_indices])
    ]

    return int(well_index), int(barrier_index)


def save_plot(
    distance_nm: np.ndarray,
    total: np.ndarray,
    attractive: np.ndarray,
    repulsive: np.ndarray,
    diameter_nm: float,
    output_file: Path,
    ylim: tuple[float, float],
) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 5.5))

    axis.plot(
        distance_nm,
        total,
        linewidth=2.2,
        label="Total potential",
    )

    outer = distance_nm >= diameter_nm

    axis.plot(
        distance_nm[outer],
        attractive[outer],
        linestyle="--",
        linewidth=1.5,
        label="Attractive Yukawa",
    )

    axis.plot(
        distance_nm[outer],
        repulsive[outer],
        linestyle=":",
        linewidth=1.8,
        label="Repulsive Yukawa",
    )

    axis.axhline(0.0, linewidth=0.8)
    axis.axvline(
        diameter_nm,
        linestyle="--",
        linewidth=1.0,
        label="Particle diameter",
    )

    axis.set_xlabel("Center-to-center distance r (nm)")
    axis.set_ylabel("Interaction potential U(r) / kBT")
    axis.set_ylim(*ylim)
    axis.set_xlim(distance_nm.min(), distance_nm.max())
    axis.legend()
    axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_file, dpi=240)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    params = HuangPotentialParameters.from_yaml(CONFIG)
    params.validate(require_salr=True)

    x = np.linspace(
        0.45,
        params.cutoff_reduced,
        20000,
        dtype=np.float64,
    )

    distance_nm = reduced_to_distance_nm(x, params)

    total = total_potential_reduced(x, params)
    attractive = attractive_yukawa_reduced(x, params)
    repulsive = repulsive_yukawa_reduced(x, params)
    reduced_force = force_reduced(x, params)

    well_index, barrier_index = find_extrema(x, total)

    well_x = x[well_index]
    well_r_nm = distance_nm[well_index]
    well_u = total[well_index]

    barrier_x = x[barrier_index]
    barrier_r_nm = distance_nm[barrier_index]
    barrier_u = total[barrier_index]

    contact_u = float(
        total_potential_reduced(
            np.array([1.0]),
            params,
        )[0]
    )

    left_x = np.array([1.0 - 1.0e-7])
    right_x = np.array([1.0 + 1.0e-7])

    left_u = float(total_potential_reduced(left_x, params)[0])
    right_u = float(total_potential_reduced(right_x, params)[0])

    left_f = float(force_reduced(left_x, params)[0])
    right_f = float(force_reduced(right_x, params)[0])

    potential_jump = abs(left_u - right_u)
    force_jump = abs(left_f - right_f)

    table = pd.DataFrame(
        {
            "x_reduced": x,
            "distance_nm": distance_nm,
            "potential_kBT": total,
            "force_reduced": reduced_force,
            "attractive_yukawa_kBT": attractive,
            "repulsive_yukawa_kBT": repulsive,
        }
    )

    csv_file = OUTPUT_DIR / "huang_pH4.5_potential.csv"
    table.to_csv(csv_file, index=False)

    save_plot(
        distance_nm=distance_nm,
        total=total,
        attractive=attractive,
        repulsive=repulsive,
        diameter_nm=params.diameter_nm,
        output_file=OUTPUT_DIR / "huang_pH4.5_potential_full.png",
        ylim=(-20.0, 120.0),
    )

    save_plot(
        distance_nm=distance_nm,
        total=total,
        attractive=attractive,
        repulsive=repulsive,
        diameter_nm=params.diameter_nm,
        output_file=OUTPUT_DIR / "huang_pH4.5_potential_zoom.png",
        ylim=(-12.0, 12.0),
    )

    print("=" * 72)
    print("Huang pH 4.5 potential validation")
    print("=" * 72)

    print(f"Configuration       : {CONFIG}")
    print(f"Particle diameter   : {params.diameter_nm:.6f} nm")
    print(f"K1                  : {params.K1_kBT:.6f} kBT")
    print(f"Z1                  : {params.Z1:.6f}")
    print(f"K2                  : {params.K2_kBT:.6f} kBT")
    print(f"Z2                  : {params.Z2:.6f}")
    print(f"Gaussian sigma      : {params.gaussian_sigma_reduced:.6f}")
    print(f"Gaussian epsilon    : {params.gaussian_epsilon_kBT:.3e} kBT")

    print("\nCalculated landmarks:")
    print(
        f"  U(x=1)            : {contact_u:.6f} kBT "
        "(expected -6.944)"
    )
    print(
        f"  Attractive minimum: x={well_x:.6f}, "
        f"r={well_r_nm:.6f} nm, U={well_u:.6f} kBT"
    )
    print(
        f"  Repulsive barrier : x={barrier_x:.6f}, "
        f"r={barrier_r_nm:.6f} nm, U={barrier_u:.6f} kBT"
    )

    print("\nContinuity near x=1:")
    print(f"  Potential jump    : {potential_jump:.6e} kBT")
    print(f"  Force jump        : {force_jump:.6e}")

    checks = {
        "contact energy": abs(contact_u + 6.944) < 1.0e-6,
        "well distance": 4.0 <= well_r_nm <= 4.4,
        "well depth": -9.0 <= well_u <= -6.0,
        "barrier distance": 5.8 <= barrier_r_nm <= 6.6,
        "barrier height": 6.5 <= barrier_u <= 8.5,
        "potential continuity": potential_jump < 1.0e-3,
        "force continuity": force_jump < 1.0e-2,
    }

    print("\nValidation checks:")

    failed: list[str] = []

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status:<4s}  {name}")

        if not passed:
            failed.append(name)

    print("\nGenerated files:")
    print(f"  {csv_file}")
    print(f"  {OUTPUT_DIR / 'huang_pH4.5_potential_full.png'}")
    print(f"  {OUTPUT_DIR / 'huang_pH4.5_potential_zoom.png'}")

    if failed:
        raise RuntimeError(
            "Potential validation failed: "
            + ", ".join(failed)
        )

    print("\nAll baseline potential checks passed.")


if __name__ == "__main__":
    main()
