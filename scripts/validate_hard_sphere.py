#!/usr/bin/env python3
"""Validate Carnahan-Starling hard-sphere thermodynamics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from huang_md.hard_sphere import (
    cs_compressibility_factor,
    cs_d_beta_pressure_d_density,
    cs_excess_chemical_potential,
    cs_excess_free_energy_per_particle,
    cs_reduced_isothermal_compressibility,
    number_density_from_packing_fraction,
)


OUTPUT_DIR = (
    ROOT
    / "results"
    / "hard_sphere_validation"
)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    diameter_nm = 4.278

    eta = np.linspace(
        1.0e-5,
        0.55,
        800,
    )

    density = (
        number_density_from_packing_fraction(
            eta,
            diameter_nm,
        )
    )

    z_value = (
        cs_compressibility_factor(eta)
    )

    a_ex = (
        cs_excess_free_energy_per_particle(
            eta
        )
    )

    mu_ex = (
        cs_excess_chemical_potential(
            eta
        )
    )

    pressure_derivative = (
        cs_d_beta_pressure_d_density(
            eta
        )
    )

    reduced_compressibility = (
        cs_reduced_isothermal_compressibility(
            eta
        )
    )

    identity_error = np.max(
        np.abs(
            mu_ex
            - (
                a_ex
                + z_value
                - 1.0
            )
        )
    )

    inverse_error = np.max(
        np.abs(
            reduced_compressibility
            * pressure_derivative
            - 1.0
        )
    )

    table = pd.DataFrame(
        {
            "packing_fraction": eta,
            "number_density_nm3": density,
            "compressibility_factor_Z": z_value,
            "beta_Aex_per_particle": a_ex,
            "beta_mu_ex": mu_ex,
            "d_betaP_d_density": pressure_derivative,
            "reduced_isothermal_compressibility": (
                reduced_compressibility
            ),
        }
    )

    csv_file = (
        OUTPUT_DIR
        / "carnahan_starling_reference.csv"
    )

    table.to_csv(
        csv_file,
        index=False,
    )

    figure, axis = plt.subplots(
        figsize=(8.0, 5.4)
    )

    axis.plot(
        eta,
        z_value,
        label="Compressibility factor Z",
    )

    axis.plot(
        eta,
        a_ex,
        label="Excess free energy",
    )

    axis.plot(
        eta,
        mu_ex,
        label="Excess chemical potential",
    )

    axis.set_xlabel("Packing fraction")
    axis.set_ylabel("Reduced thermodynamic quantity")
    axis.set_title(
        "Carnahan-Starling hard-sphere reference"
    )
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()

    figure_file = (
        OUTPUT_DIR
        / "carnahan_starling_thermodynamics.png"
    )

    figure.savefig(
        figure_file,
        dpi=240,
    )

    plt.close(figure)

    checks = {
        "all quantities finite": bool(
            np.isfinite(
                table.to_numpy()
            ).all()
        ),
        "positive pressure derivative": bool(
            (
                pressure_derivative
                > 0.0
            ).all()
        ),
        "positive compressibility": bool(
            (
                reduced_compressibility
                > 0.0
            ).all()
        ),
        "free-energy identity": (
            identity_error < 1.0e-12
        ),
        "compressibility inverse identity": (
            inverse_error < 1.0e-12
        ),
    }

    print("=" * 76)
    print("Carnahan-Starling hard-sphere validation")
    print("=" * 76)
    print(f"Hard-sphere diameter : {diameter_nm:.6f} nm")
    print(f"Maximum eta         : {eta.max():.6f}")
    print(
        "Maximum free-energy identity error : "
        f"{identity_error:.6e}"
    )
    print(
        "Maximum inverse identity error     : "
        f"{inverse_error:.6e}"
    )

    failed = []

    print("\nValidation checks:")

    for name, passed in checks.items():
        label = "PASS" if passed else "FAIL"
        print(f"  {label:<4s} {name}")

        if not passed:
            failed.append(name)

    print("\nGenerated files:")
    print(f"  {csv_file}")
    print(f"  {figure_file}")

    if failed:
        raise RuntimeError(
            "Hard-sphere validation failed: "
            + ", ".join(failed)
        )

    print(
        "\nAll hard-sphere validation checks passed."
    )


if __name__ == "__main__":
    main()
