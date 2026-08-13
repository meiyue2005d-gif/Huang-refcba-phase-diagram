#!/usr/bin/env python3
"""Validate the hard-sphere RDF over the refCBA concentration range."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from huang_md.hard_sphere import (
    cs_reduced_isothermal_compressibility,
)
from huang_md.hard_sphere_rdf import (
    MAX_PACKING_FRACTION,
    compressibility_from_rdf,
    hard_sphere_rdf_reduced,
    low_density_rdf_parameters,
)


OUTPUT_DIR = (
    ROOT
    / "results"
    / "hard_sphere_rdf_validation"
)

AVOGADRO = 6.02214076e23

MOLECULAR_WEIGHT_G_MOL = 8018.862
DIAMETER_NM = 4.278

CONCENTRATIONS_MG_ML = [
    0.1,
    0.436,
    0.5,
    2.0,
    5.0,
    10.0,
    20.0,
]


def concentration_to_number_density_nm3(
    concentration_mg_ml: float,
) -> float:
    # mg/mL is numerically equal to g/L.
    concentration_g_l = float(
        concentration_mg_ml
    )

    mol_per_litre = (
        concentration_g_l
        / MOLECULAR_WEIGHT_G_MOL
    )

    return (
        mol_per_litre
        * AVOGADRO
        / 1.0e24
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records: list[dict[str, float]] = []

    for concentration in CONCENTRATIONS_MG_ML:
        density = (
            concentration_to_number_density_nm3(
                concentration
            )
        )

        reduced_density = (
            density
            * DIAMETER_NM**3
        )

        eta = (
            np.pi
            * reduced_density
            / 6.0
        )

        parameters = (
            low_density_rdf_parameters(
                eta
            )
        )

        chi_rdf = compressibility_from_rdf(
            eta
        )

        chi_cs = float(
            np.asarray(
                cs_reduced_isothermal_compressibility(
                    eta
                )
            ).reshape(-1)[0]
        )

        records.append(
            {
                "concentration_mg_ml": concentration,
                "number_density_nm3": density,
                "reduced_density_rho_sigma3": (
                    reduced_density
                ),
                "packing_fraction": eta,
                "contact_rdf": (
                    parameters.contact_value
                ),
                "contact_amplitude": (
                    parameters.contact_amplitude
                ),
                "correction_amplitude": (
                    parameters.correction_amplitude
                ),
                "compressibility_from_rdf": chi_rdf,
                "compressibility_CS": chi_cs,
                "compressibility_absolute_error": abs(
                    chi_rdf - chi_cs
                ),
            }
        )

    table = pd.DataFrame.from_records(
        records
    )

    csv_file = (
        OUTPUT_DIR
        / "refcba_low_density_rdf_validation.csv"
    )

    table.to_csv(
        csv_file,
        index=False,
    )

    x = np.linspace(
        0.75,
        3.0,
        1000,
    )

    figure, axis = plt.subplots(
        figsize=(8.2, 5.5)
    )

    for concentration in [
        0.1,
        0.436,
        5.0,
        20.0,
    ]:
        row = table.loc[
            np.isclose(
                table["concentration_mg_ml"],
                concentration,
            )
        ].iloc[0]

        eta = float(
            row["packing_fraction"]
        )

        axis.plot(
            x,
            hard_sphere_rdf_reduced(
                x,
                eta,
            ),
            label=(
                f"{concentration:g} mg/mL, "
                f"eta={eta:.4f}"
            ),
        )

    axis.axvline(
        1.0,
        linewidth=1.0,
    )

    axis.axvline(
        2.0,
        linewidth=1.0,
        linestyle="--",
    )

    axis.set_xlabel(
        "Reduced distance r/sigma"
    )
    axis.set_ylabel(
        "Hard-sphere reference RDF g0(r)"
    )
    axis.set_title(
        "Low-density hard-sphere RDF for refCBA"
    )
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()

    figure_file = (
        OUTPUT_DIR
        / "refcba_low_density_rdf.png"
    )

    figure.savefig(
        figure_file,
        dpi=240,
    )

    plt.close(figure)

    checks = {
        "all states inside low-density range": bool(
            (
                table["packing_fraction"]
                <= MAX_PACKING_FRACTION
            ).all()
        ),
        "rho sigma^3 below 0.2": bool(
            (
                table[
                    "reduced_density_rho_sigma3"
                ]
                < 0.2
            ).all()
        ),
        "compressibility consistency": bool(
            (
                table[
                    "compressibility_absolute_error"
                ]
                < 2.0e-10
            ).all()
        ),
        "all values finite": bool(
            np.isfinite(
                table.select_dtypes(
                    include=[np.number]
                ).to_numpy()
            ).all()
        ),
    }

    print("=" * 78)
    print("Low-density hard-sphere RDF validation")
    print("=" * 78)

    print(
        table[
            [
                "concentration_mg_ml",
                "reduced_density_rho_sigma3",
                "packing_fraction",
                "contact_rdf",
                "compressibility_from_rdf",
                "compressibility_CS",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.9g}"
            ),
        )
    )

    print("\nValidation checks:")

    failed = []

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
            "Hard-sphere RDF validation failed: "
            + ", ".join(failed)
        )

    print(
        "\nAll low-density RDF validation checks passed."
    )


if __name__ == "__main__":
    main()
