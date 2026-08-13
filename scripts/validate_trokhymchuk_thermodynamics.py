#!/usr/bin/env python3
"""Validate thermodynamic consistency of the Trokhymchuk RDF.

The dimensionless compressibility obtained from the RDF is

    K_T = 1 + 24 eta * integral_0^infinity
          [g(x) - 1] x^2 dx

where x = r / sigma.

It is compared with Eq. 26 of Trokhymchuk et al. (2005).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad

from huang_md.trokhymchuk_rdf import (
    calculate_trokhymchuk_rdf_coefficients,
    depletion_rdf_value,
    structural_rdf_value,
)


OUTPUT_DIR = (
    ROOT
    / "results"
    / "trokhymchuk_thermodynamic_validation"
)


def target_compressibility(
    packing_fraction: float,
) -> float:
    """Kolafa compressibility from paper Eq. 26."""
    eta = float(packing_fraction)

    numerator = (
        3.0
        * (1.0 - eta) ** 4
    )

    denominator = (
        3.0
        + 12.0 * eta
        + 12.0 * eta**2
        - 8.0 * eta**3
        - 8.0 * eta**4
        + 4.0 * eta**5
    )

    return numerator / denominator


def rdf_compressibility(
    reduced_density: float,
) -> tuple[float, float, float, float]:
    """Calculate compressibility and integral contributions."""
    coefficients = (
        calculate_trokhymchuk_rdf_coefficients(
            reduced_density
        )
    )

    p = coefficients.base
    eta = p.packing_fraction
    merge = p.minimum_position

    # For 0 <= x < 1, g(x)=0:
    #
    # integral [g(x)-1] x^2 dx = -1/3
    core_integral = -1.0 / 3.0

    depletion_integral, depletion_error = quad(
        lambda x: (
            depletion_rdf_value(
                x,
                coefficients,
            )
            - 1.0
        )
        * x**2,
        1.0,
        merge,
        epsabs=1.0e-11,
        epsrel=1.0e-10,
        limit=300,
    )

    structural_integral, structural_error = quad(
        lambda x: (
            structural_rdf_value(
                x,
                coefficients,
            )
            - 1.0
        )
        * x**2,
        merge,
        np.inf,
        epsabs=1.0e-11,
        epsrel=1.0e-10,
        limit=500,
    )

    total_integral = (
        core_integral
        + depletion_integral
        + structural_integral
    )

    compressibility = (
        1.0
        + 24.0
        * eta
        * total_integral
    )

    integration_error = (
        depletion_error
        + structural_error
    )

    return (
        float(compressibility),
        float(core_integral),
        float(depletion_integral),
        float(structural_integral),
        float(integration_error),
    )


def first_coordination_number(
    reduced_density: float,
) -> float:
    """Calculate Eq. 38 first-shell coordination number."""
    coefficients = (
        calculate_trokhymchuk_rdf_coefficients(
            reduced_density
        )
    )

    p = coefficients.base

    integral, _ = quad(
        lambda x: (
            depletion_rdf_value(
                x,
                coefficients,
            )
            * x**2
        ),
        1.0,
        p.minimum_position,
        epsabs=1.0e-11,
        epsrel=1.0e-10,
        limit=300,
    )

    return float(
        24.0
        * p.packing_fraction
        * integral
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    densities = np.linspace(
        0.2,
        0.9,
        71,
    )

    records = []

    for reduced_density in densities:
        coefficients = (
            calculate_trokhymchuk_rdf_coefficients(
                float(reduced_density)
            )
        )

        p = coefficients.base

        (
            rdf_kt,
            core_integral,
            depletion_integral,
            structural_integral,
            integration_error,
        ) = rdf_compressibility(
            float(reduced_density)
        )

        target_kt = target_compressibility(
            p.packing_fraction
        )

        absolute_error = abs(
            rdf_kt - target_kt
        )

        relative_error = (
            absolute_error
            / abs(target_kt)
        )

        contact_from_coefficients = (
            coefficients.coefficient_a
            + coefficients.coefficient_b
            * np.cos(p.gamma)
        )

        records.append(
            {
                "reduced_density": reduced_density,
                "packing_fraction": p.packing_fraction,
                "contact_rdf_target": p.contact_rdf,
                "contact_rdf_calculated": (
                    contact_from_coefficients
                ),
                "contact_absolute_error": abs(
                    contact_from_coefficients
                    - p.contact_rdf
                ),
                "rdf_compressibility": rdf_kt,
                "target_compressibility": target_kt,
                "compressibility_absolute_error": (
                    absolute_error
                ),
                "compressibility_relative_error": (
                    relative_error
                ),
                "core_integral": core_integral,
                "depletion_integral": (
                    depletion_integral
                ),
                "structural_integral": (
                    structural_integral
                ),
                "integration_error_estimate": (
                    integration_error
                ),
                "first_coordination_number": (
                    first_coordination_number(
                        float(reduced_density)
                    )
                ),
            }
        )

    table = pd.DataFrame.from_records(
        records
    )

    csv_file = (
        OUTPUT_DIR
        / "thermodynamic_consistency.csv"
    )

    table.to_csv(
        csv_file,
        index=False,
    )

    figure, axis = plt.subplots(
        figsize=(8.2, 5.5)
    )

    axis.plot(
        table["reduced_density"],
        table["target_compressibility"],
        label="Eq. 26 target",
    )

    axis.plot(
        table["reduced_density"],
        table["rdf_compressibility"],
        linestyle="--",
        label="RDF integral",
    )

    axis.set_xlabel(
        r"Reduced density $\rho\sigma^3$"
    )
    axis.set_ylabel(
        r"Reduced compressibility $K_T$"
    )
    axis.set_title(
        "Trokhymchuk RDF thermodynamic consistency"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()

    compressibility_plot = (
        OUTPUT_DIR
        / "compressibility_comparison.png"
    )

    figure.savefig(
        compressibility_plot,
        dpi=240,
    )

    plt.close(figure)

    figure, axis = plt.subplots(
        figsize=(8.2, 5.5)
    )

    axis.plot(
        table["reduced_density"],
        table[
            "compressibility_absolute_error"
        ],
    )

    axis.set_xlabel(
        r"Reduced density $\rho\sigma^3$"
    )
    axis.set_ylabel(
        "Absolute compressibility error"
    )
    axis.set_title(
        "Compressibility consistency residual"
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()

    error_plot = (
        OUTPUT_DIR
        / "compressibility_absolute_error.png"
    )

    figure.savefig(
        error_plot,
        dpi=240,
    )

    plt.close(figure)

    representative = table.iloc[
        [0, 30, 70]
    ][
        [
            "reduced_density",
            "rdf_compressibility",
            "target_compressibility",
            "compressibility_absolute_error",
            "compressibility_relative_error",
            "first_coordination_number",
        ]
    ]

    print("=" * 84)
    print(
        "Trokhymchuk RDF thermodynamic validation"
    )
    print("=" * 84)

    print(
        representative.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.10g}"
            ),
        )
    )

    maximum_contact_error = float(
        table[
            "contact_absolute_error"
        ].max()
    )

    maximum_absolute_error = float(
        table[
            "compressibility_absolute_error"
        ].max()
    )

    maximum_relative_error = float(
        table[
            "compressibility_relative_error"
        ].max()
    )

    maximum_error_row = table.loc[
        table[
            "compressibility_absolute_error"
        ].idxmax()
    ]

    print("\nMaximum errors:")
    print(
        "  contact absolute error       = "
        f"{maximum_contact_error:.6e}"
    )
    print(
        "  compressibility abs. error   = "
        f"{maximum_absolute_error:.6e}"
    )
    print(
        "  compressibility rel. error   = "
        f"{maximum_relative_error:.6e}"
    )
    print(
        "  maximum error density        = "
        f"{maximum_error_row['reduced_density']:.6g}"
    )

    checks = {
        "all values finite": bool(
            np.isfinite(
                table.select_dtypes(
                    include=[np.number]
                ).to_numpy()
            ).all()
        ),
        "all compressibilities positive": bool(
            (
                table["rdf_compressibility"]
                > 0.0
            ).all()
        ),
        "contact condition": bool(
            maximum_contact_error
            < 1.0e-10
        ),
        "compressibility absolute residual": bool(
            maximum_absolute_error
            < 6.0e-3
        ),
        "integration convergence": bool(
            (
                table[
                    "integration_error_estimate"
                ]
                < 1.0e-8
            ).all()
        ),
    }

    failed = []

    print("\nValidation checks:")

    for name, passed in checks.items():
        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"  {status:<4s} {name}"
        )

        if not passed:
            failed.append(name)

    print("\nGenerated files:")
    print(f"  {csv_file}")
    print(f"  {compressibility_plot}")
    print(f"  {error_plot}")

    if failed:
        raise RuntimeError(
            "Thermodynamic validation failed: "
            + ", ".join(failed)
        )

    print(
        "\nAll Trokhymchuk thermodynamic "
        "checks passed."
    )


if __name__ == "__main__":
    main()
