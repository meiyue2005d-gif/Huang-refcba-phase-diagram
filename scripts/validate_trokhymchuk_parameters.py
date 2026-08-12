#!/usr/bin/env python3
"""Generate and inspect the Trokhymchuk high-density parameters."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from huang_md.trokhymchuk_parameters import (
    calculate_trokhymchuk_parameters,
)


OUTPUT_DIR = (
    ROOT
    / "results"
    / "trokhymchuk_parameter_validation"
)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    densities = np.linspace(
        0.2,
        0.9,
        141,
    )

    records = []

    for density in densities:
        result = (
            calculate_trokhymchuk_parameters(
                float(density)
            )
        )

        records.append(
            result.__dict__
        )

    table = pd.DataFrame.from_records(
        records
    )

    csv_file = (
        OUTPUT_DIR
        / "trokhymchuk_parameters.csv"
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
        table["contact_rdf"],
        label="Contact RDF",
    )

    axis.plot(
        table["reduced_density"],
        table["minimum_rdf"],
        label="First-minimum RDF",
    )

    axis.set_xlabel(
        r"Reduced density $\rho\sigma^3$"
    )
    axis.set_ylabel("RDF value")
    axis.set_title(
        "Trokhymchuk hard-sphere RDF landmarks"
    )
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()

    figure_file = (
        OUTPUT_DIR
        / "rdf_landmarks_vs_density.png"
    )

    figure.savefig(
        figure_file,
        dpi=240,
    )

    plt.close(figure)

    checks = {
        "all values finite": bool(
            np.isfinite(
                table.to_numpy(
                    dtype=np.float64
                )
            ).all()
        ),
        "contact RDF positive": bool(
            (
                table["contact_rdf"]
                > 0.0
            ).all()
        ),
        "minimum RDF positive": bool(
            (
                table["minimum_rdf"]
                > 0.0
            ).all()
        ),
        "minimum outside hard core": bool(
            (
                table["minimum_position"]
                > 1.0
            ).all()
        ),
        "contact exceeds minimum": bool(
            (
                table["contact_rdf"]
                > table["minimum_rdf"]
            ).all()
        ),
    }

    print("=" * 78)
    print("Trokhymchuk parameter validation")
    print("=" * 78)

    print(
        table.loc[
            table["reduced_density"].isin(
                [0.2, 0.5, 0.9]
            ),
            [
                "reduced_density",
                "packing_fraction",
                "mu",
                "alpha",
                "beta",
                "omega",
                "kappa",
                "minimum_position",
                "minimum_rdf",
                "contact_rdf",
            ],
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
            "Trokhymchuk parameter validation failed: "
            + ", ".join(failed)
        )

    print(
        "\nAll Trokhymchuk parameter checks passed."
    )


if __name__ == "__main__":
    main()
