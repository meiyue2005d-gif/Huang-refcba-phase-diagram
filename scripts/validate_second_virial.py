#!/usr/bin/env python3
"""Validate B2 across representative refCBA pH states."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from huang_md.parameters import (
    HuangPotentialParameters,
)
from huang_md.state_model import (
    RefCBAStateModel,
    parameters_for_state,
)
from huang_md.virial import (
    calculate_virial_properties,
)


OUTPUT_DIR = (
    ROOT
    / "results"
    / "virial_validation"
)


PH_VALUES = [
    3.0,
    4.0,
    4.3,
    4.5,
    4.7,
    4.8,
    4.8852,
    5.0,
    5.2,
    5.5,
    6.0,
    7.0,
    8.0,
    9.0,
]


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline = (
        HuangPotentialParameters.from_yaml(
            ROOT
            / "configs"
            / "huang_baseline.yaml"
        )
    )

    state_model = (
        RefCBAStateModel.from_yaml(
            ROOT
            / "configs"
            / "refcba_state_model.yaml"
        )
    )

    records: list[dict[str, float | bool]] = []

    for pH in PH_VALUES:
        params = parameters_for_state(
            baseline=baseline,
            model=state_model,
            pH=pH,
            added_nacl_mM=0.0,
        )

        virial = calculate_virial_properties(
            params
        )

        records.append(
            {
                "pH": pH,
                "added_NaCl_mM": 0.0,
                "K1_kBT": params.K1_kBT,
                "Z1": params.Z1,
                "K2_kBT": params.K2_kBT,
                "Z2": params.Z2,
                "strict_salr": params.is_salr,
                "full_B2_reduced": (
                    virial.full_B2_reduced
                ),
                "full_B2_nm3": (
                    virial.full_B2_nm3
                ),
                "core_B2_reduced": (
                    virial.core_B2_reduced
                ),
                "effective_hs_diameter_reduced": (
                    virial.effective_hs_diameter_reduced
                ),
                "effective_hs_diameter_nm": (
                    virial.effective_hs_diameter_nm
                ),
                "B2star": (
                    virial.reduced_second_virial_B2star
                ),
                "full_integration_error": (
                    virial.full_integration_error
                ),
            }
        )

    table = pd.DataFrame.from_records(
        records
    )

    csv_file = (
        OUTPUT_DIR
        / "refcba_B2_vs_pH_NaCl0.csv"
    )

    table.to_csv(
        csv_file,
        index=False,
    )

    effective_diameter_spread = float(
        table[
            "effective_hs_diameter_reduced"
        ].max()
        - table[
            "effective_hs_diameter_reduced"
        ].min()
    )

    pI_row = table.iloc[
        (
            table["pH"] - 4.8852
        ).abs().argmin()
    ]

    pH3_row = table.iloc[
        (
            table["pH"] - 3.0
        ).abs().argmin()
    ]

    pH9_row = table.iloc[
        (
            table["pH"] - 9.0
        ).abs().argmin()
    ]

    checks = {
        "core B2 positive": bool(
            (
                table["core_B2_reduced"]
                > 0
            ).all()
        ),
        "effective diameter state independent": (
            effective_diameter_spread
            < 1.0e-10
        ),
        "all virial values finite": bool(
            np.isfinite(
                table[
                    [
                        "full_B2_reduced",
                        "core_B2_reduced",
                        "B2star",
                    ]
                ].to_numpy()
            ).all()
        ),
        "pI more attractive than pH 3": (
            float(pI_row["B2star"])
            < float(pH3_row["B2star"])
        ),
        "pI more attractive than pH 9": (
            float(pI_row["B2star"])
            < float(pH9_row["B2star"])
        ),
    }

    figure, axis = plt.subplots(
        figsize=(8.0, 5.4)
    )

    axis.plot(
        table["pH"],
        table["B2star"],
        marker="o",
    )

    axis.axhline(
        0.0,
        linewidth=1.0,
    )

    axis.set_xlabel("pH")
    axis.set_ylabel(
        r"Reduced second virial coefficient "
        r"$B_2/B_2^{HS}$"
    )
    axis.set_title(
        "refCBA virial response at added NaCl = 0 mM"
    )
    axis.grid(alpha=0.25)

    figure.tight_layout()

    plot_file = (
        OUTPUT_DIR
        / "refcba_B2star_vs_pH_NaCl0.png"
    )

    figure.savefig(
        plot_file,
        dpi=240,
    )

    plt.close(figure)

    print("=" * 78)
    print("refCBA second-virial validation")
    print("=" * 78)

    print(
        table[
            [
                "pH",
                "K2_kBT",
                "Z2",
                "full_B2_reduced",
                "core_B2_reduced",
                "effective_hs_diameter_nm",
                "B2star",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.8g}"
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
    print(f"  {plot_file}")

    if failed:
        raise RuntimeError(
            "Virial validation failed: "
            + ", ".join(failed)
        )

    print(
        "\nAll second-virial validation checks passed."
    )


if __name__ == "__main__":
    main()
