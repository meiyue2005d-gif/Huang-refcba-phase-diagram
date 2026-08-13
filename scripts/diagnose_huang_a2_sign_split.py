#!/usr/bin/env python3

from dataclasses import fields, replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huang_md.parameters import HuangPotentialParameters
from huang_md.virial import total_potential_reduced
from huang_md.perturbation import perturbation_moments_reduced
from huang_md.lj_perturbation import hybrid_hard_sphere_rdf_reduced
from huang_md.hard_sphere import (
    packing_fraction,
    cs_reduced_isothermal_compressibility,
)
from huang_md.refcba_thermodynamics import (
    concentration_to_number_density_nm3,
)


BASELINE_PATH = ROOT / "configs" / "huang_baseline.yaml"

MAPPING_PATH = (
    ROOT
    / "results"
    / "liquid_theory_validation"
    / "huang_a1_K2_vs_pH.csv"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "liquid_theory_validation"
    / "huang_a2_sign_split_0p436mgml.csv"
)

PH_VALUES = [4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]

CONCENTRATION_MG_ML = 0.436
MOLECULAR_WEIGHT_KDA = 43.6


def load_baseline() -> HuangPotentialParameters:
    raw = yaml.safe_load(
        BASELINE_PATH.read_text(encoding="utf-8")
    )["model"]

    allowed = {
        field.name
        for field in fields(HuangPotentialParameters)
    }

    return HuangPotentialParameters(
        **{
            key: value
            for key, value in raw.items()
            if key in allowed
        }
    )


def main() -> None:
    baseline = load_baseline()
    mapping = pd.read_csv(MAPPING_PATH)

    sigma_hs_nm = float(baseline.diameter_nm)

    number_density = concentration_to_number_density_nm3(
        CONCENTRATION_MG_ML,
        MOLECULAR_WEIGHT_KDA,
    )

    rho_star = number_density * sigma_hs_nm**3

    eta = float(
        np.asarray(
            packing_fraction(
                number_density,
                sigma_hs_nm,
            )
        ).reshape(-1)[0]
    )

    chi_hs = float(
        np.asarray(
            cs_reduced_isothermal_compressibility(
                eta
            )
        ).reshape(-1)[0]
    )

    rows = []

    for target_pH in PH_VALUES:
        index = int(
            np.argmin(
                np.abs(
                    mapping["pH"].to_numpy(dtype=float)
                    - target_pH
                )
            )
        )

        state = mapping.iloc[index]

        params = replace(
            baseline,
            K2_kBT=float(state["K2_kBT"]),
            Z2=float(state["Z2"]),
        )

        def full_potential(distance):
            return np.asarray(
                total_potential_reduced(
                    distance,
                    params,
                ),
                dtype=float,
            )

        def attractive_potential(distance):
            return np.minimum(
                full_potential(distance),
                0.0,
            )

        def repulsive_potential(distance):
            return np.maximum(
                full_potential(distance),
                0.0,
            )

        (
            i1_full,
            i2_full,
            _,
            _,
        ) = perturbation_moments_reduced(
            potential_function=full_potential,
            packing_fraction_value=eta,
            rdf_function=hybrid_hard_sphere_rdf_reduced,
        )

        (
            i1_attractive,
            i2_attractive,
            _,
            _,
        ) = perturbation_moments_reduced(
            potential_function=attractive_potential,
            packing_fraction_value=eta,
            rdf_function=hybrid_hard_sphere_rdf_reduced,
        )

        (
            i1_repulsive,
            i2_repulsive,
            _,
            _,
        ) = perturbation_moments_reduced(
            potential_function=repulsive_potential,
            packing_fraction_value=eta,
            rdf_function=hybrid_hard_sphere_rdf_reduced,
        )

        beta_a1_full = 0.5 * rho_star * i1_full

        beta_a2_full = (
            -0.25
            * rho_star
            * chi_hs
            * i2_full
        )

        beta_a2_attractive = (
            -0.25
            * rho_star
            * chi_hs
            * i2_attractive
        )

        beta_a2_repulsive = (
            -0.25
            * rho_star
            * chi_hs
            * i2_repulsive
        )

        split_error = (
            i2_attractive
            + i2_repulsive
            - i2_full
        )

        relative_split_error = (
            abs(split_error)
            / max(abs(i2_full), 1.0e-15)
        )

        if relative_split_error > 1.0e-6:
            raise RuntimeError(
                f"Second-moment split failed at pH "
                f"{target_pH}: "
                f"relative error={relative_split_error}"
            )

        rows.append(
            {
                "pH": float(state["pH"]),
                "K2_kBT": float(state["K2_kBT"]),
                "rho_sigma3": rho_star,
                "packing_fraction": eta,
                "I1_full": i1_full,
                "I1_attractive": i1_attractive,
                "I1_repulsive": i1_repulsive,
                "I2_full": i2_full,
                "I2_attractive": i2_attractive,
                "I2_repulsive": i2_repulsive,
                "repulsive_fraction_of_I2": (
                    i2_repulsive / i2_full
                ),
                "attractive_fraction_of_I2": (
                    i2_attractive / i2_full
                ),
                "beta_a1_full": beta_a1_full,
                "beta_a2_full": beta_a2_full,
                "beta_a2_attractive_only": (
                    beta_a2_attractive
                ),
                "beta_a2_repulsive_only": (
                    beta_a2_repulsive
                ),
                "split_relative_error": (
                    relative_split_error
                ),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)

    print("=" * 100)
    print("HUANG A1 SECOND-ORDER SIGN-SPLIT AUDIT")
    print("=" * 100)

    print(
        result[
            [
                "pH",
                "K2_kBT",
                "I2_full",
                "I2_attractive",
                "I2_repulsive",
                "attractive_fraction_of_I2",
                "repulsive_fraction_of_I2",
                "beta_a1_full",
                "beta_a2_full",
                "beta_a2_attractive_only",
                "beta_a2_repulsive_only",
            ]
        ].to_string(index=False)
    )

    print()
    print("saved:", OUTPUT_PATH)
    print("HUANG_A2_SIGN_SPLIT: PASS")


if __name__ == "__main__":
    main()
