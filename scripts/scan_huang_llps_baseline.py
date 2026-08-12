#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.constants import Avogadro
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huang_md.parameters import HuangPotentialParameters
from huang_md.virial import calculate_virial_properties
from huang_md.perturbation import (
    calculate_perturbation_free_energy,
)
from huang_md.lj_perturbation import (
    hybrid_hard_sphere_rdf_reduced,
)
from huang_md.coexistence import (
    find_spinodal_densities,
    solve_fluid_coexistence,
)


BASELINE_FILE = ROOT / "configs" / "huang_baseline.yaml"

MAPPING_FILE = (
    ROOT
    / "results"
    / "liquid_theory_validation"
    / "huang_a1_K2_vs_pH.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "huang_a1_llps_baseline_coarse"
)

MOLECULAR_WEIGHT_KDA = 43.6
TARGET_CONCENTRATION_MG_ML = 0.436

# The interpolation table extends beyond the actual search
# interval because the five-point derivative stencil probes
# slightly outside each evaluated density.
TABLE_MINIMUM_DENSITY = 5.0e-6
SEARCH_MINIMUM_DENSITY = 1.0e-5

TABLE_MAXIMUM_DENSITY = 0.89
SEARCH_MAXIMUM_DENSITY = 0.885

RDF_BLEND_START = 0.20
RDF_BLEND_END = 0.25
RDF_BLEND_MARGIN = 0.005

PH_VALUES = np.array(
    [
        4.5,
        5.0,
        5.2,
        5.3,
        5.4,
        5.5,
        5.6,
        5.7,
        5.8,
        6.0,
        6.5,
        7.0,
        7.5,
    ],
    dtype=float,
)


def load_baseline() -> HuangPotentialParameters:
    raw = yaml.safe_load(
        BASELINE_FILE.read_text(encoding="utf-8")
    )["model"]

    allowed = {
        field.name
        for field in fields(HuangPotentialParameters)
    }

    kwargs = {
        key: value
        for key, value in raw.items()
        if key in allowed
    }

    return HuangPotentialParameters(**kwargs)


def build_density_grid() -> np.ndarray:
    sections = [
        np.geomspace(
            TABLE_MINIMUM_DENSITY,
            0.03,
            65,
            endpoint=False,
        ),
        np.linspace(
            0.03,
            0.19,
            75,
            endpoint=False,
        ),
        np.linspace(
            0.19,
            0.26,
            101,
            endpoint=False,
        ),
        np.linspace(
            0.26,
            TABLE_MAXIMUM_DENSITY,
            230,
        ),
    ]

    return np.unique(
        np.concatenate(sections)
    )


def reduced_density_to_concentration(
    reduced_density: float,
    hard_sphere_diameter_nm: float,
) -> float:
    number_density_nm3 = (
        float(reduced_density)
        / hard_sphere_diameter_nm**3
    )

    molecular_weight_g_mol = (
        MOLECULAR_WEIGHT_KDA * 1000.0
    )

    # 1 L = 1e24 nm^3; g/L is numerically mg/mL.
    return (
        number_density_nm3
        * 1.0e24
        / Avogadro
        * molecular_weight_g_mol
    )


def parameters_at_pH(
    baseline: HuangPotentialParameters,
    mapping: pd.DataFrame,
    pH: float,
) -> HuangPotentialParameters:
    index = int(
        np.argmin(
            np.abs(
                mapping["pH"].to_numpy(dtype=float)
                - pH
            )
        )
    )

    row = mapping.iloc[index]

    if abs(float(row["pH"]) - pH) > 1.0e-6:
        raise RuntimeError(
            f"No exact mapping row for pH={pH}."
        )

    return replace(
        baseline,
        K2_kBT=float(row["K2_kBT"]),
        Z2=float(row["Z2"]),
    )


def build_free_energy_table(
    params: HuangPotentialParameters,
    sigma_hs_nm: float,
    pH: float,
) -> pd.DataFrame:
    density_grid = build_density_grid()
    rows: list[dict[str, float]] = []

    print()
    print("=" * 78)
    print(f"Building Huang A1 free-energy table: pH={pH:g}")
    print("=" * 78)
    print(f"K2 = {params.K2_kBT:.9f}")
    print(f"Z2 = {params.Z2:.9f}")
    print(f"states = {density_grid.size}")

    for index, rho_star in enumerate(
        density_grid,
        start=1,
    ):
        number_density_nm3 = (
            rho_star / sigma_hs_nm**3
        )

        result = calculate_perturbation_free_energy(
            params=params,
            number_density_nm3=number_density_nm3,
            hard_sphere_diameter_nm=sigma_hs_nm,
            thermal_wavelength_nm=1.0,
            rdf_function=(
                hybrid_hard_sphere_rdf_reduced
            ),
        )

        rows.append(
            {
                "pH": pH,
                "reduced_density": rho_star,
                "concentration_mg_ml": (
                    reduced_density_to_concentration(
                        rho_star,
                        sigma_hs_nm,
                    )
                ),
                "beta_a1": (
                    result.beta_a1_per_particle
                ),
                "beta_a2": (
                    result.beta_a2_per_particle
                ),
                "beta_total": (
                    result.beta_total_free_energy_per_particle
                ),
                "second_to_first_ratio": (
                    result.second_to_first_abs_ratio
                ),
            }
        )

        if (
            index == 1
            or index % 50 == 0
            or index == density_grid.size
        ):
            print(
                f"  {index:4d}/{density_grid.size:4d}"
                f"  rho*={rho_star:.8f}"
                f"  beta a={result.beta_total_free_energy_per_particle:.8f}"
            )

    table = pd.DataFrame(rows)

    output_file = (
        OUTPUT_DIR
        / f"free_energy_pH{str(pH).replace('.', 'p')}.csv"
    )

    table.to_csv(output_file, index=False)

    return table


def scan_one_pH(
    baseline: HuangPotentialParameters,
    mapping: pd.DataFrame,
    sigma_hs_nm: float,
    pH: float,
) -> dict[str, object]:
    params = parameters_at_pH(
        baseline,
        mapping,
        pH,
    )

    table = build_free_energy_table(
        params,
        sigma_hs_nm,
        pH,
    )

    density = table[
        "reduced_density"
    ].to_numpy(dtype=float)

    free_energy_values = table[
        "beta_total"
    ].to_numpy(dtype=float)

    spline = CubicSpline(
        density,
        free_energy_values,
    )

    minimum_table_density = float(density[0])
    maximum_table_density = float(density[-1])

    def free_energy(rho_star: float) -> float:
        value = float(rho_star)

        if not (
            minimum_table_density
            <= value
            <= maximum_table_density
        ):
            raise ValueError(
                "Density outside interpolation table: "
                f"{value:.12g}"
            )

        return float(spline(value))

    try:
        roots = find_spinodal_densities(
            beta_free_energy_per_particle=(
                free_energy
            ),
            minimum_density=(
                SEARCH_MINIMUM_DENSITY
            ),
            maximum_density=(
                SEARCH_MAXIMUM_DENSITY
            ),
            grid_points=1800,
        )
    except Exception as exc:
        return {
            "pH": pH,
            "K2_kBT": params.K2_kBT,
            "Z2": params.Z2,
            "status": "no_spinodal",
            "error": repr(exc),
        }

    roots = [float(root) for root in roots]

    if len(roots) < 2:
        return {
            "pH": pH,
            "K2_kBT": params.K2_kBT,
            "Z2": params.Z2,
            "status": "fewer_than_two_spinodals",
            "all_spinodal_roots": repr(roots),
        }

    stitch_roots = roots[1:-1]

    stitch_roots_confined = all(
        RDF_BLEND_START - RDF_BLEND_MARGIN
        <= root
        <= RDF_BLEND_END + RDF_BLEND_MARGIN
        for root in stitch_roots
    )

    try:
        coexistence = solve_fluid_coexistence(
            beta_free_energy_per_particle=(
                free_energy
            ),
            minimum_density=(
                SEARCH_MINIMUM_DENSITY
            ),
            maximum_density=(
                SEARCH_MAXIMUM_DENSITY
            ),
            grid_points=1800,
        )
    except Exception as exc:
        return {
            "pH": pH,
            "K2_kBT": params.K2_kBT,
            "Z2": params.Z2,
            "status": "coexistence_failed",
            "all_spinodal_roots": repr(roots),
            "rdf_stitch_roots": repr(stitch_roots),
            "stitch_roots_confined": (
                stitch_roots_confined
            ),
            "error": repr(exc),
        }

    vapor_density = float(
        coexistence.vapor_density
    )

    liquid_density = float(
        coexistence.liquid_density
    )

    vapor_concentration = (
        reduced_density_to_concentration(
            vapor_density,
            sigma_hs_nm,
        )
    )

    liquid_concentration = (
        reduced_density_to_concentration(
            liquid_density,
            sigma_hs_nm,
        )
    )

    target_inside = (
        vapor_concentration
        <= TARGET_CONCENTRATION_MG_ML
        <= liquid_concentration
    )

    return {
        "pH": pH,
        "K2_kBT": params.K2_kBT,
        "Z2": params.Z2,
        "status": "coexistence_solved",
        "vapor_density_reduced": vapor_density,
        "liquid_density_reduced": liquid_density,
        "vapor_concentration_mg_ml": (
            vapor_concentration
        ),
        "liquid_concentration_mg_ml": (
            liquid_concentration
        ),
        "target_0p436_inside_two_phase": (
            target_inside
        ),
        "physical_vapor_spinodal": roots[0],
        "physical_liquid_spinodal": roots[-1],
        "all_spinodal_roots": repr(roots),
        "rdf_stitch_roots": repr(stitch_roots),
        "stitch_roots_confined": (
            stitch_roots_confined
        ),
        "number_of_spinodal_roots": len(roots),
        "optimizer_success": bool(
            coexistence.optimizer_success
        ),
        "pressure_residual": float(
            coexistence.pressure_residual
        ),
        "chemical_potential_residual": float(
            coexistence.chemical_potential_residual
        ),
        "maxwell_area_residual": float(
            coexistence.maxwell_area_residual
        ),
        "optimizer_message": str(
            coexistence.optimizer_message
        ),
    }


def estimate_target_crossing(
    results: pd.DataFrame,
) -> float | None:
    solved = results[
        results["status"] == "coexistence_solved"
    ].copy()

    if solved.empty:
        return None

    solved = solved.sort_values("pH")

    pH = solved["pH"].to_numpy(dtype=float)

    difference = (
        solved[
            "vapor_concentration_mg_ml"
        ].to_numpy(dtype=float)
        - TARGET_CONCENTRATION_MG_ML
    )

    for index in range(len(pH) - 1):
        left = difference[index]
        right = difference[index + 1]

        if left == 0.0:
            return float(pH[index])

        if left * right < 0.0:
            fraction = (
                -left / (right - left)
            )

            return float(
                pH[index]
                + fraction
                * (pH[index + 1] - pH[index])
            )

    return None


def generate_plot(results: pd.DataFrame) -> None:
    solved = results[
        results["status"] == "coexistence_solved"
    ].copy()

    if solved.empty:
        return

    solved = solved.sort_values("pH")

    figure, axis = plt.subplots(
        figsize=(7.2, 5.2)
    )

    axis.plot(
        solved["pH"],
        solved["vapor_concentration_mg_ml"],
        marker="o",
        label="Dilute branch",
    )

    axis.plot(
        solved["pH"],
        solved["liquid_concentration_mg_ml"],
        marker="o",
        label="Dense branch",
    )

    axis.axhline(
        TARGET_CONCENTRATION_MG_ML,
        linestyle="--",
        linewidth=1.2,
        label="0.436 mg/mL",
    )

    axis.set_yscale("log")
    axis.set_xlabel("pH")
    axis.set_ylabel("Protein concentration (mg/mL)")
    axis.set_title(
        "Huang A1 liquid–liquid coexistence baseline"
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIR
        / "huang_a1_llps_binodal_coarse.png",
        dpi=240,
    )

    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline = load_baseline()
    mapping = pd.read_csv(MAPPING_FILE)

    virial = calculate_virial_properties(
        baseline
    )

    sigma_hs_nm = float(
        virial.effective_hs_diameter_nm
    )

    print("=" * 78)
    print("Huang A1 baseline LLPS scan")
    print("=" * 78)
    print(f"sigma_HS = {sigma_hs_nm:.12g} nm")
    print(
        "effective/nominal = "
        f"{sigma_hs_nm / baseline.diameter_nm:.12g}"
    )
    print(
        "molecular weight = "
        f"{MOLECULAR_WEIGHT_KDA:.6g} kDa"
    )
    print(
        "target concentration = "
        f"{TARGET_CONCENTRATION_MG_ML:.6g} mg/mL"
    )
    print("pH values:", PH_VALUES.tolist())

    rows = []

    for pH in PH_VALUES:
        row = scan_one_pH(
            baseline=baseline,
            mapping=mapping,
            sigma_hs_nm=sigma_hs_nm,
            pH=float(pH),
        )

        rows.append(row)

        print()
        print(
            f"pH={pH:g} status={row['status']}"
        )

        if row["status"] == "coexistence_solved":
            print(
                "  dilute = "
                f"{row['vapor_concentration_mg_ml']:.9g}"
                " mg/mL"
            )
            print(
                "  dense  = "
                f"{row['liquid_concentration_mg_ml']:.9g}"
                " mg/mL"
            )
            print(
                "  0.436 inside = "
                f"{row['target_0p436_inside_two_phase']}"
            )
            print(
                "  pressure residual = "
                f"{row['pressure_residual']:.3e}"
            )
            print(
                "  chemical potential residual = "
                f"{row['chemical_potential_residual']:.3e}"
            )
        else:
            print(" ", row.get("error", ""))

    results = pd.DataFrame(rows)

    result_file = (
        OUTPUT_DIR
        / "huang_a1_llps_coarse_results.csv"
    )

    results.to_csv(result_file, index=False)

    generate_plot(results)

    crossing = estimate_target_crossing(results)

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(
        results[
            [
                column
                for column in [
                    "pH",
                    "K2_kBT",
                    "status",
                    "vapor_concentration_mg_ml",
                    "liquid_concentration_mg_ml",
                    "target_0p436_inside_two_phase",
                    "optimizer_success",
                    "pressure_residual",
                    "chemical_potential_residual",
                ]
                if column in results.columns
            ]
        ].to_string(index=False)
    )

    print()
    print("target crossing estimate:", crossing)

    if crossing is not None:
        if 5.35 <= crossing <= 5.75:
            print(
                "PASS crossing is consistent with "
                "the Huang pH 5.5–5.6 region."
            )
        else:
            print(
                "REVIEW crossing differs from the "
                "Huang pH 5.5–5.6 region."
            )

    print()
    print("Generated:")
    print(" ", result_file)
    print(
        " ",
        OUTPUT_DIR
        / "huang_a1_llps_binodal_coarse.png",
    )
    print()
    print("SCAN_HUANG_LLPS_BASELINE: COMPLETE")


if __name__ == "__main__":
    main()
