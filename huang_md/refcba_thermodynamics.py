"""Thermodynamic utilities for the Huang-anchored refCBA model.

The user-facing concentration unit is mg/mL. Internal density
units are particles/nm^3 and rho* = rho * sigma_HS^3.

For the requested concentration window, 0.1--20 mg/mL,
rho* remains below 0.2 and the validated low-density
hard-sphere RDF is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from huang_md.hard_sphere_rdf_low_density import (
    hard_sphere_rdf_reduced as low_density_rdf_reduced,
)
from huang_md.parameters import (
    HuangPotentialParameters,
)
from huang_md.perturbation import (
    PerturbationResult,
    calculate_perturbation_free_energy,
)
from huang_md.state_model import (
    RefCBAStateModel,
    parameters_for_state,
)


AVOGADRO_CONSTANT_PER_MOL = 6.02214076e23
LOW_DENSITY_RDF_LIMIT = 0.2


@dataclass(frozen=True)
class RefCBAConfiguration:
    """Baseline potential, state model and protein properties."""

    baseline: HuangPotentialParameters
    state_model: RefCBAStateModel
    molecular_weight_kDa: float


@dataclass(frozen=True)
class RefCBAFreeEnergyPoint:
    """One refCBA thermodynamic state."""

    pH: float
    added_nacl_mM: float
    concentration_mg_ml: float

    molecular_weight_kDa: float
    number_density_nm3: float
    hard_sphere_diameter_nm: float
    reduced_density_rho_sigma3: float
    packing_fraction: float

    K1_kBT: float
    Z1: float
    K2_kBT: float
    Z2: float

    beta_a1_per_particle: float
    beta_a2_per_particle: float
    beta_perturbation_per_particle: float
    beta_reference_free_energy_per_particle: float
    beta_total_free_energy_per_particle: float

    second_to_first_abs_ratio: float
    perturbation_status: str

    first_integral_error: float
    second_integral_error: float


def load_refcba_configuration(
    project_root: str | Path = ".",
) -> RefCBAConfiguration:
    """Load the three existing project YAML files."""
    root = Path(project_root)

    baseline_data = yaml.safe_load(
        (
            root
            / "configs"
            / "huang_baseline.yaml"
        ).read_text(encoding="utf-8")
    )["model"]

    # Descriptive YAML metadata; not a constructor field of
    # HuangPotentialParameters.
    baseline_data.pop("name", None)

    state_model_data = yaml.safe_load(
        (
            root
            / "configs"
            / "refcba_state_model.yaml"
        ).read_text(encoding="utf-8")
    )["state_model"]

    simulation_data = yaml.safe_load(
        (
            root
            / "configs"
            / "refcba_md.yaml"
        ).read_text(encoding="utf-8")
    )["simulation"]

    molecular_weight = float(
        simulation_data["molecular_weight_kDa"]
    )

    if (
        not np.isfinite(molecular_weight)
        or molecular_weight <= 0.0
    ):
        raise ValueError(
            "Molecular weight must be positive and finite."
        )

    return RefCBAConfiguration(
        baseline=HuangPotentialParameters(
            **baseline_data
        ),
        state_model=RefCBAStateModel(
            **state_model_data
        ),
        molecular_weight_kDa=molecular_weight,
    )


def concentration_to_number_density_nm3(
    concentration_mg_ml: float,
    molecular_weight_kDa: float,
) -> float:
    """Convert mg/mL to particles/nm^3.

    Numerically, 1 mg/mL equals 1 g/L.
    """
    concentration = float(
        concentration_mg_ml
    )
    molecular_weight = float(
        molecular_weight_kDa
    )

    if (
        not np.isfinite(concentration)
        or concentration <= 0.0
    ):
        raise ValueError(
            "Concentration must be positive and finite."
        )

    if (
        not np.isfinite(molecular_weight)
        or molecular_weight <= 0.0
    ):
        raise ValueError(
            "Molecular weight must be positive and finite."
        )

    return float(
        concentration
        * AVOGADRO_CONSTANT_PER_MOL
        / (
            molecular_weight
            * 1.0e27
        )
    )


def number_density_nm3_to_concentration(
    number_density_nm3: float,
    molecular_weight_kDa: float,
) -> float:
    """Convert particles/nm^3 to mg/mL."""
    number_density = float(
        number_density_nm3
    )
    molecular_weight = float(
        molecular_weight_kDa
    )

    if (
        not np.isfinite(number_density)
        or number_density <= 0.0
    ):
        raise ValueError(
            "Number density must be positive and finite."
        )

    if (
        not np.isfinite(molecular_weight)
        or molecular_weight <= 0.0
    ):
        raise ValueError(
            "Molecular weight must be positive and finite."
        )

    return float(
        number_density
        * molecular_weight
        * 1.0e27
        / AVOGADRO_CONSTANT_PER_MOL
    )


def concentration_to_reduced_density(
    concentration_mg_ml: float,
    molecular_weight_kDa: float,
    hard_sphere_diameter_nm: float,
) -> float:
    """Return rho* = rho * sigma_HS^3."""
    diameter = float(
        hard_sphere_diameter_nm
    )

    if (
        not np.isfinite(diameter)
        or diameter <= 0.0
    ):
        raise ValueError(
            "Hard-sphere diameter must be positive and finite."
        )

    number_density = (
        concentration_to_number_density_nm3(
            concentration_mg_ml,
            molecular_weight_kDa,
        )
    )

    return float(
        number_density
        * diameter**3
    )


def reduced_density_to_concentration(
    reduced_density: float,
    molecular_weight_kDa: float,
    hard_sphere_diameter_nm: float,
) -> float:
    """Convert rho* to mg/mL."""
    rho_star = float(reduced_density)
    diameter = float(
        hard_sphere_diameter_nm
    )

    if (
        not np.isfinite(rho_star)
        or rho_star <= 0.0
    ):
        raise ValueError(
            "Reduced density must be positive and finite."
        )

    if (
        not np.isfinite(diameter)
        or diameter <= 0.0
    ):
        raise ValueError(
            "Hard-sphere diameter must be positive and finite."
        )

    number_density = (
        rho_star
        / diameter**3
    )

    return number_density_nm3_to_concentration(
        number_density,
        molecular_weight_kDa,
    )


def classify_perturbation_ratio(
    second_to_first_abs_ratio: float,
) -> str:
    """Project-level numerical diagnostic.

    These thresholds are diagnostics for this reproduction
    project, not hard limits stated by Huang et al.
    """
    ratio = float(
        second_to_first_abs_ratio
    )

    if not np.isfinite(ratio):
        return "nonfinite"

    if ratio <= 0.5:
        return "relatively_reliable"

    if ratio <= 1.0:
        return "caution"

    return "uncontrolled"


def state_parameters(
    configuration: RefCBAConfiguration,
    pH: float,
    added_nacl_mM: float,
) -> HuangPotentialParameters:
    """Return the heuristic potential at one solution state."""
    return parameters_for_state(
        baseline=configuration.baseline,
        model=configuration.state_model,
        pH=float(pH),
        added_nacl_mM=float(
            added_nacl_mM
        ),
    )


def calculate_refcba_free_energy_point(
    configuration: RefCBAConfiguration,
    pH: float,
    added_nacl_mM: float,
    concentration_mg_ml: float,
) -> RefCBAFreeEnergyPoint:
    """Evaluate one point inside the requested concentration domain."""
    params = state_parameters(
        configuration,
        pH,
        added_nacl_mM,
    )

    # The Gaussian-core B2 validation already showed that
    # its equivalent hard-sphere diameter equals the model
    # diameter for the present baseline parametrization.
    hard_sphere_diameter_nm = float(
        params.diameter_nm
    )

    number_density = (
        concentration_to_number_density_nm3(
            concentration_mg_ml,
            configuration.molecular_weight_kDa,
        )
    )

    rho_star = (
        number_density
        * hard_sphere_diameter_nm**3
    )

    if rho_star >= LOW_DENSITY_RDF_LIMIT:
        raise ValueError(
            "Requested state lies outside the validated "
            "low-density RDF domain: "
            f"rho*sigma^3={rho_star:.8g}."
        )

    result: PerturbationResult = (
        calculate_perturbation_free_energy(
            params=params,
            number_density_nm3=number_density,
            hard_sphere_diameter_nm=(
                hard_sphere_diameter_nm
            ),
            thermal_wavelength_nm=1.0,
            rdf_function=(
                low_density_rdf_reduced
            ),
        )
    )

    status = classify_perturbation_ratio(
        result.second_to_first_abs_ratio
    )

    values = np.array(
        [
            number_density,
            rho_star,
            result.packing_fraction,
            result.beta_a1_per_particle,
            result.beta_a2_per_particle,
            result.beta_total_free_energy_per_particle,
            result.second_to_first_abs_ratio,
        ],
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise FloatingPointError(
            "Non-finite refCBA free-energy point generated."
        )

    return RefCBAFreeEnergyPoint(
        pH=float(pH),
        added_nacl_mM=float(
            added_nacl_mM
        ),
        concentration_mg_ml=float(
            concentration_mg_ml
        ),
        molecular_weight_kDa=(
            configuration.molecular_weight_kDa
        ),
        number_density_nm3=float(
            number_density
        ),
        hard_sphere_diameter_nm=(
            hard_sphere_diameter_nm
        ),
        reduced_density_rho_sigma3=float(
            result.reduced_density_rho_sigma3
        ),
        packing_fraction=float(
            result.packing_fraction
        ),
        K1_kBT=float(params.K1_kBT),
        Z1=float(params.Z1),
        K2_kBT=float(params.K2_kBT),
        Z2=float(params.Z2),
        beta_a1_per_particle=float(
            result.beta_a1_per_particle
        ),
        beta_a2_per_particle=float(
            result.beta_a2_per_particle
        ),
        beta_perturbation_per_particle=float(
            result.beta_perturbation_per_particle
        ),
        beta_reference_free_energy_per_particle=float(
            result.beta_reference_free_energy_per_particle
        ),
        beta_total_free_energy_per_particle=float(
            result.beta_total_free_energy_per_particle
        ),
        second_to_first_abs_ratio=float(
            result.second_to_first_abs_ratio
        ),
        perturbation_status=status,
        first_integral_error=float(
            result.first_integral_error
        ),
        second_integral_error=float(
            result.second_integral_error
        ),
    )
