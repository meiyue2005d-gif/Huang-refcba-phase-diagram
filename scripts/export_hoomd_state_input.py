#!/usr/bin/env python3
"""Export one refCBA state for a HOOMD-blue simulation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huang_md.concentration import (  # noqa: E402
    box_length_to_concentration_mg_ml,
    concentration_to_box_length_nm,
)
from huang_md.openmm_model import (  # noqa: E402
    calculate_cutoff_shift_reduced,
    generate_random_positions_nm,
)
from huang_md.parameters import HuangPotentialParameters  # noqa: E402
from huang_md.potential import (  # noqa: E402
    force_reduced,
    total_potential_reduced,
)
from huang_md.state_model import (  # noqa: E402
    RefCBAStateModel,
    parameters_for_state,
)


BASELINE_CONFIG = ROOT / "configs" / "huang_baseline.yaml"
STATE_CONFIG = ROOT / "configs" / "refcba_state_model.yaml"
MD_CONFIG = ROOT / "configs" / "refcba_md.yaml"

DALTON_KG = 1.66053906660e-27
BOLTZMANN_J_K = 1.380649e-23


def load_md_config(path: Path = MD_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    return raw["simulation"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--ph", type=float, required=True)
    parser.add_argument("--nacl-mM", type=float, required=True)
    parser.add_argument(
        "--concentration-mg-ml",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260721,
    )
    parser.add_argument(
        "--table-width",
        type=int,
        default=65536,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--state-config",
        type=Path,
        default=STATE_CONFIG,
        help="Versioned electrostatic state-model YAML.",
    )
    parser.add_argument(
        "--md-config",
        type=Path,
        default=MD_CONFIG,
        help="Protein/engine YAML; use huang_a1_md.yaml for strict A1.",
    )

    return parser


def calculate_time_unit_ps(
    diameter_nm: float,
    molecular_weight_kDa: float,
    temperature_K: float,
) -> float:
    """tau = d * sqrt(m / kBT), returned in ps."""
    diameter_m = diameter_nm * 1.0e-9
    mass_kg = (
        molecular_weight_kDa
        * 1000.0
        * DALTON_KG
    )
    kbt_joule = BOLTZMANN_J_K * temperature_K

    tau_seconds = diameter_m * np.sqrt(
        mass_kg / kbt_joule
    )

    return float(tau_seconds / 1.0e-12)


def main() -> None:
    args = build_parser().parse_args()

    if not 3.0 <= args.ph <= 9.0:
        raise ValueError("pH must be between 3 and 9.")

    if not 0.0 <= args.nacl_mM <= 500.0:
        raise ValueError("NaCl must be between 0 and 500 mM.")

    if args.concentration_mg_ml <= 0:
        raise ValueError("Concentration must be positive.")

    if args.table_width < 4096:
        raise ValueError("table-width must be at least 4096.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = HuangPotentialParameters.from_yaml(
        BASELINE_CONFIG
    )
    state_model = RefCBAStateModel.from_yaml(
        args.state_config
    )
    md_config = load_md_config(args.md_config)

    state_params = parameters_for_state(
        baseline=baseline,
        model=state_model,
        pH=args.ph,
        added_nacl_mM=args.nacl_mM,
    )
    state_params.validate()

    n_particles = int(md_config["n_particles"])
    molecular_weight_kDa = float(
        md_config["molecular_weight_kDa"]
    )
    temperature_K = float(
        md_config["temperature_K"]
    )
    timestep_fs = float(
        md_config["timestep_fs"]
    )
    collision_frequency_per_ps = float(
        md_config[
            "andersen_collision_frequency_per_ps"
        ]
    )
    initial_min_separation_reduced = float(
        md_config[
            "initial_min_separation_reduced"
        ]
    )

    box_length_nm = concentration_to_box_length_nm(
        concentration_mg_ml=args.concentration_mg_ml,
        molecular_weight_kDa=molecular_weight_kDa,
        n_particles=n_particles,
    )

    recovered_concentration = (
        box_length_to_concentration_mg_ml(
            box_length_nm=box_length_nm,
            molecular_weight_kDa=molecular_weight_kDa,
            n_particles=n_particles,
        )
    )

    diameter_nm = state_params.diameter_nm
    box_length_reduced = box_length_nm / diameter_nm

    if (
        state_params.cutoff_reduced
        >= box_length_reduced / 2.0
    ):
        raise ValueError(
            "Reduced cutoff must be below half the box length."
        )

    positions_nm = generate_random_positions_nm(
        n_particles=n_particles,
        box_length_nm=box_length_nm,
        minimum_distance_nm=(
            initial_min_separation_reduced
            * diameter_nm
        ),
        seed=args.seed,
    )

    # OpenMM positions are generated in [0, L).
    # HOOMD uses a box centered at the origin.
    positions_reduced = (
        positions_nm / diameter_nm
        - box_length_reduced / 2.0
    )

    r_min = 1.0e-4
    r_cut = state_params.cutoff_reduced

    x_grid = np.linspace(
        r_min,
        r_cut,
        args.table_width,
        dtype=np.float64,
    )

    cutoff_shift = calculate_cutoff_shift_reduced(
        state_params
    )

    table_U = (
        total_potential_reduced(
            x_grid,
            state_params,
        )
        - cutoff_shift
    )

    table_F = force_reduced(
        x_grid,
        state_params,
    )

    time_unit_ps = calculate_time_unit_ps(
        diameter_nm=diameter_nm,
        molecular_weight_kDa=molecular_weight_kDa,
        temperature_K=temperature_K,
    )

    timestep_reduced = (
        timestep_fs / 1000.0 / time_unit_ps
    )

    # In reduced units m*=1:
    # gamma* = (m*nu)*tau/m = nu*tau
    langevin_gamma_reduced = (
        collision_frequency_per_ps
        * time_unit_ps
    )

    hoomd_seed = int(args.seed) & 0xFFFF

    input_file = output_dir / "hoomd_input.npz"

    np.savez_compressed(
        input_file,
        positions_reduced=np.asarray(
            positions_reduced,
            dtype=np.float64,
        ),
        table_x=x_grid,
        table_U=table_U,
        table_F=table_F,
        r_min=np.float64(r_min),
        r_cut=np.float64(r_cut),
        box_length_reduced=np.float64(
            box_length_reduced
        ),
    )

    metadata = {
        "engine_input": "HOOMD-blue",
        "state_model_id": state_model.model_id,
        "charge_mapping": state_model.charge_mapping,
        "protein_id": state_model.protein_id,
        "protein_sequence_length_aa": len(state_model.protein_sequence),
        "state_config": str(args.state_config.resolve()),
        "md_config": str(args.md_config.resolve()),
        **state_model.applicability_flags(args.ph, args.nacl_mM),
        "pH": float(args.ph),
        "added_nacl_mM": float(args.nacl_mM),
        "concentration_mg_ml": float(
            args.concentration_mg_ml
        ),
        "recovered_concentration_mg_ml": float(
            recovered_concentration
        ),
        "n_particles": n_particles,
        "molecular_weight_kDa": molecular_weight_kDa,
        "temperature_K": temperature_K,
        "diameter_nm": diameter_nm,
        "box_length_nm": box_length_nm,
        "box_length_reduced": box_length_reduced,
        "timestep_fs": timestep_fs,
        "timestep_reduced": timestep_reduced,
        "time_unit_ps": time_unit_ps,
        "collision_frequency_per_ps": (
            collision_frequency_per_ps
        ),
        "langevin_gamma_reduced": (
            langevin_gamma_reduced
        ),
        "initial_min_separation_reduced": (
            initial_min_separation_reduced
        ),
        "seed": int(args.seed),
        "hoomd_seed": hoomd_seed,
        "K1_kBT": state_params.K1_kBT,
        "Z1": state_params.Z1,
        "K2_kBT": state_params.K2_kBT,
        "Z2": state_params.Z2,
        "gaussian_sigma_reduced": (
            state_params.gaussian_sigma_reduced
        ),
        "gaussian_epsilon_kBT": (
            state_params.gaussian_epsilon_kBT
        ),
        "cutoff_reduced": r_cut,
        "cutoff_shift_kBT": cutoff_shift,
        "table_width": int(args.table_width),
    }

    metadata_file = output_dir / "hoomd_input_metadata.json"

    with metadata_file.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 72)
    print("HOOMD state input exported")
    print("=" * 72)
    print("Output directory       :", output_dir)
    print("Input NPZ              :", input_file)
    print("Metadata               :", metadata_file)
    print("pH                     :", args.ph)
    print("NaCl                   :", args.nacl_mM, "mM")
    print(
        "Concentration          :",
        args.concentration_mg_ml,
        "mg/mL",
    )
    print("Particles              :", n_particles)
    print("Box length             :", box_length_nm, "nm")
    print("Reduced box length     :", box_length_reduced)
    print("K2                     :", state_params.K2_kBT)
    print("Z2                     :", state_params.Z2)
    print("Time unit              :", time_unit_ps, "ps")
    print("Reduced timestep       :", timestep_reduced)
    print(
        "Reduced Langevin gamma:",
        langevin_gamma_reduced,
    )
    print("HOOMD seed             :", hoomd_seed)
    print("EXPORT_HOOMD_STATE_INPUT: PASS")


if __name__ == "__main__":
    main()
