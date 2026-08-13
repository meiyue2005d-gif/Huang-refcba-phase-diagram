#!/usr/bin/env python3
"""Export a centered-slab orthorhombic input for HOOMD-blue."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STATE_CONFIG = ROOT / "configs" / "refcba_state_model.yaml"
MD_CONFIG = ROOT / "configs" / "refcba_md.yaml"

from huang_md.openmm_orthorhombic import create_orthorhombic_system, create_orthorhombic_topology
from huang_md.slab_initialization import (
    centered_slab_bounds_nm,
    generate_centered_slab_positions_nm,
    slab_fraction_from_concentrations,
)



# BEGIN ORTHORHOMBIC FACTORY COMPAT
# Load the already-working geometry factory from the original
# direct-coexistence runner. Importing the module does not run main().
import importlib.util as _importlib_util


def _load_orthorhombic_box_factory():
    runner_path = (
        ROOT
        / "scripts"
        / "run_direct_coexistence.py"
    )

    spec = _importlib_util.spec_from_file_location(
        "_refcba_openmm_direct_coexistence",
        runner_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load original slab runner: {runner_path}"
        )

    module = _importlib_util.module_from_spec(spec)

    # Register before execution for modules/classes that inspect
    # their own module name during import.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    factory = getattr(
        module,
        "orthorhombic_box_from_cubic_length",
        None,
    )

    if factory is None:
        available = sorted(
            name
            for name in vars(module)
            if (
                "orthorhombic" in name.lower()
                or "box" in name.lower()
            )
        )

        raise ImportError(
            "Original runner did not expose "
            "orthorhombic_box_from_cubic_length. "
            f"Related names: {available}"
        )

    return factory


orthorhombic_box_from_cubic_length = (
    _load_orthorhombic_box_factory()
)
# END ORTHORHOMBIC FACTORY COMPAT

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a centered dense slab in an orthorhombic "
            "box using the validated HOOMD pair table."
        )
    )

    parser.add_argument("--ph", type=float, required=True)
    parser.add_argument("--nacl-mM", type=float, required=True)

    parser.add_argument(
        "--global-concentration-mg-ml",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--initial-slab-concentration-mg-ml",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--z-aspect-ratio",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
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
    parser.add_argument("--state-config", type=Path, default=STATE_CONFIG)
    parser.add_argument("--md-config", type=Path, default=MD_CONFIG)

    return parser


def minimum_pair_distance_nm(
    positions_nm: np.ndarray,
    box_lengths_nm: np.ndarray,
) -> float:
    """Return the minimum periodic pair distance."""

    minimum = float("inf")

    for index in range(len(positions_nm) - 1):
        delta = positions_nm[index + 1 :] - positions_nm[index]

        delta -= box_lengths_nm * np.rint(
            delta / box_lengths_nm
        )

        distances = np.linalg.norm(delta, axis=1)

        if len(distances):
            minimum = min(
                minimum,
                float(distances.min()),
            )

    return minimum


def main() -> None:
    args = build_parser().parse_args()

    if args.global_concentration_mg_ml <= 0:
        raise ValueError(
            "Global concentration must be positive."
        )

    if (
        args.initial_slab_concentration_mg_ml
        <= args.global_concentration_mg_ml
    ):
        raise ValueError(
            "Initial slab concentration must exceed "
            "the global concentration."
        )

    if args.z_aspect_ratio <= 1.0:
        raise ValueError(
            "z-aspect-ratio must exceed 1."
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the validated cubic exporter to obtain the potential
    # table, reduced-unit conversion and state metadata.
    with tempfile.TemporaryDirectory(
        prefix="_cubic_reference_",
        dir=output_dir,
    ) as temporary_directory:
        reference_dir = Path(temporary_directory)

        command = [
            sys.executable,
            str(
                ROOT
                / "scripts"
                / "export_hoomd_state_input.py"
            ),
            "--ph",
            str(args.ph),
            "--nacl-mM",
            str(args.nacl_mM),
            "--concentration-mg-ml",
            str(args.global_concentration_mg_ml),
            "--seed",
            str(args.seed),
            "--table-width",
            str(args.table_width),
            "--output-dir",
            str(reference_dir),
            "--state-config",
            str(args.state_config),
            "--md-config",
            str(args.md_config),
        ]

        subprocess.run(command, check=True)

        with np.load(
            reference_dir / "hoomd_input.npz",
            allow_pickle=False,
        ) as reference_data:
            table_x = np.asarray(
                reference_data["table_x"],
                dtype=np.float64,
            ).copy()
            table_U = np.asarray(
                reference_data["table_U"],
                dtype=np.float64,
            ).copy()
            table_F = np.asarray(
                reference_data["table_F"],
                dtype=np.float64,
            ).copy()
            r_min = float(reference_data["r_min"])
            r_cut = float(reference_data["r_cut"])

        reference_metadata = json.loads(
            (
                reference_dir
                / "hoomd_input_metadata.json"
            ).read_text(encoding="utf-8")
        )

    cubic_length_nm = float(
        reference_metadata["box_length_nm"]
    )
    diameter_nm = float(
        reference_metadata["diameter_nm"]
    )
    n_particles = int(
        reference_metadata["n_particles"]
    )

    box = orthorhombic_box_from_cubic_length(
        cubic_length_nm=cubic_length_nm,
        z_aspect_ratio=args.z_aspect_ratio,
    )

    box_lengths_nm = np.asarray(
        box.lengths_nm,
        dtype=np.float64,
    )

    if box_lengths_nm.shape != (3,):
        raise ValueError(
            "Orthorhombic box lengths must have shape (3,)."
        )

    box_lengths_reduced = box_lengths_nm / diameter_nm

    slab_fraction_z = slab_fraction_from_concentrations(
        global_concentration_mg_ml=(
            args.global_concentration_mg_ml
        ),
        initial_slab_concentration_mg_ml=(
            args.initial_slab_concentration_mg_ml
        ),
    )

    slab_lower_nm, slab_upper_nm = centered_slab_bounds_nm(
        box=box,
        slab_fraction_z=slab_fraction_z,
    )

    slab_thickness_nm = (
        slab_upper_nm - slab_lower_nm
    )

    cutoff_nm = r_cut * diameter_nm

    dilute_gap_each_side_nm = (
        box_lengths_nm[2] - slab_thickness_nm
    ) / 2.0

    if slab_thickness_nm <= 2.0 * cutoff_nm:
        raise ValueError(
            "Initial slab is too thin: "
            f"slab={slab_thickness_nm:.6f} nm, "
            f"2*cutoff={2.0 * cutoff_nm:.6f} nm."
        )

    if dilute_gap_each_side_nm <= cutoff_nm:
        raise ValueError(
            "Each dilute-side gap must exceed the cutoff: "
            f"gap={dilute_gap_each_side_nm:.6f} nm, "
            f"cutoff={cutoff_nm:.6f} nm."
        )

    minimum_distance_requested_nm = (
        float(
            reference_metadata[
                "initial_min_separation_reduced"
            ]
        )
        * diameter_nm
    )

    positions_nm = generate_centered_slab_positions_nm(
        n_particles=n_particles,
        box=box,
        slab_fraction_z=slab_fraction_z,
        minimum_distance_nm=(
            minimum_distance_requested_nm
        ),
        seed=args.seed,
    )

    positions_nm = np.asarray(
        positions_nm,
        dtype=np.float64,
    )

    if positions_nm.shape != (n_particles, 3):
        raise ValueError(
            "Unexpected slab-position array shape: "
            f"{positions_nm.shape}"
        )

    # Slab generator produces coordinates in [0,L).
    # HOOMD snapshots use coordinates centered around zero.
    positions_reduced = (
        positions_nm - 0.5 * box_lengths_nm
    ) / diameter_nm

    minimum_distance_actual_nm = (
        minimum_pair_distance_nm(
            positions_nm,
            box_lengths_nm,
        )
    )

    np.savez_compressed(
        output_dir / "hoomd_input.npz",
        positions_reduced=positions_reduced,
        table_x=table_x,
        table_U=table_U,
        table_F=table_F,
        r_min=np.float64(r_min),
        r_cut=np.float64(r_cut),
        box_lengths_reduced=box_lengths_reduced,
    )

    metadata = dict(reference_metadata)

    metadata.update(
        {
            "engine_input": "HOOMD-blue",
            "initialization_type": "centered_slab",
            "pH": float(args.ph),
            "added_nacl_mM": float(args.nacl_mM),
            "added_NaCl_mM": float(args.nacl_mM),
            "global_concentration_mg_ml": float(
                args.global_concentration_mg_ml
            ),
            "concentration_mg_ml": float(
                args.global_concentration_mg_ml
            ),
            "initial_slab_concentration_mg_ml": float(
                args.initial_slab_concentration_mg_ml
            ),
            "initial_slab_fraction_z": float(
                slab_fraction_z
            ),
            "initial_slab_bounds_nm": [
                float(slab_lower_nm),
                float(slab_upper_nm),
            ],
            "initial_slab_thickness_nm": float(
                slab_thickness_nm
            ),
            "dilute_gap_each_side_nm": float(
                dilute_gap_each_side_nm
            ),
            "cubic_equivalent_box_length_nm": float(
                cubic_length_nm
            ),
            "box_lengths_nm": box_lengths_nm.tolist(),
            "box_lengths_reduced": (
                box_lengths_reduced.tolist()
            ),
            "box_aspect_ratio_z": float(
                args.z_aspect_ratio
            ),
            "box_volume_nm3": float(
                np.prod(box_lengths_nm)
            ),
            "cutoff_nm": float(cutoff_nm),
            "initial_minimum_distance_nm": float(
                minimum_distance_actual_nm
            ),
            "trajectory_box_key": "box_lengths_nm",
        }
    )

    # Remove cubic-only fields to prevent accidental misuse.
    metadata.pop("box_length_nm", None)
    metadata.pop("box_length_reduced", None)

    with (
        output_dir / "hoomd_input_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 80)
    print("HOOMD centered-slab input exported")
    print("=" * 80)
    print("Output directory       :", output_dir)
    print("pH                     :", args.ph)
    print("NaCl                   :", args.nacl_mM, "mM")
    print(
        "Global concentration   :",
        args.global_concentration_mg_ml,
        "mg/mL",
    )
    print(
        "Initial slab conc.     :",
        args.initial_slab_concentration_mg_ml,
        "mg/mL",
    )
    print("Particles              :", n_particles)
    print("Box lengths (nm)       :", box_lengths_nm)
    print(
        "Box lengths reduced    :",
        box_lengths_reduced,
    )
    print("Slab fraction z        :", slab_fraction_z)
    print(
        "Slab thickness (nm)    :",
        slab_thickness_nm,
    )
    print(
        "Dilute gap/side (nm)   :",
        dilute_gap_each_side_nm,
    )
    print(
        "Pair cutoff (nm)       :",
        cutoff_nm,
    )
    print(
        "Requested minimum r    :",
        minimum_distance_requested_nm,
    )
    print(
        "Actual minimum r       :",
        minimum_distance_actual_nm,
    )
    print("EXPORT_HOOMD_SLAB_INPUT: PASS")


if __name__ == "__main__":
    main()
