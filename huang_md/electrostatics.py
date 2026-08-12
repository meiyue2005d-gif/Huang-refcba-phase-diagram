"""pH-dependent charge and salt screening for the refCBA colloid."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


REFCBA_SEQUENCE = (
    "DYYGRFNDYDRYYGRSMF"
    "NYGWMMDGDRYNRYNRWMDYPERY"
    "MDMSGYQMDMSGRWMDMQGR"
)


@dataclass(frozen=True)
class PKaSet:
    """Initial intrinsic pKa values for Henderson-Hasselbalch charge."""

    n_terminus: float = 8.0
    c_terminus: float = 3.1

    aspartate: float = 3.9
    glutamate: float = 4.1
    histidine: float = 6.0
    cysteine: float = 8.3
    tyrosine: float = 10.1
    lysine: float = 10.5
    arginine: float = 12.5


DEFAULT_PKA = PKaSet()


def validate_sequence(sequence: str) -> str:
    """Remove whitespace, convert to upper case and validate residues."""
    clean = "".join(sequence.split()).upper()

    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    invalid = sorted(set(clean) - allowed)

    if not clean:
        raise ValueError("Protein sequence is empty.")

    if invalid:
        raise ValueError(
            "Sequence contains unsupported residues: "
            + ", ".join(invalid)
        )

    return clean


def amino_acid_counts(sequence: str = REFCBA_SEQUENCE) -> dict[str, int]:
    clean = validate_sequence(sequence)

    return {
        residue: clean.count(residue)
        for residue in sorted(set(clean))
    }


def positive_group_charge(
    pH: ArrayLike,
    pKa: float,
) -> FloatArray:
    """Fractional charge of a basic group."""
    values = np.asarray(pH, dtype=np.float64)
    return 1.0 / (1.0 + 10.0 ** (values - pKa))


def negative_group_charge(
    pH: ArrayLike,
    pKa: float,
) -> FloatArray:
    """Magnitude of the negative charge of an acidic group."""
    values = np.asarray(pH, dtype=np.float64)
    return 1.0 / (1.0 + 10.0 ** (pKa - values))


def net_charge(
    pH: ArrayLike,
    sequence: str = REFCBA_SEQUENCE,
    pka: PKaSet = DEFAULT_PKA,
    include_termini: bool = True,
) -> FloatArray:
    """Estimate sequence net charge using Henderson-Hasselbalch terms."""
    clean = validate_sequence(sequence)
    values = np.asarray(pH, dtype=np.float64)

    charge = np.zeros_like(values)

    if include_termini:
        charge += positive_group_charge(values, pka.n_terminus)
        charge -= negative_group_charge(values, pka.c_terminus)

    charge += clean.count("H") * positive_group_charge(
        values,
        pka.histidine,
    )
    charge += clean.count("K") * positive_group_charge(
        values,
        pka.lysine,
    )
    charge += clean.count("R") * positive_group_charge(
        values,
        pka.arginine,
    )

    charge -= clean.count("D") * negative_group_charge(
        values,
        pka.aspartate,
    )
    charge -= clean.count("E") * negative_group_charge(
        values,
        pka.glutamate,
    )
    charge -= clean.count("C") * negative_group_charge(
        values,
        pka.cysteine,
    )
    charge -= clean.count("Y") * negative_group_charge(
        values,
        pka.tyrosine,
    )

    return charge


def total_ionic_strength_mM(
    added_nacl_mM: ArrayLike,
    background_mM: float = 20.0,
) -> FloatArray:
    """Total monovalent ionic strength used by the implicit-salt model."""
    added = np.asarray(added_nacl_mM, dtype=np.float64)

    if background_mM <= 0:
        raise ValueError("background_mM must be positive.")

    if np.any(added < 0):
        raise ValueError("NaCl concentration cannot be negative.")

    return added + background_mM


def debye_length_nm(
    ionic_strength_mM: ArrayLike,
    temperature_K: float = 300.0,
) -> FloatArray:
    """Approximate Debye length in water for monovalent electrolyte."""
    ionic_strength = np.asarray(
        ionic_strength_mM,
        dtype=np.float64,
    )

    if np.any(ionic_strength <= 0):
        raise ValueError("Ionic strength must be positive.")

    ionic_strength_M = ionic_strength / 1000.0

    temperature_factor = sqrt(temperature_K / 298.15)

    return (
        0.304
        * temperature_factor
        / np.sqrt(ionic_strength_M)
    )


def yukawa_screening_parameter(
    ionic_strength_mM: ArrayLike,
    particle_diameter_nm: float,
    temperature_K: float = 300.0,
) -> FloatArray:
    """Return reduced Yukawa screening parameter Z2 = kappa*d."""
    if particle_diameter_nm <= 0:
        raise ValueError("particle_diameter_nm must be positive.")

    lambda_debye = debye_length_nm(
        ionic_strength_mM,
        temperature_K=temperature_K,
    )

    return particle_diameter_nm / lambda_debye


def estimate_isoelectric_point(
    sequence: str = REFCBA_SEQUENCE,
    pH_min: float = 0.0,
    pH_max: float = 14.0,
) -> float:
    """Estimate the pH at which the sequence charge crosses zero."""
    grid = np.linspace(pH_min, pH_max, 140001)
    charge = net_charge(grid, sequence=sequence)

    index = int(np.argmin(np.abs(charge)))
    return float(grid[index])
