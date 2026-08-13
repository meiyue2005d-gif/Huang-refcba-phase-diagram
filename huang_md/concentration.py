"""Concentration and periodic-box conversion utilities."""

from __future__ import annotations


AVOGADRO_CONSTANT = 6.02214076e23


def concentration_to_box_length_nm(
    concentration_mg_ml: float,
    molecular_weight_kDa: float,
    n_particles: int,
) -> float:
    """Convert protein mass concentration into cubic-box length.

    Notes
    -----
    Numerically, mg/mL is equal to g/L.

    molecular_weight_kDa is converted to g/mol by multiplying by 1000.
    One liter equals 1e24 nm^3.
    """
    if concentration_mg_ml <= 0:
        raise ValueError(
            "concentration_mg_ml must be positive."
        )

    if molecular_weight_kDa <= 0:
        raise ValueError(
            "molecular_weight_kDa must be positive."
        )

    if n_particles <= 0:
        raise ValueError(
            "n_particles must be positive."
        )

    molecular_weight_g_mol = molecular_weight_kDa * 1000.0

    volume_liter = (
        n_particles
        * molecular_weight_g_mol
        / (
            AVOGADRO_CONSTANT
            * concentration_mg_ml
        )
    )

    volume_nm3 = volume_liter * 1.0e24

    return volume_nm3 ** (1.0 / 3.0)


def box_length_to_concentration_mg_ml(
    box_length_nm: float,
    molecular_weight_kDa: float,
    n_particles: int,
) -> float:
    """Inverse conversion from cubic-box length to concentration."""
    if box_length_nm <= 0:
        raise ValueError("box_length_nm must be positive.")

    if molecular_weight_kDa <= 0:
        raise ValueError(
            "molecular_weight_kDa must be positive."
        )

    if n_particles <= 0:
        raise ValueError("n_particles must be positive.")

    volume_nm3 = box_length_nm**3
    volume_liter = volume_nm3 / 1.0e24

    molecular_weight_g_mol = molecular_weight_kDa * 1000.0

    return (
        n_particles
        * molecular_weight_g_mol
        / (
            AVOGADRO_CONSTANT
            * volume_liter
        )
    )
