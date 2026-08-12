"""Centered-slab initialization for orthorhombic periodic boxes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OrthorhombicBox:
    length_x_nm: float
    length_y_nm: float
    length_z_nm: float

    def __post_init__(self) -> None:
        values = np.asarray(self.lengths_nm, dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("All box lengths must be positive and finite.")

    @property
    def lengths_nm(self) -> tuple[float, float, float]:
        return (
            float(self.length_x_nm),
            float(self.length_y_nm),
            float(self.length_z_nm),
        )

    @property
    def volume_nm3(self) -> float:
        return float(
            self.length_x_nm
            * self.length_y_nm
            * self.length_z_nm
        )


def orthorhombic_box_from_cubic_length(
    cubic_length_nm: float,
    z_aspect_ratio: float = 3.0,
) -> OrthorhombicBox:
    """Preserve volume with Lx=Ly and Lz/Lx=z_aspect_ratio."""
    length = float(cubic_length_nm)
    aspect = float(z_aspect_ratio)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("cubic_length_nm must be positive and finite.")
    if not np.isfinite(aspect) or aspect < 1.0:
        raise ValueError("z_aspect_ratio must be finite and at least 1.")
    transverse = (length**3 / aspect) ** (1.0 / 3.0)
    return OrthorhombicBox(
        transverse,
        transverse,
        aspect * transverse,
    )


def slab_fraction_from_concentrations(
    global_concentration_mg_ml: float,
    initial_slab_concentration_mg_ml: float,
) -> float:
    """Fraction of z occupied when all particles start in the slab."""
    global_c = float(global_concentration_mg_ml)
    slab_c = float(initial_slab_concentration_mg_ml)
    if not np.isfinite(global_c) or global_c <= 0.0:
        raise ValueError("Global concentration must be positive and finite.")
    if not np.isfinite(slab_c) or slab_c <= global_c:
        raise ValueError(
            "Initial slab concentration must be finite and exceed "
            "the global concentration."
        )
    return float(global_c / slab_c)


def centered_slab_bounds_nm(
    box: OrthorhombicBox,
    slab_fraction_z: float,
) -> tuple[float, float]:
    fraction = float(slab_fraction_z)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("slab_fraction_z must lie between 0 and 1.")
    thickness = fraction * box.length_z_nm
    lower = 0.5 * (box.length_z_nm - thickness)
    return float(lower), float(lower + thickness)


def minimum_pair_distance_nm(
    positions_nm: FloatArray,
    box: OrthorhombicBox,
) -> float:
    """Minimum-image pair distance."""
    positions = np.asarray(positions_nm, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions_nm must have shape (n_particles, 3).")
    if len(positions) < 2:
        return float("inf")
    lengths = np.asarray(box.lengths_nm)
    minimum_squared = float("inf")
    for index in range(len(positions) - 1):
        displacement = positions[index + 1 :] - positions[index]
        displacement -= lengths * np.round(displacement / lengths)
        minimum_squared = min(
            minimum_squared,
            float(np.min(np.sum(displacement**2, axis=1))),
        )
    return float(np.sqrt(minimum_squared))


def generate_centered_slab_positions_nm(
    n_particles: int,
    box: OrthorhombicBox,
    slab_fraction_z: float,
    minimum_distance_nm: float,
    seed: int,
    maximum_attempts_per_particle: int = 200000,
) -> FloatArray:
    """Random nonoverlapping positions in a centered z slab."""
    if n_particles <= 0:
        raise ValueError("n_particles must be positive.")
    minimum_distance = float(minimum_distance_nm)
    if not np.isfinite(minimum_distance) or minimum_distance <= 0.0:
        raise ValueError("minimum_distance_nm must be positive and finite.")
    if maximum_attempts_per_particle <= 0:
        raise ValueError("maximum_attempts_per_particle must be positive.")

    lower_z, upper_z = centered_slab_bounds_nm(box, slab_fraction_z)
    rng = np.random.default_rng(int(seed))
    positions = np.empty((n_particles, 3), dtype=np.float64)
    lengths = np.asarray(box.lengths_nm)
    minimum_squared = minimum_distance**2

    for particle_index in range(n_particles):
        accepted = False
        for _ in range(maximum_attempts_per_particle):
            candidate = np.asarray(
                [
                    rng.uniform(0.0, box.length_x_nm),
                    rng.uniform(0.0, box.length_y_nm),
                    rng.uniform(lower_z, upper_z),
                ]
            )
            if particle_index == 0:
                accepted = True
            else:
                displacement = candidate - positions[:particle_index]
                displacement -= lengths * np.round(displacement / lengths)
                accepted = bool(
                    np.all(
                        np.sum(displacement**2, axis=1)
                        >= minimum_squared
                    )
                )
            if accepted:
                positions[particle_index] = candidate
                break
        if not accepted:
            raise RuntimeError(
                f"Failed to place particle {particle_index + 1}/"
                f"{n_particles}; lower slab concentration or separation."
            )
    return positions
