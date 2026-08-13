"""Tests for the second-order LJ perturbation module."""

from __future__ import annotations

import sys
from math import pi
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from huang_md.lj_perturbation import (
    beta_lj_free_energy_per_particle,
    calculate_lj_perturbation_free_energy,
    hybrid_hard_sphere_rdf_reduced,
    lj_beta_potential_reduced,
    make_lj_free_energy_function,
)


def assert_close(
    actual: float,
    expected: float,
    tolerance: float,
    name: str,
) -> None:
    error = abs(actual - expected)

    if error > tolerance:
        raise AssertionError(
            f"{name}: expected={expected:.12g}, "
            f"actual={actual:.12g}, "
            f"error={error:.3e}"
        )

    print(
        f"PASS  {name:<46s} "
        f"value={actual:.10g}"
    )


def main() -> None:
    minimum_position = 2.0 ** (1.0 / 6.0)

    distances = np.array(
        [
            0.5,
            1.0,
            minimum_position,
            4.999,
            5.0,
            6.0,
        ],
        dtype=np.float64,
    )

    potential = lj_beta_potential_reduced(
        distances,
        temperature_reduced=1.0,
    )

    assert_close(
        potential[0],
        0.0,
        0.0,
        "hard-sphere core potential suppressed",
    )

    assert_close(
        potential[1],
        0.0,
        1.0e-14,
        "LJ zero crossing at x=1",
    )

    assert_close(
        potential[2],
        -1.0,
        1.0e-13,
        "LJ minimum at x=2^(1/6)",
    )

    assert_close(
        potential[4],
        0.0,
        0.0,
        "potential cut exactly at x=5",
    )

    assert_close(
        potential[5],
        0.0,
        0.0,
        "potential zero beyond cutoff",
    )

    eta_low = pi * 0.1 / 6.0
    eta_high = pi * 0.5 / 6.0

    rdf_low = hybrid_hard_sphere_rdf_reduced(
        np.array([0.5, 1.0, 2.0, 8.0]),
        eta_low,
    )

    rdf_high = hybrid_hard_sphere_rdf_reduced(
        np.array([0.5, 1.0, 2.0, 8.0]),
        eta_high,
    )

    if rdf_low[0] != 0.0 or rdf_high[0] != 0.0:
        raise AssertionError(
            "Hybrid RDF violates hard-core exclusion."
        )

    print("PASS  hybrid RDF hard-core exclusion")

    if not (
        np.isfinite(rdf_low).all()
        and np.isfinite(rdf_high).all()
    ):
        raise AssertionError(
            "Hybrid RDF generated non-finite values."
        )

    print("PASS  hybrid RDF finite in both density regimes")

    # Confirm continuity across both numerical blend boundaries.
    probe_distances = np.linspace(
        1.0,
        5.0,
        101,
    )

    for boundary in [0.2, 0.25]:
        delta = 1.0e-7

        below = hybrid_hard_sphere_rdf_reduced(
            probe_distances,
            pi * (boundary - delta) / 6.0,
        )

        above = hybrid_hard_sphere_rdf_reduced(
            probe_distances,
            pi * (boundary + delta) / 6.0,
        )

        maximum_jump = float(
            np.max(
                np.abs(
                    np.asarray(above)
                    - np.asarray(below)
                )
            )
        )

        if maximum_jump > 1.0e-4:
            raise AssertionError(
                f"RDF blend discontinuity at rho*={boundary}: "
                f"{maximum_jump:.6e}"
            )

        print(
            "PASS  hybrid RDF continuity "
            f"rho*={boundary:.2f} "
            f"max_jump={maximum_jump:.3e}"
        )

    result_t1 = (
        calculate_lj_perturbation_free_energy(
            temperature_reduced=1.0,
            reduced_density=0.5,
        )
    )

    result_t2 = (
        calculate_lj_perturbation_free_energy(
            temperature_reduced=2.0,
            reduced_density=0.5,
        )
    )

    assert_close(
        result_t2.first_moment_integral,
        0.5 * result_t1.first_moment_integral,
        1.0e-8,
        "first moment scales as inverse temperature",
    )

    assert_close(
        result_t2.second_moment_integral,
        0.25 * result_t1.second_moment_integral,
        1.0e-8,
        "second moment scales as inverse T squared",
    )

    assert_close(
        result_t2.beta_a1_per_particle,
        0.5 * result_t1.beta_a1_per_particle,
        1.0e-8,
        "first-order free energy temperature scaling",
    )

    assert_close(
        result_t2.beta_a2_per_particle,
        0.25 * result_t1.beta_a2_per_particle,
        1.0e-8,
        "second-order free energy temperature scaling",
    )

    if result_t1.second_moment_integral <= 0.0:
        raise AssertionError(
            "Squared-potential moment must be positive."
        )

    print("PASS  squared-potential moment is positive")

    if result_t1.beta_a2_per_particle >= 0.0:
        raise AssertionError(
            "Second-order perturbation term must be negative."
        )

    print("PASS  second-order free-energy term is negative")

    reconstructed_reference = (
        result_t1.beta_ideal_free_energy_per_particle
        + result_t1.beta_hard_sphere_excess_per_particle
    )

    assert_close(
        result_t1.beta_reference_free_energy_per_particle,
        reconstructed_reference,
        1.0e-13,
        "reference free-energy decomposition",
    )

    reconstructed_total = (
        result_t1.beta_reference_free_energy_per_particle
        + result_t1.beta_a1_per_particle
        + result_t1.beta_a2_per_particle
    )

    assert_close(
        result_t1.beta_total_free_energy_per_particle,
        reconstructed_total,
        1.0e-13,
        "total free-energy decomposition",
    )

    free_energy_function = (
        make_lj_free_energy_function(
            temperature_reduced=1.0
        )
    )

    assert_close(
        free_energy_function(0.5),
        result_t1.beta_total_free_energy_per_particle,
        1.0e-13,
        "coexistence free-energy wrapper",
    )

    assert_close(
        beta_lj_free_energy_per_particle(
            reduced_density=0.5,
            temperature_reduced=1.0,
        ),
        result_t1.beta_total_free_energy_per_particle,
        1.0e-13,
        "direct free-energy wrapper",
    )

    for density in [
        1.0e-5,
        0.1,
        0.2,
        0.225,
        0.25,
        0.5,
        0.88,
    ]:
        result = (
            calculate_lj_perturbation_free_energy(
                temperature_reduced=1.0,
                reduced_density=density,
            )
        )

        values = np.asarray(
            list(result.__dict__.values()),
            dtype=np.float64,
        )

        if not np.isfinite(values).all():
            raise AssertionError(
                f"Non-finite result at rho*={density}."
            )

        print(
            f"PASS  finite LJ state rho*={density:<7g} "
            f"beta_a={result.beta_total_free_energy_per_particle:.8g} "
            f"|a2/a1|={result.second_to_first_abs_ratio:.6g}"
        )

    print(
        "\nAll LJ perturbation tests passed."
    )


if __name__ == "__main__":
    main()
