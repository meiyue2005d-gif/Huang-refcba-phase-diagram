"""Verify that perturbation integrals accept interchangeable RDFs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from huang_md.perturbation import (
    perturbation_moments_reduced,
)


def finite_square_well(
    distance: np.ndarray,
) -> np.ndarray:
    x = np.asarray(distance, dtype=np.float64)

    return np.where(
        (x >= 1.0) & (x < 2.0),
        -1.0,
        0.0,
    )


def unit_rdf(
    distance: np.ndarray,
    packing_fraction: float,
) -> np.ndarray:
    del packing_fraction

    x = np.asarray(distance, dtype=np.float64)

    return np.where(
        x < 1.0,
        0.0,
        1.0,
    )


def doubled_rdf(
    distance: np.ndarray,
    packing_fraction: float,
) -> np.ndarray:
    del packing_fraction

    x = np.asarray(distance, dtype=np.float64)

    return np.where(
        x < 1.0,
        0.0,
        2.0,
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
        f"PASS  {name:<38s} "
        f"value={actual:.10g}"
    )


def main() -> None:
    first_unit, second_unit, _, _ = (
        perturbation_moments_reduced(
            potential_function=finite_square_well,
            packing_fraction_value=0.02,
            rdf_function=unit_rdf,
        )
    )

    first_double, second_double, _, _ = (
        perturbation_moments_reduced(
            potential_function=finite_square_well,
            packing_fraction_value=0.02,
            rdf_function=doubled_rdf,
        )
    )

    assert_close(
        first_double,
        2.0 * first_unit,
        1.0e-10,
        "first moment follows injected RDF",
    )

    assert_close(
        second_double,
        2.0 * second_unit,
        1.0e-10,
        "second moment follows injected RDF",
    )

    if first_unit >= 0.0:
        raise AssertionError(
            "Square-well first moment should be negative."
        )

    if second_unit <= 0.0:
        raise AssertionError(
            "Squared-potential moment should be positive."
        )

    print("PASS  injected RDF changes both integrals")
    print("\nAll RDF injection tests passed.")


if __name__ == "__main__":
    main()
