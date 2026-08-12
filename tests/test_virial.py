"""Deterministic tests for second-virial calculations."""

from __future__ import annotations

import sys
from math import pi
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from huang_md.parameters import HuangPotentialParameters
from huang_md.virial import (
    calculate_virial_properties,
    gaussian_soft_core_reduced,
    hard_sphere_diameter_from_B2_reduced,
    second_virial_reduced_from_callable,
)


def assert_close(
    actual: float,
    expected: float,
    tolerance: float,
    name: str,
) -> None:
    difference = abs(
        actual - expected
    )

    if difference > tolerance:
        raise AssertionError(
            f"{name}: expected {expected:.12g}, "
            f"obtained {actual:.12g}, "
            f"difference={difference:.3e}"
        )

    print(
        f"PASS  {name:<36s} "
        f"value={actual:.10g}"
    )


def main() -> None:
    zero_B2, _ = (
        second_virial_reduced_from_callable(
            potential_function=lambda x: (
                np.zeros_like(x)
            ),
            upper_bound_reduced=2.0,
            breakpoints=(1.0,),
        )
    )

    assert_close(
        zero_B2,
        0.0,
        1.0e-10,
        "zero potential B2",
    )

    hard_sphere_B2, _ = (
        second_virial_reduced_from_callable(
            potential_function=lambda x: (
                np.where(
                    x < 1.0,
                    1.0e6,
                    0.0,
                )
            ),
            upper_bound_reduced=2.0,
            breakpoints=(1.0,),
        )
    )

    expected_hard_sphere_B2 = (
        2.0 * pi / 3.0
    )

    assert_close(
        hard_sphere_B2,
        expected_hard_sphere_B2,
        1.0e-8,
        "unit hard-sphere B2",
    )

    recovered_diameter = (
        hard_sphere_diameter_from_B2_reduced(
            hard_sphere_B2
        )
    )

    assert_close(
        recovered_diameter,
        1.0,
        1.0e-8,
        "hard-sphere diameter recovery",
    )

    baseline = (
        HuangPotentialParameters.from_yaml(
            ROOT
            / "configs"
            / "huang_baseline.yaml"
        )
    )

    x_at_contact = np.array(
        [1.0],
        dtype=np.float64,
    )

    core_at_contact = float(
        gaussian_soft_core_reduced(
            x_at_contact,
            baseline,
        )[0]
    )

    assert_close(
        core_at_contact,
        0.0,
        1.0e-10,
        "soft-core contact continuity",
    )

    result = calculate_virial_properties(
        baseline
    )

    if result.core_B2_reduced <= 0:
        raise AssertionError(
            "Baseline core B2 is not positive."
        )

    print(
        f"PASS  {'baseline core B2 positive':<36s} "
        f"value={result.core_B2_reduced:.10g}"
    )

    if not (
        0.5
        < result.effective_hs_diameter_reduced
        <= 1.0
    ):
        raise AssertionError(
            "Effective hard-sphere diameter is outside "
            "the expected reduced range."
        )

    print(
        f"PASS  {'effective diameter range':<36s} "
        f"value="
        f"{result.effective_hs_diameter_reduced:.10g}"
    )

    if not np.isfinite(
        result.full_B2_reduced
    ):
        raise AssertionError(
            "Full baseline B2 is not finite."
        )

    print(
        f"PASS  {'baseline full B2 finite':<36s} "
        f"value={result.full_B2_reduced:.10g}"
    )

    print("\nAll virial tests passed.")


if __name__ == "__main__":
    main()
