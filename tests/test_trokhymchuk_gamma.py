"""Independent checks for corrected Trokhymchuk gamma."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from huang_md.trokhymchuk_parameters import (
    calculate_trokhymchuk_parameters,
    corrected_gamma,
)


EXPECTED = {
    0.2: 0.6455696846528698,
    0.5: 0.08873509806476783,
    0.9: -0.29492413138777634,
}


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
    results = []

    for density, expected_gamma in EXPECTED.items():
        result = calculate_trokhymchuk_parameters(
            density
        )

        results.append(result)

        assert_close(
            result.gamma,
            expected_gamma,
            1.0e-11,
            f"corrected gamma rho*={density}",
        )

        direct_gamma = corrected_gamma(
            packing_fraction=(
                result.packing_fraction
            ),
            mu_value=result.mu,
            alpha_value=result.alpha_py,
            beta_value=result.beta_py,
        )

        assert_close(
            direct_gamma,
            result.gamma,
            1.0e-14,
            f"stored/direct gamma rho*={density}",
        )

        if not np.isfinite(result.gamma):
            raise AssertionError(
                f"Non-finite gamma at rho*={density}."
            )

    if not (
        results[0].gamma
        > results[1].gamma
        > results[2].gamma
    ):
        raise AssertionError(
            "Gamma does not decrease over the test densities."
        )

    print("PASS  gamma decreases over test densities")
    print("\nAll corrected gamma tests passed.")


if __name__ == "__main__":
    main()
