"""Tests for the complete Trokhymchuk hard-sphere RDF."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from huang_md.trokhymchuk_rdf import (
    calculate_trokhymchuk_rdf_coefficients,
    depletion_rdf_value,
    structural_rdf_value,
    trokhymchuk_rdf_from_reduced_density,
    trokhymchuk_rdf_reduced,
)


EXPECTED_COEFFICIENTS = {
    0.2: {
        "a": 0.6066351536736664,
        "b": 0.8950866489529509,
        "c": -13.474772313935752,
        "delta": -10.3749584872753,
    },
    0.5: {
        "a": 0.5316677651117945,
        "b": 1.6420189247871073,
        "c": -3.719921186907006,
        "delta": -10.582144076100334,
    },
    0.9: {
        "a": 0.3795438685509991,
        "b": 5.014651349601387,
        "c": -1.8159770150850025,
        "delta": -10.575235923220292,
    },
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
        f"PASS  {name:<45s} "
        f"value={actual:.10g}"
    )


def main() -> None:
    for density, reference in (
        EXPECTED_COEFFICIENTS.items()
    ):
        coefficients = (
            calculate_trokhymchuk_rdf_coefficients(
                density
            )
        )

        p = coefficients.base

        assert_close(
            coefficients.coefficient_a,
            reference["a"],
            1.0e-11,
            f"A coefficient rho*={density}",
        )

        assert_close(
            coefficients.coefficient_b,
            reference["b"],
            1.0e-11,
            f"B coefficient rho*={density}",
        )

        assert_close(
            coefficients.coefficient_c,
            reference["c"],
            1.0e-11,
            f"C coefficient rho*={density}",
        )

        assert_close(
            coefficients.structural_phase,
            reference["delta"],
            1.0e-11,
            f"delta coefficient rho*={density}",
        )

        contact = (
            trokhymchuk_rdf_from_reduced_density(
                1.0,
                density,
            )
        )

        assert_close(
            contact,
            p.contact_rdf,
            1.0e-11,
            f"contact condition rho*={density}",
        )

        depletion_merge = depletion_rdf_value(
            p.minimum_position,
            coefficients,
        )

        structural_merge = structural_rdf_value(
            p.minimum_position,
            coefficients,
        )

        assert_close(
            depletion_merge,
            p.minimum_rdf,
            1.0e-11,
            f"depletion merge value rho*={density}",
        )

        assert_close(
            structural_merge,
            p.minimum_rdf,
            1.0e-11,
            f"structural merge value rho*={density}",
        )

        assert_close(
            depletion_merge,
            structural_merge,
            1.0e-11,
            f"branch continuity rho*={density}",
        )

        # The structural branch is constructed to have
        # exactly zero derivative at the merging point.
        assert_close(
            coefficients.structural_derivative_at_merge,
            0.0,
            1.0e-12,
            f"structural zero slope rho*={density}",
        )

        # Published polynomial parameterizations slightly
        # relax the exact depletion-branch zero-slope
        # condition. The residual remains approximately
        # 10^-3 over the fitted range.
        if (
            abs(
                coefficients.depletion_derivative_at_merge
            )
            > 5.0e-3
        ):
            raise AssertionError(
                "Depletion merge slope is too large at "
                f"rho*={density}: "
                f"{coefficients.depletion_derivative_at_merge}"
            )

        print(
            "PASS  depletion near-zero slope "
            f"rho*={density:<3} "
            f"value="
            f"{coefficients.depletion_derivative_at_merge:.6g}"
        )

        distances = np.linspace(
            1.0,
            8.0,
            4001,
        )

        rdf = (
            trokhymchuk_rdf_from_reduced_density(
                distances,
                density,
            )
        )

        if not np.isfinite(rdf).all():
            raise AssertionError(
                f"Non-finite RDF at rho*={density}."
            )

        if np.min(rdf) <= 0.0:
            raise AssertionError(
                f"Non-positive RDF outside the core "
                f"at rho*={density}."
            )

        print(
            f"PASS  positive finite RDF rho*={density}"
        )

        tail = (
            trokhymchuk_rdf_from_reduced_density(
                20.0,
                density,
            )
        )

        assert_close(
            tail,
            1.0,
            1.0e-6,
            f"long-range limit rho*={density}",
        )

        core = (
            trokhymchuk_rdf_from_reduced_density(
                np.array(
                    [0.0, 0.5, 0.999999]
                ),
                density,
            )
        )

        if not np.all(core == 0.0):
            raise AssertionError(
                f"Hard-core condition failed at rho*={density}."
            )

        print(
            f"PASS  hard-core exclusion rho*={density}"
        )

        eta = p.packing_fraction

        wrapper_value = trokhymchuk_rdf_reduced(
            np.array(
                [1.0, p.minimum_position, 3.0]
            ),
            eta,
        )

        direct_value = (
            trokhymchuk_rdf_from_reduced_density(
                np.array(
                    [1.0, p.minimum_position, 3.0]
                ),
                density,
            )
        )

        if not np.allclose(
            wrapper_value,
            direct_value,
            rtol=0.0,
            atol=1.0e-13,
        ):
            raise AssertionError(
                "Packing-fraction wrapper disagrees "
                f"at rho*={density}."
            )

        print(
            f"PASS  packing-fraction wrapper rho*={density}"
        )

    for invalid_density in [
        0.0,
        0.19,
        0.91,
        1.0,
    ]:
        try:
            trokhymchuk_rdf_from_reduced_density(
                1.5,
                invalid_density,
            )
        except ValueError:
            print(
                "PASS  rejected invalid density "
                f"{invalid_density}"
            )
        else:
            raise AssertionError(
                "Accepted invalid density "
                f"{invalid_density}"
            )

    print(
        "\nAll Trokhymchuk RDF tests passed."
    )


if __name__ == "__main__":
    main()
