"""Tests for the corrected Trokhymchuk parameter layer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from huang_md.trokhymchuk_parameters import (
    calculate_trokhymchuk_parameters,
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
        f"PASS  {name:<42s} "
        f"value={actual:.10g}"
    )


def main() -> None:
    expected = {
        0.2: {
            "packing_fraction": 0.10471975511965977,
            "mu": 0.9887339013323715,
            "minimum_position": 1.900694859742741,
            "minimum_rdf": 0.9841771654773829,
            "contact_rdf": 1.3215920088158102,
        },
        0.5: {
            "packing_fraction": 0.2617993877991494,
            "mu": 1.3036764265043055,
            "minimum_position": 1.7365495856767639,
            "minimum_rdf": 0.8895597772838117,
            "contact_rdf": 2.1672263696915732,
        },
        0.9: {
            "packing_fraction": 0.47123889803846897,
            "mu": 1.564580356291778,
            "minimum_position": 1.5218191013859608,
            "minimum_rdf": 0.6160396473491216,
            "contact_rdf": 5.177683622358253,
        },
    }

    results = []

    for density, reference in expected.items():
        result = (
            calculate_trokhymchuk_parameters(
                density
            )
        )

        results.append(result)

        assert_close(
            result.packing_fraction,
            reference["packing_fraction"],
            1.0e-13,
            f"packing fraction rho*={density}",
        )

        assert_close(
            result.mu,
            reference["mu"],
            1.0e-11,
            f"corrected mu rho*={density}",
        )

        assert_close(
            result.minimum_position,
            reference["minimum_position"],
            1.0e-11,
            f"minimum position rho*={density}",
        )

        assert_close(
            result.minimum_rdf,
            reference["minimum_rdf"],
            1.0e-11,
            f"minimum RDF rho*={density}",
        )

        assert_close(
            result.contact_rdf,
            reference["contact_rdf"],
            1.0e-11,
            f"contact RDF rho*={density}",
        )

        values = np.array(
            list(result.__dict__.values()),
            dtype=np.float64,
        )

        if not np.isfinite(values).all():
            raise AssertionError(
                f"Non-finite parameters at rho*={density}."
            )

        print(
            f"PASS  finite parameters rho*={density}"
        )

    if not (
        results[0].contact_rdf
        < results[1].contact_rdf
        < results[2].contact_rdf
    ):
        raise AssertionError(
            "Contact RDF does not increase with density."
        )

    print(
        "PASS  contact RDF increases with density"
    )

    if not (
        results[0].minimum_position
        > results[1].minimum_position
        > results[2].minimum_position
    ):
        raise AssertionError(
            "First-minimum position does not decrease "
            "with density."
        )

    print(
        "PASS  minimum position decreases with density"
    )

    for invalid_density in [
        0.0,
        0.19,
        0.91,
        1.0,
    ]:
        try:
            calculate_trokhymchuk_parameters(
                invalid_density
            )
        except ValueError:
            print(
                "PASS  rejected out-of-range density "
                f"{invalid_density}"
            )
        else:
            raise AssertionError(
                "Accepted invalid reduced density: "
                f"{invalid_density}"
            )

    print(
        "\nAll Trokhymchuk parameter tests passed."
    )


if __name__ == "__main__":
    main()
