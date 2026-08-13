"""Tests for the low-density hard-sphere RDF."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from huang_md.hard_sphere import (
    cs_reduced_isothermal_compressibility,
)
from huang_md.hard_sphere_rdf import (
    MAX_PACKING_FRACTION,
    compressibility_from_rdf,
    cs_contact_rdf,
    hard_sphere_rdf_reduced,
    low_density_rdf_parameters,
)


def assert_close(
    actual: float,
    expected: float,
    tolerance: float,
    name: str,
) -> None:
    difference = abs(actual - expected)

    if difference > tolerance:
        raise AssertionError(
            f"{name}: expected {expected:.12g}, "
            f"obtained {actual:.12g}, "
            f"difference={difference:.3e}"
        )

    print(
        f"PASS  {name:<42s} "
        f"value={actual:.10g}"
    )


def main() -> None:
    eta_values = [
        1.0e-6,
        0.0013422877,
        0.0153932077,
        0.0307864155,
        0.0615728309,
    ]

    for eta in eta_values:
        distances = np.linspace(
            0.0,
            4.0,
            4001,
        )

        rdf = hard_sphere_rdf_reduced(
            distances,
            eta,
        )

        if np.any(rdf < -1.0e-12):
            raise AssertionError(
                f"Negative RDF found at eta={eta}."
            )

        print(
            f"PASS  nonnegative RDF at eta={eta:.8f}"
        )

        core = hard_sphere_rdf_reduced(
            np.array([0.0, 0.5, 0.999999]),
            eta,
        )

        if not np.all(core == 0.0):
            raise AssertionError(
                f"Hard-core condition failed at eta={eta}."
            )

        print(
            f"PASS  hard-core exclusion at eta={eta:.8f}"
        )

        contact = float(
            hard_sphere_rdf_reduced(
                np.array([1.0]),
                eta,
            )[0]
        )

        assert_close(
            contact,
            cs_contact_rdf(eta),
            1.0e-12,
            f"contact value eta={eta:.8f}",
        )

        tail = hard_sphere_rdf_reduced(
            np.array([2.0, 3.0, 10.0]),
            eta,
        )

        if not np.allclose(
            tail,
            1.0,
            atol=1.0e-14,
            rtol=0.0,
        ):
            raise AssertionError(
                f"Long-range limit failed at eta={eta}."
            )

        print(
            f"PASS  long-range limit at eta={eta:.8f}"
        )

        rdf_compressibility = (
            compressibility_from_rdf(eta)
        )

        cs_compressibility = float(
            np.asarray(
                cs_reduced_isothermal_compressibility(
                    eta
                )
            ).reshape(-1)[0]
        )

        assert_close(
            rdf_compressibility,
            cs_compressibility,
            2.0e-11,
            f"compressibility eta={eta:.8f}",
        )

        parameters = low_density_rdf_parameters(
            eta
        )

        if not np.isfinite(
            parameters.correction_amplitude
        ):
            raise AssertionError(
                "Non-finite RDF correction amplitude."
            )

    try:
        hard_sphere_rdf_reduced(
            np.array([1.2]),
            MAX_PACKING_FRACTION + 0.01,
        )
    except ValueError:
        print(
            "PASS  low-density range enforcement"
        )
    else:
        raise AssertionError(
            "Out-of-range packing fraction was accepted."
        )

    print(
        "\nAll low-density hard-sphere RDF tests passed."
    )


if __name__ == "__main__":
    main()
