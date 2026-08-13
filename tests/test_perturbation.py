"""Tests for second-order liquid perturbation theory."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from huang_md.parameters import (
    HuangPotentialParameters,
)
from huang_md.perturbation import (
    calculate_perturbation_free_energy,
    perturbation_moments_reduced,
)
from huang_md.state_model import (
    RefCBAStateModel,
    parameters_for_state,
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
    first, second, _, _ = (
        perturbation_moments_reduced(
            potential_function=lambda x: (
                np.zeros_like(x)
            ),
            packing_fraction_value=0.02,
        )
    )

    assert_close(
        first,
        0.0,
        1.0e-12,
        "zero-potential first moment",
    )

    assert_close(
        second,
        0.0,
        1.0e-12,
        "zero-potential second moment",
    )

    baseline = (
        HuangPotentialParameters.from_yaml(
            ROOT
            / "configs"
            / "huang_baseline.yaml"
        )
    )

    state_model = (
        RefCBAStateModel.from_yaml(
            ROOT
            / "configs"
            / "refcba_state_model.yaml"
        )
    )

    pH3_params = parameters_for_state(
        baseline=baseline,
        model=state_model,
        pH=3.0,
        added_nacl_mM=0.0,
    )

    pI_params = parameters_for_state(
        baseline=baseline,
        model=state_model,
        pH=4.8852,
        added_nacl_mM=0.0,
    )

    density_low = 7.509969319836157e-06
    density_high = 1.5019938639672312e-03
    sigma_hs = 4.278

    pH3_low = calculate_perturbation_free_energy(
        pH3_params,
        density_low,
        sigma_hs,
    )

    pI_low = calculate_perturbation_free_energy(
        pI_params,
        density_low,
        sigma_hs,
    )

    pI_high = calculate_perturbation_free_energy(
        pI_params,
        density_high,
        sigma_hs,
    )

    for name, result in [
        ("pH3 low density", pH3_low),
        ("pI low density", pI_low),
        ("pI high density", pI_high),
    ]:
        values = np.array(
            [
                result.first_moment_integral,
                result.second_moment_integral,
                result.beta_a1_per_particle,
                result.beta_a2_per_particle,
                result.beta_total_free_energy_per_particle,
            ]
        )

        if not np.isfinite(values).all():
            raise AssertionError(
                f"Non-finite result for {name}."
            )

        print(
            f"PASS  finite perturbation result: {name}"
        )

        if result.second_moment_integral < 0.0:
            raise AssertionError(
                f"Negative squared-potential integral: {name}"
            )

        if result.beta_a2_per_particle > 1.0e-12:
            raise AssertionError(
                f"Second-order term is positive: {name}"
            )

    print(
        "PASS  second-order terms are non-positive"
    )

    # Compare the first-order mean interaction.
    # For a very strong interaction, the negative second-order u^2
    # correction may dominate and reverse the ordering of a1+a2.
    if not (
        pI_low.beta_a1_per_particle
        < pH3_low.beta_a1_per_particle
    ):
        raise AssertionError(
            "The first-order term does not identify pI as "
            "more attractive than pH 3."
        )

    print(
        "PASS  pI first-order interaction is more attractive "
        "than pH 3"
    )

    for name, result in [
        ("pH3 low density", pH3_low),
        ("pI low density", pI_low),
        ("pI high density", pI_high),
    ]:
        ratio = result.second_to_first_abs_ratio

        if not np.isfinite(ratio) or ratio < 0.0:
            raise AssertionError(
                f"Invalid perturbation convergence ratio: {name}"
            )

        print(
            f"INFO  {name:<24s} "
            f"|a2|/|a1|={ratio:.6g}"
        )

    if not (
        abs(
            pI_high.beta_perturbation_per_particle
        )
        >
        abs(
            pI_low.beta_perturbation_per_particle
        )
    ):
        raise AssertionError(
            "Perturbation magnitude did not grow with density."
        )

    print(
        "PASS  perturbation magnitude grows with density"
    )

    print(
        "\nAll liquid-perturbation tests passed."
    )


if __name__ == "__main__":
    main()
