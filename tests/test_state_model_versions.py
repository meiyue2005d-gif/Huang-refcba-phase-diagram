from __future__ import annotations

from pathlib import Path

import numpy as np

from huang_md.state_model import RefCBAStateModel, calculate_K2_kBT


ROOT = Path(__file__).resolve().parents[1]


def _k2(model: RefCBAStateModel, pH: float, salt: float = 0.0) -> float:
    return float(calculate_K2_kBT([pH], [salt], model)[0])


def test_legacy_configuration_preserves_original_mapping() -> None:
    model = RefCBAStateModel.from_yaml(
        ROOT / "configs" / "refcba_state_model_legacy.yaml"
    )
    assert np.isclose(_k2(model, 4.5), 53.056, rtol=0, atol=1e-10)
    assert np.isclose(
        _k2(model, 9.0),
        127.8248732165736,
        rtol=0,
        atol=1e-10,
    )


def test_eq10_magnitude_maps_charge_reversal_without_discontinuity() -> None:
    model = RefCBAStateModel.from_yaml(
        ROOT / "configs" / "refcba_state_model.yaml"
    )
    assert np.isclose(_k2(model, 4.5), 53.056, rtol=0, atol=1e-10)
    assert _k2(model, 4.8852) < 0.01
    assert _k2(model, 7.0) > 0.0
    assert _k2(model, 9.0) > _k2(model, 7.0)


def test_huang_a1_eq10_matches_project_reference_table() -> None:
    model = RefCBAStateModel.from_yaml(
        ROOT / "configs" / "huang_a1_state_model.yaml"
    )
    expected = {4.5: 53.056, 5.0: 46.6180923428, 7.0: 11.1943434037}
    for pH, target in expected.items():
        assert np.isclose(_k2(model, pH), target, rtol=0, atol=1e-8)


def test_applicability_flags_make_salt_extrapolation_explicit() -> None:
    model = RefCBAStateModel.from_yaml(
        ROOT / "configs" / "huang_a1_state_model.yaml"
    )
    assert model.applicability_flags(6.0, 0.0) == {
        "outside_declared_pH_range": False,
        "added_salt_is_extrapolation": False,
        "charge_sign_reversal_is_extrapolation": False,
    }
    assert model.applicability_flags(8.0, 100.0) == {
        "outside_declared_pH_range": True,
        "added_salt_is_extrapolation": True,
        "charge_sign_reversal_is_extrapolation": False,
    }

    refcba = RefCBAStateModel.from_yaml(
        ROOT / "configs" / "refcba_state_model.yaml"
    )
    assert refcba.applicability_flags(7.0, 0.0)[
        "charge_sign_reversal_is_extrapolation"
    ]
