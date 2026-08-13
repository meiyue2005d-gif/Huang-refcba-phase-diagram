"""Versioned pH- and salt-dependent Huang-style colloidal models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray

from huang_md.electrostatics import (
    REFCBA_SEQUENCE,
    net_charge,
    total_ionic_strength_mM,
    validate_sequence,
)
from huang_md.parameters import HuangPotentialParameters


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RefCBAStateModel:
    """Calibrated pH- and salt-dependent electrostatic model.

    ``legacy_absolute_power`` exactly preserves the original project model.
    ``huang_eq10_charge_magnitude`` preserves the first revised project model.
    ``gouy_chapman_charge_salt_magnitude`` additionally places the ionic-
    strength correction inside the Gouy-Chapman ``asinh`` argument.  This is
    the default for new salt scans; applying a square-root salt factor after
    Eq. 10 is retained only for reproducibility.
    """

    reference_pH: float
    reference_added_NaCl_mM: float
    background_ionic_strength_mM: float

    K2_reference_kBT: float
    Z2_reference: float

    charge_exponent: float
    salt_amplitude_exponent: float
    minimum_charge_e: float

    model_id: str = "legacy_refcba_abs_charge_v1"
    charge_mapping: str = "legacy_absolute_power"
    protein_id: str = "refCBA_62aa"
    protein_sequence: str = REFCBA_SEQUENCE
    thermal_voltage_mV: float = 25.851999786435535
    validity_pH_min: float = 3.0
    validity_pH_max: float = 9.0
    validity_added_NaCl_max_mM: float = 0.0

    @classmethod
    def from_yaml(cls, filename: str | Path) -> "RefCBAStateModel":
        path = Path(filename)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)

        if "state_model" not in raw:
            raise ValueError(
                "YAML must contain a top-level 'state_model' section."
            )

        data = raw["state_model"]

        sequence = data.get("protein_sequence", REFCBA_SEQUENCE)
        sequence_fasta = data.get("sequence_fasta")
        if sequence_fasta is not None:
            fasta_path = (path.parent / str(sequence_fasta)).resolve()
            if not fasta_path.exists():
                raise FileNotFoundError(
                    f"Protein FASTA file not found: {fasta_path}"
                )
            sequence = "".join(
                line.strip()
                for line in fasta_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith(">")
            )

        model = cls(
            reference_pH=float(data["reference_pH"]),
            reference_added_NaCl_mM=float(
                data["reference_added_NaCl_mM"]
            ),
            background_ionic_strength_mM=float(
                data["background_ionic_strength_mM"]
            ),
            K2_reference_kBT=float(data["K2_reference_kBT"]),
            Z2_reference=float(data["Z2_reference"]),
            charge_exponent=float(data["charge_exponent"]),
            salt_amplitude_exponent=float(
                data["salt_amplitude_exponent"]
            ),
            minimum_charge_e=float(data["minimum_charge_e"]),
            model_id=str(
                data.get("model_id", "legacy_refcba_abs_charge_v1")
            ),
            charge_mapping=str(
                data.get("charge_mapping", "legacy_absolute_power")
            ),
            protein_id=str(data.get("protein_id", "refCBA_62aa")),
            protein_sequence=validate_sequence(str(sequence)),
            thermal_voltage_mV=float(
                data.get("thermal_voltage_mV", 25.851999786435535)
            ),
            validity_pH_min=float(data.get("validity_pH_min", 3.0)),
            validity_pH_max=float(data.get("validity_pH_max", 9.0)),
            validity_added_NaCl_max_mM=float(
                data.get("validity_added_NaCl_max_mM", 0.0)
            ),
        )

        model.validate()
        return model

    def validate(self) -> None:
        if self.background_ionic_strength_mM <= 0:
            raise ValueError(
                "background_ionic_strength_mM must be positive."
            )

        if self.K2_reference_kBT <= 0:
            raise ValueError("K2_reference_kBT must be positive.")

        if self.Z2_reference <= 0:
            raise ValueError("Z2_reference must be positive.")

        if self.charge_exponent <= 0:
            raise ValueError("charge_exponent must be positive.")

        if self.salt_amplitude_exponent < 0:
            raise ValueError(
                "salt_amplitude_exponent cannot be negative."
            )

        if self.minimum_charge_e < 0:
            raise ValueError("minimum_charge_e cannot be negative.")

        allowed_mappings = {
            "legacy_absolute_power",
            "huang_eq10_charge_magnitude",
            "gouy_chapman_charge_salt_magnitude",
        }
        if self.charge_mapping not in allowed_mappings:
            raise ValueError(
                "charge_mapping must be one of: "
                + ", ".join(sorted(allowed_mappings))
            )

        if not self.model_id.strip():
            raise ValueError("model_id cannot be empty.")

        validate_sequence(self.protein_sequence)

        if self.thermal_voltage_mV <= 0:
            raise ValueError("thermal_voltage_mV must be positive.")

        if self.validity_pH_min >= self.validity_pH_max:
            raise ValueError("validity_pH_min must be below validity_pH_max.")

        if self.validity_added_NaCl_max_mM < 0:
            raise ValueError(
                "validity_added_NaCl_max_mM cannot be negative."
            )

    def sequence(self, override: str | None = None) -> str:
        """Return a validated explicit override or the configured sequence."""
        return validate_sequence(
            self.protein_sequence if override is None else override
        )

    def applicability_flags(
        self,
        pH: float,
        added_nacl_mM: float,
    ) -> dict[str, bool]:
        """Report whether a state lies outside the declared calibration."""
        reference_charge = reference_charge_e(self)
        state_charge = float(
            net_charge([pH], sequence=self.protein_sequence)[0]
        )
        return {
            "outside_declared_pH_range": not (
                self.validity_pH_min <= pH <= self.validity_pH_max
            ),
            "added_salt_is_extrapolation": (
                added_nacl_mM > self.validity_added_NaCl_max_mM
            ),
            "charge_sign_reversal_is_extrapolation": (
                state_charge * reference_charge < 0.0
            ),
        }


def _as_array(values: ArrayLike) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def reference_charge_e(
    model: RefCBAStateModel,
    sequence: str | None = None,
) -> float:
    """Signed protein charge at the calibration pH."""
    charge = float(
        np.asarray(
            net_charge(
                np.array([model.reference_pH]),
                sequence=model.sequence(sequence),
            )
        )[0]
    )

    if abs(charge) <= 1.0e-12:
        raise ValueError(
            "Reference pH is too close to the isoelectric point."
        )

    return charge


def ionic_strength_mM(
    added_nacl_mM: ArrayLike,
    model: RefCBAStateModel,
) -> FloatArray:
    """Total ionic strength used by the implicit-salt model."""
    return total_ionic_strength_mM(
        added_nacl_mM,
        background_mM=model.background_ionic_strength_mM,
    )


def reference_ionic_strength_mM(
    model: RefCBAStateModel,
) -> float:
    """Total ionic strength at the Huang calibration state."""
    value = ionic_strength_mM(
        np.array([model.reference_added_NaCl_mM]),
        model,
    )
    return float(value[0])


def calculate_K2_kBT(
    pH: ArrayLike,
    added_nacl_mM: ArrayLike,
    model: RefCBAStateModel,
    sequence: str | None = None,
) -> FloatArray:
    """Calculate the pH- and salt-dependent repulsive amplitude.

    The model is normalized so that pH=4.5 and added NaCl=0 mM
    reproduce Huang Table S1 K2=53.056 kBT.
    """
    pH_array = _as_array(pH)
    salt_array = _as_array(added_nacl_mM)

    pH_broadcast, salt_broadcast = np.broadcast_arrays(
        pH_array,
        salt_array,
    )

    charge = net_charge(
        pH_broadcast,
        sequence=model.sequence(sequence),
    )

    charge_reference = reference_charge_e(
        model,
        sequence=sequence,
    )

    current_ionic_strength = ionic_strength_mM(
        salt_broadcast,
        model,
    )
    ionic_strength_reference = reference_ionic_strength_mM(model)

    if model.charge_mapping == "legacy_absolute_power":
        charge = np.abs(charge)
        if model.minimum_charge_e > 0:
            charge = np.maximum(charge, model.minimum_charge_e)
        charge_factor = (
            charge / abs(charge_reference)
        ) ** model.charge_exponent
        charge_mapped_K2 = model.K2_reference_kBT * charge_factor
    elif model.charge_mapping == "huang_eq10_charge_magnitude":
        # Huang et al. Eq. 10, using charge magnitude because two identical
        # proteins remain mutually repulsive after both reverse sign. Huang
        # did not calibrate across a sign reversal, so metadata separately
        # flags that region as an extrapolation.
        charge_ratio_magnitude = np.abs(charge / charge_reference)
        voltage = model.thermal_voltage_mV
        charge_mapped_K2 = 2.0 * voltage * np.arcsinh(
            charge_ratio_magnitude
            * np.sinh(model.K2_reference_kBT / (2.0 * voltage))
        )
    else:
        # Gouy-Chapman/Grahame extension at fixed charge density:
        # sigma is proportional to sqrt(I) * sinh(psi / 2 V_T).
        # Huang Eq. 10 is recovered exactly at the reference ionic strength.
        charge_ratio_magnitude = np.abs(charge / charge_reference)
        voltage = model.thermal_voltage_mV
        salt_ratio = np.sqrt(
            ionic_strength_reference / current_ionic_strength
        )
        charge_mapped_K2 = 2.0 * voltage * np.arcsinh(
            charge_ratio_magnitude
            * salt_ratio
            * np.sinh(model.K2_reference_kBT / (2.0 * voltage))
        )

    salt_factor = (
        ionic_strength_reference / current_ionic_strength
    ) ** model.salt_amplitude_exponent

    return (
        charge_mapped_K2 * salt_factor
    )


def calculate_Z2(
    added_nacl_mM: ArrayLike,
    model: RefCBAStateModel,
) -> FloatArray:
    """Calculate Z2 using Debye square-root ionic-strength scaling.

    The absolute value is anchored to Huang Table S1:
    Z2=1.483 at total ionic strength 20 mM.
    """
    salt_array = _as_array(added_nacl_mM)

    current_ionic_strength = ionic_strength_mM(
        salt_array,
        model,
    )

    ionic_strength_reference = reference_ionic_strength_mM(
        model
    )

    return (
        model.Z2_reference
        * np.sqrt(
            current_ionic_strength
            / ionic_strength_reference
        )
    )


def parameters_for_state(
    baseline: HuangPotentialParameters,
    model: RefCBAStateModel,
    pH: float,
    added_nacl_mM: float,
) -> HuangPotentialParameters:
    """Return complete Huang potential parameters for one state."""
    K2 = float(
        calculate_K2_kBT(
            np.array([pH]),
            np.array([added_nacl_mM]),
            model,
        )[0]
    )

    Z2 = float(
        calculate_Z2(
            np.array([added_nacl_mM]),
            model,
        )[0]
    )

    return replace(
        baseline,
        K2_kBT=K2,
        Z2=Z2,
    )
