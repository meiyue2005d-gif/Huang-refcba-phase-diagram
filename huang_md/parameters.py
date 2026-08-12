"""Parameter definitions for the Huang reflectin colloidal model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class HuangPotentialParameters:
    """Parameters expressed in reduced energy units of kBT."""

    temperature_K: float
    reference_pH: float
    reference_ionic_strength_mM: float

    radius_gyration_nm: float
    diameter_nm: float

    K1_kBT: float
    Z1: float
    K2_kBT: float
    Z2: float

    gaussian_sigma_reduced: float
    gaussian_epsilon_kBT: float
    cutoff_reduced: float

    @classmethod
    def from_yaml(
        cls,
        filename: str | Path,
    ) -> "HuangPotentialParameters":
        path = Path(filename)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)

        if "model" not in raw:
            raise ValueError("YAML file must contain a top-level 'model' section.")

        data = raw["model"]

        return cls(
            temperature_K=float(data["temperature_K"]),
            reference_pH=float(data["reference_pH"]),
            reference_ionic_strength_mM=float(
                data["reference_ionic_strength_mM"]
            ),
            radius_gyration_nm=float(data["radius_gyration_nm"]),
            diameter_nm=float(data["diameter_nm"]),
            K1_kBT=float(data["K1_kBT"]),
            Z1=float(data["Z1"]),
            K2_kBT=float(data["K2_kBT"]),
            Z2=float(data["Z2"]),
            gaussian_sigma_reduced=float(
                data["gaussian_sigma_reduced"]
            ),
            gaussian_epsilon_kBT=float(
                data["gaussian_epsilon_kBT"]
            ),
            cutoff_reduced=float(data["cutoff_reduced"]),
        )

    def validate(
        self,
        require_salr: bool = False,
    ) -> None:
        """Validate the numerical pair-potential parameters.

        Parameters
        ----------
        require_salr
            When True, additionally require Z1 > Z2 so that the
            repulsion is longer ranged than the attraction.

            High-salt states may legitimately leave the strict SA-LR
            regime, so ordinary state calculations use False.
        """
        if self.temperature_K <= 0:
            raise ValueError("temperature_K must be positive.")

        if self.diameter_nm <= 0:
            raise ValueError("diameter_nm must be positive.")

        if self.K1_kBT <= 0:
            raise ValueError("K1 must be positive.")

        if self.K2_kBT < 0:
            raise ValueError("K2 cannot be negative.")

        if self.Z1 <= 0 or self.Z2 <= 0:
            raise ValueError("Z1 and Z2 must be positive.")

        if require_salr and self.Z1 <= self.Z2:
            raise ValueError(
                "Strict SA-LR requires Z1 > Z2, so the repulsion "
                "is longer ranged than the attraction."
            )

        if self.gaussian_sigma_reduced <= 0:
            raise ValueError("Gaussian sigma must be positive.")

        if self.gaussian_epsilon_kBT <= 0:
            raise ValueError("Gaussian epsilon must be positive.")

        if self.cutoff_reduced <= 1:
            raise ValueError(
                "cutoff_reduced must be greater than 1."
            )

    @property
    def is_salr(self) -> bool:
        """Whether this state satisfies the strict SA-LR condition."""
        return self.Z1 > self.Z2

    def validate_salr(self) -> None:
        """Validate parameters and require the strict SA-LR form."""
        self.validate(require_salr=True)
