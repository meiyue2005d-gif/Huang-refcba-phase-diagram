#!/usr/bin/env python3
"""Validate refCBA pH charge and salt-screening calculations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from huang_md.electrostatics import (
    REFCBA_SEQUENCE,
    amino_acid_counts,
    debye_length_nm,
    estimate_isoelectric_point,
    net_charge,
    total_ionic_strength_mM,
    yukawa_screening_parameter,
)


OUTPUT_DIR = ROOT / "results" / "electrostatics_validation"

PARTICLE_DIAMETER_NM = 4.278
TEMPERATURE_K = 300.0
BACKGROUND_IONIC_STRENGTH_MILLI_MOLAR = 20.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pH_values = np.linspace(3.0, 9.0, 121)
    charges = net_charge(pH_values)

    added_salt_values = np.arange(
        0.0,
        500.0 + 1.0e-9,
        10.0,
    )

    total_ionic_strength = total_ionic_strength_mM(
        added_salt_values,
        background_mM=BACKGROUND_IONIC_STRENGTH_MILLI_MOLAR,
    )

    debye_lengths = debye_length_nm(
        total_ionic_strength,
        temperature_K=TEMPERATURE_K,
    )

    z2_values = yukawa_screening_parameter(
        total_ionic_strength,
        particle_diameter_nm=PARTICLE_DIAMETER_NM,
        temperature_K=TEMPERATURE_K,
    )

    charge_table = pd.DataFrame(
        {
            "pH": pH_values,
            "net_charge_e": charges,
            "absolute_charge_e": np.abs(charges),
        }
    )

    salt_table = pd.DataFrame(
        {
            "added_NaCl_mM": added_salt_values,
            "total_ionic_strength_mM": total_ionic_strength,
            "debye_length_nm": debye_lengths,
            "Z2_reduced": z2_values,
        }
    )

    charge_table.to_csv(
        OUTPUT_DIR / "refcba_charge_vs_pH.csv",
        index=False,
    )

    salt_table.to_csv(
        OUTPUT_DIR / "salt_screening.csv",
        index=False,
    )

    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    axis.plot(pH_values, charges, linewidth=2.0)
    axis.axhline(0.0, linewidth=1.0)
    axis.set_xlabel("pH")
    axis.set_ylabel("Estimated refCBA net charge (e)")
    axis.set_xlim(3.0, 9.0)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "refcba_charge_vs_pH.png",
        dpi=240,
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    axis.plot(
        added_salt_values,
        debye_lengths,
        linewidth=2.0,
    )
    axis.set_xlabel("Added NaCl (mM)")
    axis.set_ylabel("Debye length (nm)")
    axis.set_xlim(0.0, 500.0)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "debye_length_vs_NaCl.png",
        dpi=240,
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    axis.plot(
        added_salt_values,
        z2_values,
        linewidth=2.0,
    )
    axis.set_xlabel("Added NaCl (mM)")
    axis.set_ylabel("Reduced screening parameter Z2")
    axis.set_xlim(0.0, 500.0)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "Z2_vs_NaCl.png",
        dpi=240,
    )
    plt.close(figure)

    selected_pH = np.array(
        [3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0]
    )
    selected_charges = net_charge(selected_pH)

    selected_salt = np.array(
        [0.0, 50.0, 100.0, 200.0, 500.0]
    )
    selected_total_I = total_ionic_strength_mM(
        selected_salt,
        background_mM=BACKGROUND_IONIC_STRENGTH_MILLI_MOLAR,
    )
    selected_debye = debye_length_nm(selected_total_I)
    selected_z2 = yukawa_screening_parameter(
        selected_total_I,
        particle_diameter_nm=PARTICLE_DIAMETER_NM,
    )

    print("=" * 72)
    print("refCBA electrostatics validation")
    print("=" * 72)

    print(f"Sequence length : {len(REFCBA_SEQUENCE)} aa")
    print(f"Sequence        : {REFCBA_SEQUENCE}")
    print(f"Residue counts  : {amino_acid_counts()}")
    print(
        f"Estimated pI    : {estimate_isoelectric_point():.4f}"
    )

    print("\nSelected pH values:")
    print("      pH       net charge (e)")
    for pH, charge in zip(selected_pH, selected_charges):
        print(f"  {pH:7.2f}       {charge:10.5f}")

    print("\nSelected salt values:")
    print(
        "  added NaCl    total I      Debye length       Z2"
    )
    print(
        "      (mM)       (mM)            (nm)"
    )

    for salt, ionic, length, z2 in zip(
        selected_salt,
        selected_total_I,
        selected_debye,
        selected_z2,
    ):
        print(
            f"  {salt:10.1f}"
            f"  {ionic:10.1f}"
            f"  {length:14.6f}"
            f"  {z2:10.6f}"
        )

    print("\nGenerated files:")
    for output in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {output}")

    print("\nElectrostatics validation completed.")


if __name__ == "__main__":
    main()
