#!/usr/bin/env python3
"""Generate the NaCl=0 pilot pH-concentration scan."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = ROOT / "results" / "pilot_scan_nacl0"
MANIFEST_FILE = OUTPUT_ROOT / "pilot_manifest.csv"

PH_VALUES = [
    4.0,
    4.3,
    4.5,
    4.7,
    4.8,
    4.8852,
    5.0,
    5.2,
    5.5,
    6.0,
]

CONCENTRATIONS_MG_ML = [
    0.5,
    2.0,
    5.0,
    10.0,
    20.0,
]

NACL_MILLI_MOLAR = 0.0
BASE_SEED = 20260720


def safe_number(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []

    task_index = 0

    for pH in PH_VALUES:
        for concentration in CONCENTRATIONS_MG_ML:
            seed = BASE_SEED + task_index

            directory_name = (
                f"task_{task_index:03d}"
                f"_pH{safe_number(pH)}"
                f"_c{safe_number(concentration)}"
                f"_seed{seed}"
            )

            records.append(
                {
                    "task_id": task_index,
                    "pH": pH,
                    "nacl_mM": NACL_MILLI_MOLAR,
                    "concentration_mg_ml": concentration,
                    "seed": seed,
                    "output_dir": str(
                        OUTPUT_ROOT / directory_name
                    ),
                }
            )

            task_index += 1

    table = pd.DataFrame.from_records(records)
    table.to_csv(MANIFEST_FILE, index=False)

    print("=" * 72)
    print("Pilot scan manifest generated")
    print("=" * 72)
    print(f"pH values       : {len(PH_VALUES)}")
    print(f"Concentrations  : {len(CONCENTRATIONS_MG_ML)}")
    print(f"Total tasks     : {len(table)}")
    print(f"Manifest        : {MANIFEST_FILE}")

    print("\nFirst five tasks:")
    print(table.head().to_string(index=False))


if __name__ == "__main__":
    main()
