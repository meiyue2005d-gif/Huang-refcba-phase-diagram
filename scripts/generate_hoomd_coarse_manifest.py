#!/usr/bin/env python3

from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "hoomd_coarse_224_0p5ns"
MANIFEST = OUTPUT / "manifest.tsv"

PH_VALUES = [3.0, 4.0, 4.5, 4.8852, 5.5, 7.0, 9.0]
SALT_VALUES = [0.0, 100.0, 300.0, 500.0]
CONCENTRATIONS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
SEED = 20260718


def tag_number(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    rows = []

    for ph in PH_VALUES:
        for salt in SALT_VALUES:
            for concentration in CONCENTRATIONS:
                state_id = (
                    f"pH{tag_number(ph)}"
                    f"_nacl{tag_number(salt)}"
                    f"_c{tag_number(concentration)}"
                    f"_seed{SEED}"
                )

                rows.append(
                    {
                        "state_id": state_id,
                        "ph": ph,
                        "nacl_mM": salt,
                        "concentration_mg_ml": concentration,
                        "seed": SEED,
                    }
                )

    with MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "state_id",
                "ph",
                "nacl_mM",
                "concentration_mg_ml",
                "seed",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print("States  :", len(rows))
    print("Manifest:", MANIFEST)
    print("GENERATE_HOOMD_COARSE_MANIFEST: PASS")


if __name__ == "__main__":
    main()
