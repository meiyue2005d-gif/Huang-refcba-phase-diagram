"""Deterministic tests for periodic cluster detection."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from huang_md.clusters import analyze_cluster_frame


BOX_LENGTH_NM = 20.0
BOND_CUTOFF_NM = 2.0


def check(
    name: str,
    positions: np.ndarray,
    expected_sizes: list[int],
) -> None:
    result = analyze_cluster_frame(
        positions_nm=positions,
        box_length_nm=BOX_LENGTH_NM,
        bond_cutoff_nm=BOND_CUTOFF_NM,
    )

    actual_sizes = result.cluster_sizes.tolist()

    if actual_sizes != expected_sizes:
        raise AssertionError(
            f"{name}: expected {expected_sizes}, "
            f"obtained {actual_sizes}"
        )

    print(
        f"PASS  {name:<28s} "
        f"cluster sizes={actual_sizes}"
    )


def main() -> None:
    # 1. Four separated particles: four monomers.
    check(
        "all monomers",
        np.array(
            [
                [2.0, 2.0, 2.0],
                [7.0, 2.0, 2.0],
                [12.0, 2.0, 2.0],
                [17.0, 2.0, 2.0],
            ]
        ),
        [1, 1, 1, 1],
    )

    # 2. One ordinary dimer and two monomers.
    check(
        "ordinary dimer",
        np.array(
            [
                [2.0, 2.0, 2.0],
                [3.5, 2.0, 2.0],
                [10.0, 2.0, 2.0],
                [15.0, 2.0, 2.0],
            ]
        ),
        [2, 1, 1],
    )

    # 3. Dimer crossing the x periodic boundary.
    # Distance is 0.8 nm under the minimum-image convention.
    check(
        "periodic-boundary dimer",
        np.array(
            [
                [0.4, 5.0, 5.0],
                [19.6, 5.0, 5.0],
                [8.0, 8.0, 8.0],
            ]
        ),
        [2, 1],
    )

    # 4. Three-particle connected chain.
    # Particle 1 and 3 are not directly bonded, but all belong
    # to one connected component through particle 2.
    check(
        "connected trimer",
        np.array(
            [
                [2.0, 10.0, 10.0],
                [3.5, 10.0, 10.0],
                [5.0, 10.0, 10.0],
                [14.0, 14.0, 14.0],
            ]
        ),
        [3, 1],
    )

    # 5. Two independent dimers.
    check(
        "two dimers",
        np.array(
            [
                [2.0, 2.0, 2.0],
                [3.0, 2.0, 2.0],
                [12.0, 12.0, 12.0],
                [13.0, 12.0, 12.0],
            ]
        ),
        [2, 2],
    )

    print("\nAll periodic cluster tests passed.")


if __name__ == "__main__":
    main()
