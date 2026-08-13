"""Deterministic tests for periodic wrapping detection."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from huang_md.percolation import (
    analyze_percolation_frame,
)


BOX_LENGTH_NM = 20.0


def check(
    name: str,
    positions: np.ndarray,
    cutoff_nm: float,
    expected_wraps: tuple[bool, bool, bool],
) -> None:
    result = analyze_percolation_frame(
        positions_nm=positions,
        box_length_nm=BOX_LENGTH_NM,
        bond_cutoff_nm=cutoff_nm,
    )

    actual = (
        result.wraps_x,
        result.wraps_y,
        result.wraps_z,
    )

    if actual != expected_wraps:
        raise AssertionError(
            f"{name}: expected {expected_wraps}, "
            f"obtained {actual}; "
            f"winding={result.winding_vectors.tolist()}"
        )

    print(
        f"PASS  {name:<28s} "
        f"wraps={actual}"
    )


def main() -> None:
    # A compact cluster does not wrap.
    check(
        "compact cluster",
        np.array(
            [
                [5.0, 5.0, 5.0],
                [6.0, 5.0, 5.0],
                [6.0, 6.0, 5.0],
                [5.0, 6.0, 5.0],
            ]
        ),
        cutoff_nm=1.1,
        expected_wraps=(False, False, False),
    )

    # A dimer crossing the boundary is one local bond, not a
    # periodic wrapping cycle.
    check(
        "boundary-crossing dimer",
        np.array(
            [
                [0.4, 5.0, 5.0],
                [19.6, 5.0, 5.0],
            ]
        ),
        cutoff_nm=1.0,
        expected_wraps=(False, False, False),
    )

    # Five particles form a closed ring around the x direction.
    check(
        "x-direction wrapping ring",
        np.array(
            [
                [0.5, 10.0, 10.0],
                [4.5, 10.0, 10.0],
                [8.5, 10.0, 10.0],
                [12.5, 10.0, 10.0],
                [16.5, 10.0, 10.0],
            ]
        ),
        cutoff_nm=4.1,
        expected_wraps=(True, False, False),
    )

    # Equivalent ring around y.
    check(
        "y-direction wrapping ring",
        np.array(
            [
                [10.0, 0.5, 10.0],
                [10.0, 4.5, 10.0],
                [10.0, 8.5, 10.0],
                [10.0, 12.5, 10.0],
                [10.0, 16.5, 10.0],
            ]
        ),
        cutoff_nm=4.1,
        expected_wraps=(False, True, False),
    )

    print("\nAll periodic percolation tests passed.")


if __name__ == "__main__":
    main()
