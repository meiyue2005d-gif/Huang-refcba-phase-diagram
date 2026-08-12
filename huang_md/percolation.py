"""Periodic wrapping/percolation analysis for bonded colloids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from huang_md.clusters import (
    find_periodic_bond_pairs,
    validate_positions,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PercolationFrameResult:
    n_bonds: int
    wraps_x: bool
    wraps_y: bool
    wraps_z: bool
    wraps_any: bool
    winding_vectors: IntArray


class PeriodicUnionFind:
    """Union-find carrying integer periodic-image offsets.

    image_offset[i] stores m_i - m_parent(i), where m is the
    integer lattice image of a particle's unwrapped coordinate.
    """

    def __init__(self, n_items: int) -> None:
        self.parent = np.arange(n_items, dtype=np.int64)
        self.rank = np.zeros(n_items, dtype=np.int64)
        self.image_offset = np.zeros(
            (n_items, 3),
            dtype=np.int64,
        )

    def find(
        self,
        item: int,
    ) -> tuple[int, IntArray]:
        parent = int(self.parent[item])

        if parent == item:
            return item, np.zeros(3, dtype=np.int64)

        root, parent_offset = self.find(parent)

        total_offset = (
            self.image_offset[item]
            + parent_offset
        )

        self.parent[item] = root
        self.image_offset[item] = total_offset

        return root, total_offset.copy()

    def add_constraint(
        self,
        first: int,
        second: int,
        image_shift_second_minus_first: IntArray,
    ) -> IntArray:
        """Add m_second - m_first = image_shift.

        Returns a nonzero winding vector when the new bond closes
        a topologically nontrivial periodic cycle.
        """
        shift = np.asarray(
            image_shift_second_minus_first,
            dtype=np.int64,
        )

        root_first, offset_first = self.find(first)
        root_second, offset_second = self.find(second)

        if root_first == root_second:
            implied_shift = (
                offset_second
                - offset_first
            )

            return shift - implied_shift

        # m_root_second - m_root_first
        root_relation = (
            shift
            + offset_first
            - offset_second
        )

        if self.rank[root_first] >= self.rank[root_second]:
            self.parent[root_second] = root_first
            self.image_offset[root_second] = root_relation

            if self.rank[root_first] == self.rank[root_second]:
                self.rank[root_first] += 1
        else:
            self.parent[root_first] = root_second
            self.image_offset[root_first] = -root_relation

        return np.zeros(3, dtype=np.int64)


def analyze_percolation_frame(
    positions_nm: FloatArray,
    box_length_nm: float,
    bond_cutoff_nm: float,
) -> PercolationFrameResult:
    """Detect nonzero winding cycles under cubic PBC."""
    wrapped = validate_positions(
        positions_nm,
        box_length_nm,
    )

    bond_pairs = find_periodic_bond_pairs(
        positions_nm=wrapped,
        box_length_nm=box_length_nm,
        bond_cutoff_nm=bond_cutoff_nm,
    )

    union_find = PeriodicUnionFind(
        wrapped.shape[0]
    )

    winding_vectors: list[IntArray] = []

    for first_raw, second_raw in bond_pairs:
        first = int(first_raw)
        second = int(second_raw)

        raw_displacement = (
            wrapped[second]
            - wrapped[first]
        )

        image_shift = -np.rint(
            raw_displacement / box_length_nm
        ).astype(np.int64)

        winding = union_find.add_constraint(
            first=first,
            second=second,
            image_shift_second_minus_first=image_shift,
        )

        if np.any(winding != 0):
            winding_vectors.append(winding)

    if winding_vectors:
        winding_array = np.asarray(
            winding_vectors,
            dtype=np.int64,
        ).reshape(-1, 3)
    else:
        winding_array = np.empty(
            (0, 3),
            dtype=np.int64,
        )

    wraps_axes = np.any(
        winding_array != 0,
        axis=0,
    ) if winding_array.size else np.zeros(
        3,
        dtype=bool,
    )

    return PercolationFrameResult(
        n_bonds=int(bond_pairs.shape[0]),
        wraps_x=bool(wraps_axes[0]),
        wraps_y=bool(wraps_axes[1]),
        wraps_z=bool(wraps_axes[2]),
        wraps_any=bool(np.any(wraps_axes)),
        winding_vectors=winding_array,
    )
