"""Periodic-boundary cluster analysis for colloidal trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ClusterFrameResult:
    """Cluster statistics for one trajectory frame."""

    n_particles: int
    n_bonds: int
    n_clusters_total: int
    n_nontrivial_clusters: int

    monomer_count: int
    monomer_fraction: float

    clustered_particle_count: int
    clustered_fraction: float

    largest_cluster_size: int
    largest_cluster_fraction: float

    number_average_cluster_size: float
    mean_nontrivial_cluster_size: float
    weight_average_nontrivial_cluster_size: float

    cluster_sizes: IntArray


class UnionFind:
    """Disjoint-set structure for connected-component analysis."""

    def __init__(self, n_items: int) -> None:
        self.parent = np.arange(n_items, dtype=np.int64)
        self.size = np.ones(n_items, dtype=np.int64)

    def find(self, item: int) -> int:
        root = item

        while self.parent[root] != root:
            root = int(self.parent[root])

        while self.parent[item] != item:
            next_item = int(self.parent[item])
            self.parent[item] = root
            item = next_item

        return root

    def union(self, first: int, second: int) -> None:
        root_first = self.find(first)
        root_second = self.find(second)

        if root_first == root_second:
            return

        if self.size[root_first] < self.size[root_second]:
            root_first, root_second = root_second, root_first

        self.parent[root_second] = root_first
        self.size[root_first] += self.size[root_second]


def validate_positions(
    positions_nm: FloatArray,
    box_length_nm: float,
) -> FloatArray:
    """Validate and wrap positions into the primary periodic box."""
    positions = np.asarray(positions_nm, dtype=np.float64)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            "positions_nm must have shape (n_particles, 3)."
        )

    if positions.shape[0] == 0:
        raise ValueError("The position array is empty.")

    if not np.all(np.isfinite(positions)):
        raise ValueError("Positions contain NaN or infinity.")

    if box_length_nm <= 0:
        raise ValueError("box_length_nm must be positive.")

    return np.mod(positions, box_length_nm)


def find_periodic_bond_pairs(
    positions_nm: FloatArray,
    box_length_nm: float,
    bond_cutoff_nm: float,
) -> IntArray:
    """Find particle pairs within the bond cutoff under cubic PBC."""
    if bond_cutoff_nm <= 0:
        raise ValueError("bond_cutoff_nm must be positive.")

    if bond_cutoff_nm >= box_length_nm / 2.0:
        raise ValueError(
            "bond_cutoff_nm must be smaller than half the box length."
        )

    wrapped = validate_positions(
        positions_nm,
        box_length_nm,
    )

    tree = cKDTree(
        wrapped,
        boxsize=box_length_nm,
    )

    pairs = tree.query_pairs(
        r=bond_cutoff_nm,
        output_type="ndarray",
    )

    pairs = np.asarray(pairs, dtype=np.int64)

    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    return pairs.reshape(-1, 2)


def cluster_sizes_from_pairs(
    n_particles: int,
    bond_pairs: IntArray,
) -> IntArray:
    """Return connected-component sizes in descending order."""
    if n_particles <= 0:
        raise ValueError("n_particles must be positive.")

    union_find = UnionFind(n_particles)

    for first, second in bond_pairs:
        union_find.union(
            int(first),
            int(second),
        )

    roots = np.array(
        [
            union_find.find(index)
            for index in range(n_particles)
        ],
        dtype=np.int64,
    )

    _, counts = np.unique(
        roots,
        return_counts=True,
    )

    return np.sort(
        counts.astype(np.int64)
    )[::-1]


def analyze_cluster_frame(
    positions_nm: FloatArray,
    box_length_nm: float,
    bond_cutoff_nm: float,
) -> ClusterFrameResult:
    """Calculate cluster statistics for one frame."""
    positions = validate_positions(
        positions_nm,
        box_length_nm,
    )

    n_particles = int(positions.shape[0])

    pairs = find_periodic_bond_pairs(
        positions_nm=positions,
        box_length_nm=box_length_nm,
        bond_cutoff_nm=bond_cutoff_nm,
    )

    cluster_sizes = cluster_sizes_from_pairs(
        n_particles=n_particles,
        bond_pairs=pairs,
    )

    monomer_count = int(
        np.sum(cluster_sizes == 1)
    )

    nontrivial_sizes = cluster_sizes[
        cluster_sizes >= 2
    ]

    clustered_particle_count = int(
        np.sum(nontrivial_sizes)
    )

    largest_cluster_size = int(
        cluster_sizes[0]
    )

    n_clusters_total = int(
        cluster_sizes.size
    )

    n_nontrivial_clusters = int(
        nontrivial_sizes.size
    )

    number_average_cluster_size = (
        n_particles / n_clusters_total
    )

    if n_nontrivial_clusters > 0:
        mean_nontrivial = float(
            np.mean(nontrivial_sizes)
        )

        weight_average_nontrivial = float(
            np.sum(nontrivial_sizes**2)
            / np.sum(nontrivial_sizes)
        )
    else:
        mean_nontrivial = 0.0
        weight_average_nontrivial = 0.0

    return ClusterFrameResult(
        n_particles=n_particles,
        n_bonds=int(pairs.shape[0]),
        n_clusters_total=n_clusters_total,
        n_nontrivial_clusters=n_nontrivial_clusters,
        monomer_count=monomer_count,
        monomer_fraction=monomer_count / n_particles,
        clustered_particle_count=clustered_particle_count,
        clustered_fraction=(
            clustered_particle_count / n_particles
        ),
        largest_cluster_size=largest_cluster_size,
        largest_cluster_fraction=(
            largest_cluster_size / n_particles
        ),
        number_average_cluster_size=float(
            number_average_cluster_size
        ),
        mean_nontrivial_cluster_size=mean_nontrivial,
        weight_average_nontrivial_cluster_size=(
            weight_average_nontrivial
        ),
        cluster_sizes=cluster_sizes,
    )
