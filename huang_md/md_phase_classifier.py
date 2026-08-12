"""Conservative MD phase classifier for the refCBA project.

The public three-state output is:

    soluble
    llps
    aggregate

Internally, soluble is separated into dispersed fluid and
mobile finite-cluster fluid.

Ordinary homogeneous NVT simulations are not allowed to
declare LLPS. LLPS requires an independent direct-coexistence
test plus evidence of dynamic exchange.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


CLUSTER_FLUID_THRESHOLD = 0.80
PERCOLATION_THRESHOLD = 0.50


@dataclass(frozen=True)
class MDPhaseMetrics:
    mean_clustered_fraction: float
    mean_largest_cluster_fraction: float
    percolation_fraction: float
    final_msd_nm2: float
    frozen_finite_clusters: bool

    direct_coexistence_confirmed: bool = False
    dynamic_exchange_confirmed: bool = False


@dataclass(frozen=True)
class MDPhaseClassification:
    final_phase: str
    internal_state: str
    confidence: str
    rationale: str


def _validate_fraction(
    value: float,
    name: str,
) -> float:
    number = float(value)

    if not math.isfinite(number):
        raise ValueError(
            f"{name} must be finite."
        )

    if number < 0.0 or number > 1.0:
        raise ValueError(
            f"{name} must lie in [0, 1], "
            f"received {number}."
        )

    return number


def classify_md_phase(
    metrics: MDPhaseMetrics,
) -> MDPhaseClassification:
    """Classify one MD state conservatively."""
    clustered_fraction = _validate_fraction(
        metrics.mean_clustered_fraction,
        "mean_clustered_fraction",
    )

    largest_fraction = _validate_fraction(
        metrics.mean_largest_cluster_fraction,
        "mean_largest_cluster_fraction",
    )

    percolation_fraction = _validate_fraction(
        metrics.percolation_fraction,
        "percolation_fraction",
    )

    final_msd = float(
        metrics.final_msd_nm2
    )

    if (
        not math.isfinite(final_msd)
        or final_msd < 0.0
    ):
        raise ValueError(
            "final_msd_nm2 must be nonnegative "
            "and finite."
        )

    is_percolated = (
        percolation_fraction
        > PERCOLATION_THRESHOLD
    )

    is_arrested = bool(
        metrics.frozen_finite_clusters
    )

    direct_coexistence = bool(
        metrics.direct_coexistence_confirmed
    )

    dynamic_exchange = bool(
        metrics.dynamic_exchange_confirmed
    )

    if (
        direct_coexistence
        and dynamic_exchange
        and (is_arrested or is_percolated)
    ):
        return MDPhaseClassification(
            final_phase="needs_review",
            internal_state=(
                "conflicting_llps_and_aggregate_signals"
            ),
            confidence="low",
            rationale=(
                "Direct coexistence and exchange were "
                "reported, but the state is also arrested "
                "or percolated."
            ),
        )

    if (
        direct_coexistence
        and dynamic_exchange
        and not is_arrested
        and not is_percolated
    ):
        return MDPhaseClassification(
            final_phase="llps",
            internal_state=(
                "mobile_two_phase_coexistence"
            ),
            confidence="high",
            rationale=(
                "A persistent dense and dilute coexistence "
                "profile was independently confirmed, with "
                "dynamic particle exchange and no arrest."
            ),
        )

    if (
        direct_coexistence
        and not dynamic_exchange
    ):
        return MDPhaseClassification(
            final_phase="needs_review",
            internal_state=(
                "persistent_dense_domain_without_"
                "confirmed_exchange"
            ),
            confidence="low",
            rationale=(
                "A dense domain persisted, but liquid-like "
                "dynamic exchange was not demonstrated."
            ),
        )

    if is_arrested:
        return MDPhaseClassification(
            final_phase="aggregate",
            internal_state=(
                "kinetically_arrested_finite_aggregates"
            ),
            confidence="medium",
            rationale=(
                "Finite clusters satisfy the project "
                "kinetic-arrest diagnostic."
            ),
        )

    if is_percolated:
        return MDPhaseClassification(
            final_phase="aggregate",
            internal_state=(
                "percolated_cluster_network"
            ),
            confidence="medium",
            rationale=(
                "A sample-spanning bonded network exists "
                "for more than half of the analyzed time."
            ),
        )

    if (
        clustered_fraction
        >= CLUSTER_FLUID_THRESHOLD
    ):
        return MDPhaseClassification(
            final_phase="soluble",
            internal_state=(
                "mobile_finite_cluster_fluid"
            ),
            confidence="medium",
            rationale=(
                "At least 80% of particles participate in "
                "clusters, but no kinetic arrest, "
                "percolation, or direct LLPS evidence "
                "is present."
            ),
        )

    return MDPhaseClassification(
        final_phase="soluble",
        internal_state=(
            "dispersed_or_dilute_cluster_fluid"
        ),
        confidence="medium",
        rationale=(
            "No kinetic arrest, persistent percolation, "
            "or independently confirmed LLPS was found."
        ),
    )
