#!/usr/bin/env python3

from pathlib import Path
import csv
import json
import math

import numpy as np


PARTICLES = 500
FRAME_INTERVAL_NS = 0.1

CONDITIONS = {
    "pH8.75_NaCl100": [
        Path(
            "results/llps_expansion_30ns_round5b/"
            "pH8p75_nacl100_c20p0_seed20260801"
        ),
        Path(
            "results/llps_expansion_30ns_round5c/"
            "pH8p75_nacl100_c20p0_seed20260802"
        ),
    ],
    "pH9.00_NaCl90": [
        Path(
            "results/llps_expansion_30ns_round5b/"
            "pH9p00_nacl90_c20p0_seed20260801"
        ),
        Path(
            "results/llps_expansion_30ns_round5c/"
            "pH9p00_nacl90_c20p0_seed20260802"
        ),
    ],
    "pH9.00_NaCl110": [
        Path(
            "results/llps_expansion_30ns_round5b/"
            "pH9p00_nacl110_c20p0_seed20260801"
        ),
        Path(
            "results/llps_expansion_30ns_round5c/"
            "pH9p00_nacl110_c20p0_seed20260802"
        ),
    ],
}


def get_number(data, keys, default=float("nan")):
    for key in keys:
        value = data.get(key)

        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return default


def read_metrics(run):
    cluster_file = (
        run / "cluster_analysis" / "cluster_summary.json"
    )
    dynamics_file = (
        run / "dynamics_analysis" / "dynamics_summary.json"
    )
    series_file = (
        run / "cluster_analysis"
        / "cluster_statistics_by_frame.csv"
    )

    required = [
        cluster_file,
        dynamics_file,
        series_file,
    ]

    missing = [
        path for path in required if not path.exists()
    ]

    if missing:
        print("SKIP，缺少文件：", run)

        for path in missing:
            print("  MISSING:", path)

        return None

    cluster = json.loads(
        cluster_file.read_text(encoding="utf-8")
    )
    dynamics = json.loads(
        dynamics_file.read_text(encoding="utf-8")
    )

    mean_size = get_number(
        cluster,
        ["mean_largest_cluster_size"],
    )
    max_size = get_number(
        cluster,
        ["maximum_largest_cluster_size"],
    )
    final_size = get_number(
        cluster,
        ["final_largest_cluster_size"],
    )

    mean_fraction = mean_size / PARTICLES
    max_fraction = max_size / PARTICLES
    final_fraction = final_size / PARTICLES
    dfm = final_fraction - mean_fraction

    clustered_fraction = get_number(
        cluster,
        ["mean_clustered_fraction"],
    )

    if math.isnan(clustered_fraction):
        monomer_fraction = get_number(
            cluster,
            ["mean_monomer_fraction"],
        )

        if not math.isnan(monomer_fraction):
            clustered_fraction = 1.0 - monomer_fraction

    bond = get_number(
        dynamics,
        [
            "final_initial_bond_survival",
            "final_initial_bond_survival_fraction",
        ],
    )

    retention = get_number(
        dynamics,
        [
            "mean_consecutive_retention",
            "mean_consecutive_bond_retention",
        ],
    )

    percolation = get_number(
        dynamics,
        ["percolation_fraction"],
    )

    values = []

    with series_file.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for record in csv.DictReader(handle):
            if record.get("largest_cluster_fraction"):
                values.append(
                    float(
                        record["largest_cluster_fraction"]
                    )
                )
            elif record.get("largest_cluster_size"):
                values.append(
                    float(record["largest_cluster_size"])
                    / PARTICLES
                )

    slope = float("nan")

    if len(values) >= 5:
        late_y = np.asarray(
            values[int(0.8 * len(values)):],
            dtype=float,
        )
        late_x = (
            np.arange(len(late_y), dtype=float)
            * FRAME_INTERVAL_NS
        )

        if len(late_y) >= 2:
            slope = float(
                np.polyfit(late_x, late_y, 1)[0]
            )

    soluble = (
        not math.isnan(clustered_fraction)
        and clustered_fraction < 0.10
        and max_fraction < 0.02
    )

    percolated = (
        not math.isnan(percolation)
        and percolation > 0.5
    )

    arrested = (
        not math.isnan(clustered_fraction)
        and clustered_fraction >= 0.50
        and not math.isnan(bond)
        and not math.isnan(retention)
        and bond >= 0.80
        and retention >= 0.95
    )

    dynamic = (
        max_fraction >= 0.14
        and not math.isnan(retention)
        and retention < 0.95
        and (
            (
                not math.isnan(slope)
                and slope >= 0.001
            )
            or dfm >= 0.025
        )
    )

    viscoelastic = (
        max_fraction >= 0.18
        and final_fraction >= 0.16
        and not math.isnan(slope)
        and slope >= 0.001
        and not math.isnan(retention)
        and retention >= 0.95
        and (
            math.isnan(bond)
            or bond < 0.80
        )
    )

    if soluble:
        evidence = "soluble_or_monomeric"
    elif percolated:
        evidence = "percolated_cluster"
    elif arrested:
        evidence = "arrested_aggregation_support"
    elif dynamic:
        evidence = "dynamic_condensation_support"
    elif viscoelastic:
        evidence = (
            "slow_dynamic_or_viscoelastic_candidate"
        )
    else:
        evidence = "finite_mobile_cluster"

    return {
        "seed": run.name.split("_seed")[-1],
        "state": run.name,
        "clustered": clustered_fraction,
        "mean": mean_fraction,
        "max": max_fraction,
        "final": final_fraction,
        "dfm": dfm,
        "bond": bond,
        "retain": retention,
        "percolation": percolation,
        "slope": slope,
        "evidence": evidence,
    }


all_rows = []

for condition, runs in CONDITIONS.items():
    for run in runs:
        row = read_metrics(run)

        if row is not None:
            row["condition"] = condition
            all_rows.append(row)


print()
print(
    f"{'CONDITION':20s} "
    f"{'SEED':10s} "
    f"{'CLUST':>7s} "
    f"{'MEAN':>7s} "
    f"{'MAX':>7s} "
    f"{'FINAL':>7s} "
    f"{'D-FM':>7s} "
    f"{'BOND':>7s} "
    f"{'RETAIN':>7s} "
    f"{'SLOPE':>10s} "
    f"EVIDENCE"
)

for row in all_rows:
    print(
        f"{row['condition']:20s} "
        f"{row['seed']:10s} "
        f"{row['clustered']:7.3f} "
        f"{row['mean']:7.3f} "
        f"{row['max']:7.3f} "
        f"{row['final']:7.3f} "
        f"{row['dfm']:7.3f} "
        f"{row['bond']:7.3f} "
        f"{row['retain']:7.3f} "
        f"{row['slope']:10.5f} "
        f"{row['evidence']}"
    )


print()
print("=" * 80)
print("MULTI-SEED CONSENSUS")
print("=" * 80)

for condition in CONDITIONS:
    selected = [
        row
        for row in all_rows
        if row["condition"] == condition
    ]

    labels = [
        row["evidence"]
        for row in selected
    ]

    dynamic_count = labels.count(
        "dynamic_condensation_support"
    )
    arrested_count = labels.count(
        "arrested_aggregation_support"
    )
    viscoelastic_count = labels.count(
        "slow_dynamic_or_viscoelastic_candidate"
    )
    finite_count = labels.count(
        "finite_mobile_cluster"
    )

    if len(selected) < 2:
        conclusion = "第二种子结果不完整"
    elif dynamic_count == 2:
        conclusion = "multi_seed_llps_supported"
    elif arrested_count == 2:
        conclusion = (
            "multi_seed_arrested_aggregation"
        )
    elif viscoelastic_count == 2:
        conclusion = (
            "multi_seed_viscoelastic_candidate"
        )
    elif dynamic_count == 1:
        conclusion = (
            "boundary_unresolved_seed_dependent"
        )
    elif (
        viscoelastic_count > 0
        or arrested_count > 0
    ):
        conclusion = (
            "seed_dependent_transition_state"
        )
    elif finite_count == 2:
        conclusion = "finite_mobile_cluster"
    else:
        conclusion = "unresolved"

    print(f"{condition:20s}: {conclusion}")


output = Path(
    "results/llps_expansion_30ns_round5c/"
    "round5b_round5c_multiseed_comparison.csv"
)

columns = [
    "condition",
    "seed",
    "state",
    "clustered",
    "mean",
    "max",
    "final",
    "dfm",
    "bond",
    "retain",
    "percolation",
    "slope",
    "evidence",
]

with output.open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=columns,
    )
    writer.writeheader()
    writer.writerows(all_rows)

print()
print("Saved:", output)
