#!/usr/bin/env python3

from pathlib import Path
import csv
import json
import math

import numpy as np


ROOT = Path("results/pH9_nacl78p125_c20_seed20260863")
PARTICLES = 500
FRAME_INTERVAL_NS = 0.1
rows = []


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


for run in sorted(ROOT.glob("pH*_seed20260863")):
    cluster_file = (
        run / "cluster_analysis" / "cluster_summary.json"
    )
    dynamics_file = (
        run / "dynamics_analysis" / "dynamics_summary.json"
    )
    series_file = (
        run
        / "cluster_analysis"
        / "cluster_statistics_by_frame.csv"
    )

    required = [
        cluster_file,
        dynamics_file,
        series_file,
    ]

    if not all(path.exists() for path in required):
        print("SKIP，缺少分析文件：", run.name)

        for path in required:
            if not path.exists():
                print("  MISSING:", path)

        continue

    cluster = json.loads(
        cluster_file.read_text(encoding="utf-8")
    )
    dynamics = json.loads(
        dynamics_file.read_text(encoding="utf-8")
    )

    mean_size = get_number(
        cluster,
        [
            "mean_largest_cluster_size",
            "mean_largest_size",
        ],
    )

    max_size = get_number(
        cluster,
        [
            "maximum_largest_cluster_size",
            "max_largest_cluster_size",
        ],
    )

    final_size = get_number(
        cluster,
        [
            "final_largest_cluster_size",
            "last_largest_cluster_size",
        ],
    )

    mean_fraction = mean_size / PARTICLES
    max_fraction = max_size / PARTICLES
    final_fraction = final_size / PARTICLES
    dfm = final_fraction - mean_fraction

    clustered_fraction = get_number(
        cluster,
        [
            "mean_clustered_fraction",
            "clustered_fraction_mean",
        ],
    )

    if math.isnan(clustered_fraction):
        monomer_fraction = get_number(
            cluster,
            [
                "mean_monomer_fraction",
                "monomer_fraction_mean",
            ],
        )

        if not math.isnan(monomer_fraction):
            clustered_fraction = 1.0 - monomer_fraction

    bond = get_number(
        dynamics,
        [
            "final_initial_bond_survival",
            "final_initial_bond_survival_fraction",
            "final_bond_survival",
        ],
    )

    retention = get_number(
        dynamics,
        [
            "mean_consecutive_retention",
            "mean_consecutive_bond_retention",
            "consecutive_retention_mean",
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
                    float(record["largest_cluster_fraction"])
                )
            elif record.get("largest_cluster_size"):
                values.append(
                    float(record["largest_cluster_size"])
                    / PARTICLES
                )

    slope = float("nan")

    if len(values) >= 5:
        late = np.asarray(
            values[int(0.8 * len(values)):],
            dtype=float,
        )

        time_ns = (
            np.arange(len(late), dtype=float)
            * FRAME_INTERVAL_NS
        )

        if len(late) >= 2:
            slope = float(
                np.polyfit(time_ns, late, 1)[0]
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

    if soluble:
        evidence = "soluble_or_monomeric"
    elif percolated:
        evidence = "percolated_cluster"
    elif arrested:
        evidence = "arrested_aggregation_support"
    elif dynamic:
        evidence = "dynamic_condensation_support"
    else:
        evidence = "finite_mobile_cluster"

    rows.append({
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
    })


if not rows:
    raise SystemExit(
        "没有可汇总结果，请先运行 analyze_trajectory.py "
        "和 analyze_state_dynamics.py"
    )


output = ROOT / "pH9_nacl78p125_c20_seed20260863_metrics.csv"

columns = [
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
    writer.writerows(rows)


print()
print(
    f"{'STATE':49s} "
    f"{'CLUST':>7s} "
    f"{'MEAN':>7s} "
    f"{'MAX':>7s} "
    f"{'FINAL':>7s} "
    f"{'D-FM':>7s} "
    f"{'BOND':>7s} "
    f"{'RETAIN':>7s} "
    f"{'PERC':>7s} "
    f"{'SLOPE':>10s} "
    f"EVIDENCE"
)

for row in rows:
    print(
        f"{row['state']:49s} "
        f"{row['clustered']:7.3f} "
        f"{row['mean']:7.3f} "
        f"{row['max']:7.3f} "
        f"{row['final']:7.3f} "
        f"{row['dfm']:7.3f} "
        f"{row['bond']:7.3f} "
        f"{row['retain']:7.3f} "
        f"{row['percolation']:7.3f} "
        f"{row['slope']:10.5f} "
        f"{row['evidence']}"
    )

print()
print("Saved:", output)

print()
print("===== 建议补第二种子的点 =====")

selected = [
    row
    for row in rows
    if row["evidence"]
    == "dynamic_condensation_support"
]

if selected:
    for row in selected:
        print(row["state"])
else:
    print("本轮没有单种子动态凝聚支持点")
