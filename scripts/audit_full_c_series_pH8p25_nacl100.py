#!/usr/bin/env python3

import csv
import re
import statistics
from pathlib import Path

ROOT = Path("results")
OUT = ROOT / "trusted_database_20260731"
OUT.mkdir(parents=True, exist_ok=True)

RUN_PATTERN = re.compile(
    r"^pH8p2500?_nacl100_c"
    r"(?P<c>\d+(?:p\d+)?)_seed(?P<seed>\d+)$"
)


def decode(value):
    return float(value.replace("p", "."))


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mean_value(rows, column):
    values = []

    for row in rows:
        try:
            values.append(float(row[column]))
        except (KeyError, TypeError, ValueError):
            pass

    return statistics.mean(values) if values else None


def max_value(rows, column):
    values = []

    for row in rows:
        try:
            values.append(float(row[column]))
        except (KeyError, TypeError, ValueError):
            pass

    return max(values) if values else None


records = []

for cluster_file in ROOT.rglob(
    "cluster_analysis/cluster_statistics_by_frame.csv"
):
    run_dir = cluster_file.parent.parent
    run_id = run_dir.name

    match = RUN_PATTERN.match(run_id)
    if match is None:
        continue

    concentration = decode(match.group("c"))
    seed = int(match.group("seed"))

    cluster_rows = read_rows(cluster_file)
    if not cluster_rows:
        continue

    start = max(
        0,
        min(
            len(cluster_rows) - 5,
            int(len(cluster_rows) * 0.8),
        ),
    )
    tail = cluster_rows[start:]

    final = cluster_rows[-1]

    dynamics_file = (
        run_dir
        / "dynamics_analysis"
        / "dynamics_by_frame.csv"
    )

    dynamics_tail = []

    if dynamics_file.exists():
        dynamics_rows = read_rows(dynamics_file)

        d_start = max(
            0,
            min(
                len(dynamics_rows) - 5,
                int(len(dynamics_rows) * 0.8),
            ),
        )
        dynamics_tail = dynamics_rows[d_start:]

    try:
        n_particles = (
            int(float(final["monomer_count"]))
            + int(float(final["clustered_particle_count"]))
        )
    except Exception:
        n_particles = 500

    mean_lcf = mean_value(tail, "largest_cluster_fraction")
    maximum_lcf = max_value(tail, "largest_cluster_fraction")
    mean_clustered = mean_value(tail, "clustered_fraction")
    weight_average = mean_value(
        tail,
        "weight_average_nontrivial_cluster_size",
    )
    mean_cluster_size = mean_value(
        tail,
        "mean_nontrivial_cluster_size",
    )
    retention = mean_value(
        dynamics_tail,
        "consecutive_bond_retention",
    )
    msd = mean_value(dynamics_tail, "msd_nm2")

    largest_particle_count = (
        mean_lcf * n_particles
        if mean_lcf is not None
        else None
    )

    # 暂定的结构尺度分类，不直接依赖旧 evidence 标签
    if mean_lcf is None:
        structural_class = "unknown"

    elif mean_lcf < 0.02:
        structural_class = "soluble_or_small_oligomers"

    elif mean_lcf < 0.05:
        structural_class = "intermediate_clustered"

    else:
        structural_class = "mesoscopic_condensed_cluster"

    metric_files = list(run_dir.parent.glob("*metrics.csv"))
    old_evidence = ""

    if metric_files:
        metric_rows = read_rows(metric_files[0])
        if metric_rows:
            old_evidence = metric_rows[0].get("evidence", "")

    records.append({
        "concentration_mg_ml": concentration,
        "seed": seed,
        "run_id": run_id,
        "N_particles": n_particles,
        "tail_frames": len(tail),
        "mean_clustered_fraction": mean_clustered,
        "mean_largest_cluster_fraction": mean_lcf,
        "maximum_largest_cluster_fraction": maximum_lcf,
        "mean_largest_cluster_particles": largest_particle_count,
        "final_largest_cluster_size": final.get(
            "largest_cluster_size",
            "",
        ),
        "mean_nontrivial_cluster_size": mean_cluster_size,
        "mean_weight_average_cluster_size": weight_average,
        "mean_bond_retention": retention,
        "mean_msd_nm2": msd,
        "old_evidence": old_evidence,
        "structural_class": structural_class,
        "cluster_file": str(cluster_file),
    })


records.sort(
    key=lambda row: (
        row["concentration_mg_ml"],
        row["seed"],
    )
)

fields = [
    "concentration_mg_ml",
    "seed",
    "run_id",
    "N_particles",
    "tail_frames",
    "mean_clustered_fraction",
    "mean_largest_cluster_fraction",
    "maximum_largest_cluster_fraction",
    "mean_largest_cluster_particles",
    "final_largest_cluster_size",
    "mean_nontrivial_cluster_size",
    "mean_weight_average_cluster_size",
    "mean_bond_retention",
    "mean_msd_nm2",
    "old_evidence",
    "structural_class",
    "cluster_file",
]

output = OUT / "pH8p25_nacl100_full_concentration_audit.csv"

with output.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)

print(
    "c\tseed\tclustered\tlargest_frac\tlargest_N\t"
    "weight_avg\tretention\told_label\tstructural_class"
)

for row in records:
    def fmt(value, digits=6):
        if value is None:
            return ""
        return f"{value:.{digits}f}"

    print(
        fmt(row["concentration_mg_ml"], 6),
        row["seed"],
        fmt(row["mean_clustered_fraction"]),
        fmt(row["mean_largest_cluster_fraction"]),
        fmt(row["mean_largest_cluster_particles"], 2),
        fmt(row["mean_weight_average_cluster_size"], 3),
        fmt(row["mean_bond_retention"]),
        row["old_evidence"],
        row["structural_class"],
        sep="\t",
    )

print()
print("Saved:", output)
