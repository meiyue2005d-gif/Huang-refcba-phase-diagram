#!/usr/bin/env python3

import csv
import statistics
from pathlib import Path

ROOT = Path("results")

RUNS = [
    (
        "candidate_c0p3125",
        "pH8p2500_nacl100_c0p3125_seed20260845",
    ),
    (
        "candidate_c0p625",
        "pH8p2500_nacl100_c0p625_seed20260846",
    ),
    (
        "candidate_c0p9375",
        "pH8p2500_nacl100_c0p9375_seed20260847",
    ),
    (
        "candidate_c1p640625",
        "pH8p2500_nacl100_c1p640625_seed20260848",
    ),
    (
        "reference_soluble",
        "pH4p250_nacl0_c20p0_seed20260821",
    ),
    (
        "reference_LLPS",
        "pH8p500_nacl100_c20p0_seed20260822",
    ),
    (
        "reference_aggregation",
        "pH8p250_nacl100_c20p0_seed20260821",
    ),
]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mean_numeric(rows, column):
    values = []

    for row in rows:
        try:
            values.append(float(row[column]))
        except (KeyError, TypeError, ValueError):
            pass

    return statistics.mean(values) if values else None


def locate_unique(pattern):
    matches = list(ROOT.rglob(pattern))

    if not matches:
        return None

    if len(matches) > 1:
        print(
            f"WARNING：找到多个文件，使用第一个：{pattern}",
            flush=True,
        )

    return matches[0]


records = []

for label, run_id in RUNS:
    cluster_path = locate_unique(
        f"{run_id}/cluster_analysis/"
        "cluster_statistics_by_frame.csv"
    )

    dynamics_path = locate_unique(
        f"{run_id}/dynamics_analysis/"
        "dynamics_by_frame.csv"
    )

    if cluster_path is None:
        print(f"缺少 cluster 文件：{run_id}")
        continue

    cluster_rows = read_csv(cluster_path)

    if not cluster_rows:
        print(f"cluster 文件为空：{run_id}")
        continue

    # 使用最后20%的生产阶段帧，至少取最后5帧
    start = max(0, min(
        len(cluster_rows) - 5,
        int(len(cluster_rows) * 0.8),
    ))
    cluster_tail = cluster_rows[start:]

    final = cluster_rows[-1]

    dynamics_tail = []
    if dynamics_path is not None:
        dynamics_rows = read_csv(dynamics_path)
        d_start = max(0, min(
            len(dynamics_rows) - 5,
            int(len(dynamics_rows) * 0.8),
        ))
        dynamics_tail = dynamics_rows[d_start:]

    try:
        particle_number = (
            int(float(final["monomer_count"]))
            + int(float(final["clustered_particle_count"]))
        )
    except Exception:
        particle_number = ""

    record = {
        "label": label,
        "run_id": run_id,
        "N_particles": particle_number,
        "tail_frames": len(cluster_tail),
        "mean_clustered_fraction": mean_numeric(
            cluster_tail,
            "clustered_fraction",
        ),
        "mean_largest_cluster_fraction": mean_numeric(
            cluster_tail,
            "largest_cluster_fraction",
        ),
        "final_largest_cluster_size": final.get(
            "largest_cluster_size",
            "",
        ),
        "final_largest_cluster_fraction": final.get(
            "largest_cluster_fraction",
            "",
        ),
        "mean_monomer_fraction": mean_numeric(
            cluster_tail,
            "monomer_fraction",
        ),
        "mean_nontrivial_clusters": mean_numeric(
            cluster_tail,
            "n_nontrivial_clusters",
        ),
        "mean_nontrivial_cluster_size": mean_numeric(
            cluster_tail,
            "mean_nontrivial_cluster_size",
        ),
        "mean_weight_average_cluster_size": mean_numeric(
            cluster_tail,
            "weight_average_nontrivial_cluster_size",
        ),
        "mean_bond_retention": mean_numeric(
            dynamics_tail,
            "consecutive_bond_retention",
        ),
        "mean_msd_nm2": mean_numeric(
            dynamics_tail,
            "msd_nm2",
        ),
        "cluster_file": str(cluster_path),
    }

    records.append(record)


fields = [
    "label",
    "run_id",
    "N_particles",
    "tail_frames",
    "mean_clustered_fraction",
    "mean_largest_cluster_fraction",
    "final_largest_cluster_size",
    "final_largest_cluster_fraction",
    "mean_monomer_fraction",
    "mean_nontrivial_clusters",
    "mean_nontrivial_cluster_size",
    "mean_weight_average_cluster_size",
    "mean_bond_retention",
    "mean_msd_nm2",
    "cluster_file",
]

output = (
    ROOT
    / "trusted_database_20260731"
    / "low_concentration_cluster_audit.csv"
)
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)

print()
print("label\tN\tclustered\tlargest_frac\tlargest_size\t"
      "monomer_frac\tmean_cluster_size\tweight_avg\tretention\tmsd")

for row in records:
    print(
        row["label"],
        row["N_particles"],
        f'{row["mean_clustered_fraction"]:.6f}'
        if row["mean_clustered_fraction"] is not None else "",
        f'{row["mean_largest_cluster_fraction"]:.6f}'
        if row["mean_largest_cluster_fraction"] is not None else "",
        row["final_largest_cluster_size"],
        f'{row["mean_monomer_fraction"]:.6f}'
        if row["mean_monomer_fraction"] is not None else "",
        f'{row["mean_nontrivial_cluster_size"]:.4f}'
        if row["mean_nontrivial_cluster_size"] is not None else "",
        f'{row["mean_weight_average_cluster_size"]:.4f}'
        if row["mean_weight_average_cluster_size"] is not None else "",
        f'{row["mean_bond_retention"]:.6f}'
        if row["mean_bond_retention"] is not None else "",
        f'{row["mean_msd_nm2"]:.6f}'
        if row["mean_msd_nm2"] is not None else "",
        sep="\t",
    )

print()
print("Saved:", output)
