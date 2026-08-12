#!/usr/bin/env python3

import csv
import math
import statistics
from pathlib import Path

ROOT = Path("results")
OUT = ROOT / "trusted_database_20260731"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = [
    {
        "label": "c5_seed20260828",
        "concentration": 5.0,
        "seed": 20260828,
        "patterns": [
            "pH8p2500_nacl100_c5p0_seed20260828",
            "pH8p250_nacl100_c5p0_seed20260828",
        ],
    },
    {
        "label": "c10_seed20260827",
        "concentration": 10.0,
        "seed": 20260827,
        "patterns": [
            "pH8p2500_nacl100_c10p0_seed20260827",
            "pH8p250_nacl100_c10p0_seed20260827",
        ],
    },
    {
        "label": "c20_seed20260821",
        "concentration": 20.0,
        "seed": 20260821,
        "patterns": [
            "pH8p2500_nacl100_c20p0_seed20260821",
            "pH8p250_nacl100_c20p0_seed20260821",
        ],
    },
    {
        "label": "c20_seed20260822",
        "concentration": 20.0,
        "seed": 20260822,
        "patterns": [
            "pH8p2500_nacl100_c20p0_seed20260822",
            "pH8p250_nacl100_c20p0_seed20260822",
        ],
    },
]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def numeric_values(rows, column):
    values = []

    for row in rows:
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError):
            continue

        if math.isfinite(value):
            values.append(value)

    return values


def mean_or_none(values):
    return statistics.mean(values) if values else None


def sd_or_none(values):
    return statistics.pstdev(values) if len(values) >= 2 else 0.0 if values else None


def linear_slope(x_values, y_values):
    pairs = [
        (float(x), float(y))
        for x, y in zip(x_values, y_values)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]

    if len(pairs) < 2:
        return None

    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]

    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)

    denominator = sum((x - x_mean) ** 2 for x in xs)

    if denominator == 0:
        return 0.0

    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in pairs
    )

    return numerator / denominator


def tail_slice(rows, fraction=0.20, minimum=10):
    if not rows:
        return []

    count = max(minimum, int(len(rows) * fraction))
    count = min(count, len(rows))

    return rows[-count:]


def head_slice(rows, fraction=0.20, minimum=10):
    if not rows:
        return []

    count = max(minimum, int(len(rows) * fraction))
    count = min(count, len(rows))

    return rows[:count]


def locate_run_dir(patterns):
    matches = []

    for pattern in patterns:
        for path in ROOT.rglob(pattern):
            if path.is_dir():
                cluster_file = (
                    path
                    / "cluster_analysis"
                    / "cluster_statistics_by_frame.csv"
                )

                if cluster_file.exists():
                    matches.append(path)

    unique = []
    seen = set()

    for path in matches:
        resolved = str(path.resolve())

        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)

    if not unique:
        return None

    # 优先选择 production_thermo 最长的目录
    def duration_score(run_dir):
        thermo = run_dir / "production_thermo.csv"

        if not thermo.exists():
            return -1.0

        rows = read_csv(thermo)
        times = numeric_values(rows, "time_ps")

        return max(times) if times else -1.0

    unique.sort(key=duration_score, reverse=True)

    if len(unique) > 1:
        print(
            "WARNING：同一运行找到多个目录，选择时长最长者：",
            unique[0],
        )

    return unique[0]


def find_exact_evidence(run_id):
    required = {
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
    }

    evidence_values = set()
    files = set()

    for path in ROOT.rglob("*.csv"):
        if OUT in path.parents:
            continue

        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)

                if not reader.fieldnames:
                    continue

                if not required.issubset(set(reader.fieldnames)):
                    continue

                for row in reader:
                    if str(row.get("state", "")).strip() == run_id:
                        evidence = str(row.get("evidence", "")).strip()

                        if evidence:
                            evidence_values.add(evidence)

                        files.add(str(path))

        except Exception:
            continue

    return "|".join(sorted(evidence_values)), "|".join(sorted(files))


def structure_class(largest_fraction, weight_average):
    if largest_fraction is None:
        return "unknown"

    if largest_fraction < 0.02:
        return "soluble_or_small_oligomers"

    if largest_fraction < 0.05:
        return "intermediate_clustered"

    if weight_average is not None and weight_average >= 10:
        return "mesoscopic_condensed"

    return "intermediate_to_mesoscopic"


def contact_persistence(retention):
    if retention is None:
        return "unknown"

    if retention >= 0.997:
        return "very_persistent_contacts"

    if retention >= 0.990:
        return "persistent_contacts"

    return "more_rearranging_contacts"


def growth_class(delta_largest, slope_per_ns):
    if delta_largest is None or slope_per_ns is None:
        return "unknown"

    if delta_largest > 0.02 and slope_per_ns > 0:
        return "continued_growth"

    if delta_largest < -0.02 and slope_per_ns < 0:
        return "net_decay"

    return "approximately_plateaued"


records = []

for config in RUNS:
    run_dir = locate_run_dir(config["patterns"])

    if run_dir is None:
        print(
            f'ERROR：找不到运行目录：{config["label"]}'
        )
        continue

    run_id = run_dir.name

    cluster_file = (
        run_dir
        / "cluster_analysis"
        / "cluster_statistics_by_frame.csv"
    )
    dynamics_file = (
        run_dir
        / "dynamics_analysis"
        / "dynamics_by_frame.csv"
    )
    thermo_file = run_dir / "production_thermo.csv"

    cluster_rows = read_csv(cluster_file)

    if not cluster_rows:
        print(f"ERROR：团簇文件为空：{cluster_file}")
        continue

    dynamics_rows = (
        read_csv(dynamics_file)
        if dynamics_file.exists()
        else []
    )

    thermo_rows = (
        read_csv(thermo_file)
        if thermo_file.exists()
        else []
    )

    cluster_early = head_slice(cluster_rows)
    cluster_late = tail_slice(cluster_rows)
    dynamics_late = tail_slice(dynamics_rows)

    early_largest = mean_or_none(
        numeric_values(
            cluster_early,
            "largest_cluster_fraction",
        )
    )

    late_largest_values = numeric_values(
        cluster_late,
        "largest_cluster_fraction",
    )

    late_largest = mean_or_none(late_largest_values)
    late_largest_sd = sd_or_none(late_largest_values)

    delta_largest = (
        late_largest - early_largest
        if late_largest is not None
        and early_largest is not None
        else None
    )

    cluster_times_ns = [
        value / 1000.0
        for value in numeric_values(cluster_rows, "time_ps")
    ]

    all_largest = numeric_values(
        cluster_rows,
        "largest_cluster_fraction",
    )

    largest_slope = linear_slope(
        cluster_times_ns,
        all_largest,
    )

    late_clustered = mean_or_none(
        numeric_values(
            cluster_late,
            "clustered_fraction",
        )
    )

    late_weight_average = mean_or_none(
        numeric_values(
            cluster_late,
            "weight_average_nontrivial_cluster_size",
        )
    )

    late_mean_cluster_size = mean_or_none(
        numeric_values(
            cluster_late,
            "mean_nontrivial_cluster_size",
        )
    )

    final_row = cluster_rows[-1]

    try:
        final_largest_size = float(
            final_row["largest_cluster_size"]
        )
    except (KeyError, TypeError, ValueError):
        final_largest_size = None

    try:
        n_particles = (
            float(final_row["monomer_count"])
            + float(final_row["clustered_particle_count"])
        )
    except (KeyError, TypeError, ValueError):
        n_particles = 500.0

    mean_largest_particles = (
        late_largest * n_particles
        if late_largest is not None
        else None
    )

    late_retention = mean_or_none(
        numeric_values(
            dynamics_late,
            "consecutive_bond_retention",
        )
    )

    late_initial_survival = mean_or_none(
        numeric_values(
            dynamics_late,
            "initial_bond_survival",
        )
    )

    wraps_values = []

    for row in dynamics_late:
        value = str(row.get("wraps_any", "")).strip().lower()

        if value in {"true", "1", "yes"}:
            wraps_values.append(1.0)
        elif value in {"false", "0", "no"}:
            wraps_values.append(0.0)

    wrapping_fraction = mean_or_none(wraps_values)

    dynamics_times_ns = [
        value / 1000.0
        for value in numeric_values(
            dynamics_rows,
            "elapsed_time_ps",
        )
    ]

    msd_values = numeric_values(
        dynamics_rows,
        "msd_nm2",
    )

    msd_slope = linear_slope(
        dynamics_times_ns,
        msd_values,
    )

    late_msd = mean_or_none(
        numeric_values(
            dynamics_late,
            "msd_nm2",
        )
    )

    duration_candidates = numeric_values(
        thermo_rows,
        "time_ps",
    )

    if duration_candidates:
        duration_ns = max(duration_candidates) / 1000.0
    elif cluster_times_ns:
        duration_ns = max(cluster_times_ns)
    else:
        duration_ns = None

    exact_evidence, evidence_files = find_exact_evidence(
        run_id
    )

    structural = structure_class(
        late_largest,
        late_weight_average,
    )

    persistence = contact_persistence(late_retention)

    growth = growth_class(
        delta_largest,
        largest_slope,
    )

    if structural == "soluble_or_small_oligomers":
        phase_status = "soluble_or_oligomeric"

    elif structural == "intermediate_clustered":
        phase_status = "structural_transition"

    elif structural in {
        "mesoscopic_condensed",
        "intermediate_to_mesoscopic",
    }:
        phase_status = (
            "condensed_unresolved_LLPS_vs_arrested"
        )

    else:
        phase_status = "unresolved"

    records.append({
        "label": config["label"],
        "concentration_mg_ml": config["concentration"],
        "seed": config["seed"],
        "run_id": run_id,
        "duration_ns": duration_ns,
        "n_frames_cluster": len(cluster_rows),
        "n_frames_dynamics": len(dynamics_rows),
        "mean_clustered_fraction_late": late_clustered,
        "mean_largest_cluster_fraction_early": early_largest,
        "mean_largest_cluster_fraction_late": late_largest,
        "sd_largest_cluster_fraction_late": late_largest_sd,
        "delta_largest_fraction_late_minus_early": delta_largest,
        "largest_fraction_slope_per_ns": largest_slope,
        "mean_largest_cluster_particles_late": mean_largest_particles,
        "final_largest_cluster_size": final_largest_size,
        "mean_nontrivial_cluster_size_late": late_mean_cluster_size,
        "weight_average_cluster_size_late": late_weight_average,
        "mean_consecutive_bond_retention_late": late_retention,
        "mean_initial_bond_survival_late": late_initial_survival,
        "contact_rearrangement_proxy": (
            1.0 - late_retention
            if late_retention is not None
            else None
        ),
        "mean_msd_late_nm2": late_msd,
        "msd_slope_nm2_per_ns": msd_slope,
        "wrapping_fraction_late": wrapping_fraction,
        "structure_class": structural,
        "growth_class": growth,
        "contact_persistence_class": persistence,
        "phase_status": phase_status,
        "exact_evidence": exact_evidence,
        "run_dir": str(run_dir),
        "cluster_file": str(cluster_file),
        "dynamics_file": (
            str(dynamics_file)
            if dynamics_file.exists()
            else ""
        ),
        "evidence_files": evidence_files,
    })


records.sort(
    key=lambda row: (
        row["concentration_mg_ml"],
        row["seed"],
    )
)

output_csv = OUT / "c5_c10_c20_dynamics_audit.csv"
output_log = OUT / "c5_c10_c20_dynamics_audit_summary.txt"

fields = list(records[0].keys()) if records else []

with output_csv.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)


def fmt(value, digits=6):
    if value is None or value == "":
        return ""

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


lines = [
    (
        "c\tseed\tduration\tclustered\tlargest_early\t"
        "largest_late\tdelta_largest\tlargest_N\t"
        "weight_avg\tretention\trearrangement\t"
        "msd_slope\twrap_fraction\tgrowth\t"
        "structure\tphase_status\texact_evidence"
    )
]

for row in records:
    lines.append(
        "\t".join([
            fmt(row["concentration_mg_ml"], 3),
            str(row["seed"]),
            fmt(row["duration_ns"], 3),
            fmt(row["mean_clustered_fraction_late"]),
            fmt(row["mean_largest_cluster_fraction_early"]),
            fmt(row["mean_largest_cluster_fraction_late"]),
            fmt(
                row[
                    "delta_largest_fraction_late_minus_early"
                ]
            ),
            fmt(
                row["mean_largest_cluster_particles_late"],
                2,
            ),
            fmt(row["weight_average_cluster_size_late"], 3),
            fmt(
                row[
                    "mean_consecutive_bond_retention_late"
                ]
            ),
            fmt(row["contact_rearrangement_proxy"]),
            fmt(row["msd_slope_nm2_per_ns"]),
            fmt(row["wrapping_fraction_late"]),
            row["growth_class"],
            row["structure_class"],
            row["phase_status"],
            row["exact_evidence"],
        ])
    )

summary = "\n".join(lines) + "\n"

output_log.write_text(summary, encoding="utf-8")

print(summary, end="")
print("Saved CSV:", output_csv)
print("Saved summary:", output_log)

# 同时列出可能存在的轨迹文件，便于下一步做真实粒子交换分析
trajectory_extensions = {
    ".gsd",
    ".dcd",
    ".xtc",
    ".trr",
    ".xyz",
    ".lammpstrj",
}

trajectory_lines = []

for row in records:
    run_dir = Path(row["run_dir"])

    for path in run_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in trajectory_extensions:
            trajectory_lines.append(
                f'{row["run_id"]}\t{path}'
            )

trajectory_file = OUT / "c5_c10_c20_trajectory_files.txt"
trajectory_file.write_text(
    "\n".join(trajectory_lines)
    + ("\n" if trajectory_lines else ""),
    encoding="utf-8",
)

print("Trajectory files found:", len(trajectory_lines))
print("Trajectory list:", trajectory_file)
