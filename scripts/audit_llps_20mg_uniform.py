#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path.cwd()
OUTPUT_ROOT = Path("results/llps_20mg_uniform_audit")

PARTICLES = 500
FRAME_INTERVAL_NS = 0.1

TARGET_CONDITIONS = {
    (4.0, 300.0),
    (9.0, 100.0),
}

TARGET_CONCENTRATIONS = {
    17.5,
    18.75,
    19.375,
    20.0,
}

RUN_PATTERN = re.compile(
    r"pH(?P<ph>\d+(?:p\d+)?)"
    r"_nacl(?P<nacl>\d+(?:p\d+)?)"
    r"_c(?P<conc>\d+(?:p\d+)?)"
    r"_seed(?P<seed>\d+)",
    re.IGNORECASE,
)


def token_to_float(token: str) -> float:
    return float(token.lower().replace("p", "."))


def approximately_equal(a: float, b: float, tolerance: float = 1e-6) -> bool:
    return abs(a - b) <= tolerance


def parse_run_name(run_name: str):
    match = RUN_PATTERN.search(run_name)

    if not match:
        return None

    return {
        "ph": token_to_float(match.group("ph")),
        "nacl_mM": token_to_float(match.group("nacl")),
        "concentration_mg_ml": token_to_float(match.group("conc")),
        "seed": match.group("seed"),
    }


def trajectory_frame_count(trajectory_file: Path) -> int | None:
    try:
        with np.load(trajectory_file, allow_pickle=False) as data:
            preferred_keys = [
                "positions",
                "trajectory_positions",
                "coords",
                "coordinates",
            ]

            for key in preferred_keys:
                if key in data.files:
                    array = data[key]

                    if array.ndim == 3 and array.shape[-1] == 3:
                        return int(array.shape[0])

            for key in data.files:
                array = data[key]

                if array.ndim == 3 and array.shape[-1] == 3:
                    return int(array.shape[0])

    except Exception as exc:
        print(f"WARNING: 无法读取 {trajectory_file}: {exc}")

    return None


def discover_target_runs():
    discovered = []

    for trajectory_file in PROJECT_ROOT.rglob("trajectory_positions.npz"):
        run_dir = trajectory_file.parent
        metadata = parse_run_name(run_dir.name)

        if metadata is None:
            continue

        condition = (
            metadata["ph"],
            metadata["nacl_mM"],
        )

        if condition not in TARGET_CONDITIONS:
            continue

        if not any(
            approximately_equal(
                metadata["concentration_mg_ml"],
                target,
            )
            for target in TARGET_CONCENTRATIONS
        ):
            continue

        frames = trajectory_frame_count(trajectory_file)

        # 当前30 ns生产轨迹为300帧。
        # 排除0.5 ns粗筛及其他长度轨迹。
        if frames != 300:
            continue

        metadata["frames"] = frames
        metadata["run_dir"] = run_dir
        metadata["trajectory_file"] = trajectory_file

        discovered.append(metadata)

    discovered.sort(
        key=lambda item: (
            item["ph"],
            item["nacl_mM"],
            item["concentration_mg_ml"],
            int(item["seed"]),
            str(item["run_dir"]),
        )
    )

    return discovered


def print_discovered_runs(runs):
    print()
    print("=" * 110)
    print("DISCOVERED 30 ns TARGET RUNS")
    print("=" * 110)

    if not runs:
        print("没有找到符合条件的300帧轨迹。")
        return

    print(
        f"{'pH':>5s} "
        f"{'NaCl':>7s} "
        f"{'c(mg/mL)':>10s} "
        f"{'SEED':>10s} "
        f"{'FRAMES':>7s} "
        f"PATH"
    )

    for item in runs:
        print(
            f"{item['ph']:5.1f} "
            f"{item['nacl_mM']:7.0f} "
            f"{item['concentration_mg_ml']:10.3f} "
            f"{item['seed']:>10s} "
            f"{item['frames']:7d} "
            f"{item['run_dir']}"
        )


def run_uniform_analysis(run_dir: Path):
    trajectory_script = Path("scripts/analyze_trajectory.py")
    dynamics_script = Path("scripts/analyze_state_dynamics.py")

    if not trajectory_script.exists():
        raise FileNotFoundError(
            f"缺少分析脚本：{trajectory_script}"
        )

    if not dynamics_script.exists():
        raise FileNotFoundError(
            f"缺少分析脚本：{dynamics_script}"
        )

    print()
    print("=" * 90)
    print("UNIFORM REANALYSIS:", run_dir)
    print("=" * 90)

    subprocess.run(
        [
            sys.executable,
            "-u",
            str(trajectory_script),
            "--input-dir",
            str(run_dir),
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "-u",
            str(dynamics_script),
            "--input-dir",
            str(run_dir),
        ],
        check=True,
    )


def get_number(
    data: dict,
    keys: list[str],
    default: float = float("nan"),
) -> float:
    for key in keys:
        if key not in data:
            continue

        value = data[key]

        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return default


def read_largest_cluster_series(
    series_file: Path,
) -> list[float]:
    values = []

    with series_file.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            fraction_value = row.get(
                "largest_cluster_fraction"
            )
            size_value = row.get(
                "largest_cluster_size"
            )

            if fraction_value not in (None, ""):
                values.append(float(fraction_value))
            elif size_value not in (None, ""):
                values.append(
                    float(size_value) / PARTICLES
                )

    return values


def calculate_late_slope(
    values: list[float],
) -> float:
    if len(values) < 5:
        return float("nan")

    start = int(0.8 * len(values))
    late_y = np.asarray(
        values[start:],
        dtype=float,
    )

    if len(late_y) < 2:
        return float("nan")

    late_x = (
        np.arange(len(late_y), dtype=float)
        * FRAME_INTERVAL_NS
    )

    return float(
        np.polyfit(late_x, late_y, 1)[0]
    )


def classify_state(
    max_fraction: float,
    final_fraction: float,
    mean_fraction: float,
    bond: float,
    retention: float,
    percolation: float,
    slope: float,
) -> str:
    dynamic_support = (
        max_fraction >= 0.14
        and not math.isnan(retention)
        and retention < 0.95
        and (
            (
                not math.isnan(slope)
                and slope >= 0.001
            )
            or final_fraction - mean_fraction >= 0.025
        )
    )

    if dynamic_support:
        return "dynamic_condensation_support"

    if (
        not math.isnan(percolation)
        and percolation > 0.5
    ):
        return "percolated_cluster"

    if (
        not math.isnan(bond)
        and not math.isnan(retention)
        and bond >= 0.80
        and retention >= 0.95
    ):
        return "arrested_aggregation_support"

    return "finite_mobile_cluster"


def collect_metrics(item):
    run_dir = item["run_dir"]

    cluster_file = (
        run_dir
        / "cluster_analysis"
        / "cluster_summary.json"
    )

    dynamics_file = (
        run_dir
        / "dynamics_analysis"
        / "dynamics_summary.json"
    )

    series_file = (
        run_dir
        / "cluster_analysis"
        / "cluster_statistics_by_frame.csv"
    )

    missing_files = [
        path
        for path in [
            cluster_file,
            dynamics_file,
            series_file,
        ]
        if not path.exists()
    ]

    if missing_files:
        print()
        print("SKIP，缺少分析文件：", run_dir)

        for path in missing_files:
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

    clustered_fraction = get_number(
        cluster,
        [
            "mean_clustered_fraction",
            "clustered_fraction_mean",
        ],
    )

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
        [
            "percolation_fraction",
        ],
    )

    final_msd = get_number(
        dynamics,
        [
            "final_corrected_msd",
            "final_corrected_msd_nm2",
            "corrected_msd_final",
        ],
    )

    values = read_largest_cluster_series(
        series_file
    )

    slope = calculate_late_slope(values)

    evidence = classify_state(
        max_fraction=max_fraction,
        final_fraction=final_fraction,
        mean_fraction=mean_fraction,
        bond=bond,
        retention=retention,
        percolation=percolation,
        slope=slope,
    )

    return {
        "ph": item["ph"],
        "nacl_mM": item["nacl_mM"],
        "concentration_mg_ml":
            item["concentration_mg_ml"],
        "seed": item["seed"],
        "frames": item["frames"],
        "mean_clustered_fraction":
            clustered_fraction,
        "mean_largest_fraction":
            mean_fraction,
        "maximum_largest_fraction":
            max_fraction,
        "final_largest_fraction":
            final_fraction,
        "final_minus_mean":
            final_fraction - mean_fraction,
        "final_initial_bond_survival":
            bond,
        "mean_consecutive_retention":
            retention,
        "percolation_fraction":
            percolation,
        "final_corrected_msd_nm2":
            final_msd,
        "late_slope_per_ns":
            slope,
        "evidence":
            evidence,
        "run_directory":
            str(run_dir),
    }


def write_metrics_csv(rows):
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_ROOT
        / "boundary_uniform_metrics.csv"
    )

    columns = [
        "ph",
        "nacl_mM",
        "concentration_mg_ml",
        "seed",
        "frames",
        "mean_clustered_fraction",
        "mean_largest_fraction",
        "maximum_largest_fraction",
        "final_largest_fraction",
        "final_minus_mean",
        "final_initial_bond_survival",
        "mean_consecutive_retention",
        "percolation_fraction",
        "final_corrected_msd_nm2",
        "late_slope_per_ns",
        "evidence",
        "run_directory",
    ]

    with output_file.open(
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

    return output_file


def print_metrics_table(rows):
    print()
    print("=" * 145)
    print("UNIFORM BOUNDARY METRICS")
    print("=" * 145)

    print(
        f"{'pH':>4s} "
        f"{'NaCl':>6s} "
        f"{'c':>8s} "
        f"{'SEED':>9s} "
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
            f"{row['ph']:4.1f} "
            f"{row['nacl_mM']:6.0f} "
            f"{row['concentration_mg_ml']:8.3f} "
            f"{row['seed']:>9s} "
            f"{row['mean_largest_fraction']:7.3f} "
            f"{row['maximum_largest_fraction']:7.3f} "
            f"{row['final_largest_fraction']:7.3f} "
            f"{row['final_minus_mean']:7.3f} "
            f"{row['final_initial_bond_survival']:7.3f} "
            f"{row['mean_consecutive_retention']:7.3f} "
            f"{row['percolation_fraction']:7.3f} "
            f"{row['late_slope_per_ns']:10.5f} "
            f"{row['evidence']}"
        )


def print_twenty_mg_consensus(rows):
    print()
    print("=" * 90)
    print("20 mg/mL MULTI-SEED CONSENSUS")
    print("=" * 90)

    for ph, nacl in sorted(TARGET_CONDITIONS):
        selected = [
            row
            for row in rows
            if approximately_equal(row["ph"], ph)
            and approximately_equal(
                row["nacl_mM"],
                nacl,
            )
            and approximately_equal(
                row["concentration_mg_ml"],
                20.0,
            )
        ]

        print()
        print(
            f"pH {ph:.1f}, NaCl {nacl:.0f} mM:"
        )

        if not selected:
            print("  没有找到20 mg/mL的300帧轨迹")
            continue

        dynamic_count = sum(
            row["evidence"]
            == "dynamic_condensation_support"
            for row in selected
        )

        finite_count = sum(
            row["evidence"]
            == "finite_mobile_cluster"
            for row in selected
        )

        arrested_count = sum(
            row["evidence"]
            == "arrested_aggregation_support"
            for row in selected
        )

        percolated_count = sum(
            row["evidence"]
            == "percolated_cluster"
            for row in selected
        )

        print(f"  总种子数           : {len(selected)}")
        print(f"  动态凝聚支持       : {dynamic_count}")
        print(f"  有限可移动团簇     : {finite_count}")
        print(f"  冻结聚集支持       : {arrested_count}")
        print(f"  贯通团簇           : {percolated_count}")

        if (
            len(selected) >= 2
            and dynamic_count == len(selected)
        ):
            conclusion = (
                "多种子一致支持动态凝聚，"
                "可保留为LLPS支持点"
            )
        elif dynamic_count > 0:
            conclusion = (
                "种子结果不一致，"
                "应标记为边界未决/LLPS候选"
            )
        elif arrested_count > 0:
            conclusion = (
                "不支持液态LLPS，"
                "存在冻结聚集证据"
            )
        else:
            conclusion = (
                "当前统一判据下不支持LLPS，"
                "原标签可能是假阳性"
            )

        print("  统一结论           :", conclusion)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "统一重算20 mg/mL并汇总"
            "17.5–20 mg/mL边界状态"
        )
    )

    parser.add_argument(
        "--discover",
        action="store_true",
        help="只搜索并显示符合条件的30 ns轨迹",
    )

    parser.add_argument(
        "--reanalyze-20",
        action="store_true",
        help="使用当前脚本重新分析全部20 mg/mL轨迹",
    )

    args = parser.parse_args()

    runs = discover_target_runs()
    print_discovered_runs(runs)

    if args.discover:
        return

    twenty_mg_runs = [
        item
        for item in runs
        if approximately_equal(
            item["concentration_mg_ml"],
            20.0,
        )
    ]

    if args.reanalyze_20:
        if not twenty_mg_runs:
            print()
            print(
                "ERROR：没有找到20 mg/mL的"
                "300帧轨迹，未执行重分析。"
            )
            sys.exit(1)

        for item in twenty_mg_runs:
            run_uniform_analysis(
                item["run_dir"]
            )

    rows = []

    for item in runs:
        row = collect_metrics(item)

        if row is not None:
            rows.append(row)

    rows.sort(
        key=lambda row: (
            row["ph"],
            row["nacl_mM"],
            row["concentration_mg_ml"],
            int(row["seed"]),
            row["run_directory"],
        )
    )

    if not rows:
        print()
        print("没有可汇总的分析结果。")
        sys.exit(1)

    output_file = write_metrics_csv(rows)

    print_metrics_table(rows)
    print_twenty_mg_consensus(rows)

    print()
    print("Saved:", output_file)


if __name__ == "__main__":
    main()
