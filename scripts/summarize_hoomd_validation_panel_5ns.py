#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SCAN_ROOT = (
    ROOT
    / "results"
    / "hoomd_coarse_224_0p5ns"
)

RESULT_ROOT = (
    ROOT
    / "results"
    / "hoomd_validation_panel_5ns"
)

PANEL = SCAN_ROOT / "validation_panel_5ns.tsv"

COARSE_METRICS = (
    SCAN_ROOT
    / "hoomd_coarse_224_metrics.csv"
)

OUTPUT = (
    RESULT_ROOT
    / "validation_panel_5ns_summary.csv"
)

SHORTLIST_OUTPUT = (
    RESULT_ROOT
    / "llps_candidate_shortlist.csv"
)


def linear_slope_per_ns(
    frame: pd.DataFrame,
    column: str,
) -> float:
    if len(frame) < 2:
        return float("nan")

    x = frame["time_ps"].to_numpy(dtype=float)
    y = frame[column].to_numpy(dtype=float)

    if np.ptp(x) <= 0:
        return float("nan")

    return float(
        np.polyfit(x, y, 1)[0] * 1000.0
    )


def classify_5ns(row: pd.Series) -> str:
    clustered = float(
        row["late_mean_clustered_fraction"]
    )
    largest = float(
        row["late_mean_largest_cluster"]
    )
    slope = float(
        row["late_largest_slope_particles_per_ns"]
    )
    size_range = float(
        row["late_largest_range"]
    )
    growth = float(
        row["largest_growth_particles"]
    )

    # 可溶状态
    if (
        clustered <= 0.05
        and largest <= 2.0
    ):
        return "soluble"

    # 高聚集比例且晚期进入平台
    if (
        clustered >= 0.85
        and abs(slope) <= 0.5
        and size_range <= 3
    ):
        return "arrested_aggregation"

    # 最大团簇仍持续显著增长
    if (
        slope >= 1.0
        and growth >= 5
    ):
        return "dynamic_coarsening"

    # 中等团簇比例且最大团簇平台化
    if (
        0.15 <= clustered < 0.85
        and largest <= 30
        and abs(slope) <= 0.5
        and size_range <= 5
    ):
        return "finite_cluster_fluid"

    # 较弱的低聚状态
    if (
        clustered < 0.35
        and largest <= 8
        and abs(slope) < 1.0
    ):
        return "weak_oligomer"

    return "review_required"


panel = pd.read_csv(PANEL, sep="\t")
coarse = pd.read_csv(COARSE_METRICS)

rows = []
missing = []

for _, state in panel.iterrows():
    state_id = str(state["state_id"])
    directory = RESULT_ROOT / state_id

    summary_path = (
        directory
        / "cluster_analysis"
        / "cluster_summary.json"
    )

    frame_path = (
        directory
        / "cluster_analysis"
        / "cluster_statistics_by_frame.csv"
    )

    thermo_path = (
        directory
        / "production_thermo.csv"
    )

    required = [
        summary_path,
        frame_path,
        thermo_path,
    ]

    if any(not path.exists() for path in required):
        missing.append(state_id)
        continue

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    frames = pd.read_csv(frame_path)
    thermo = pd.read_csv(thermo_path)

    n_frames = len(frames)

    early = frames.iloc[
        :max(1, int(0.2 * n_frames))
    ].copy()

    late = frames.iloc[
        int(0.8 * n_frames):
    ].copy()

    largest = frames[
        "largest_cluster_size"
    ]

    late_largest = late[
        "largest_cluster_size"
    ]

    late_clustered = late[
        "clustered_fraction"
    ]

    first_largest = int(largest.iloc[0])
    final_largest = int(largest.iloc[-1])
    maximum_largest = int(largest.max())

    coarse_row = coarse[
        coarse["state_id"] == state_id
    ]

    if len(coarse_row) == 1:
        coarse_row = coarse_row.iloc[0]

        coarse_late_clustered = float(
            coarse_row[
                "late_mean_clustered_fraction"
            ]
        )

        coarse_late_largest = float(
            coarse_row[
                "late_mean_largest_cluster"
            ]
        )

        coarse_slope = float(
            coarse_row[
                "late_largest_slope_particles_per_ns"
            ]
        )
    else:
        coarse_late_clustered = float("nan")
        coarse_late_largest = float("nan")
        coarse_slope = float("nan")

    row = {
        "state_id": state_id,
        "pH": float(state["pH"]),
        "nacl_mM": float(state["nacl_mM"]),
        "concentration_mg_ml": float(
            state["concentration_mg_ml"]
        ),
        "screening_class_0p5ns":
            state["screening_class"],
        "selection_reason":
            state["selection_reason"],

        "frames_5ns": n_frames,

        "mean_monomer_fraction_5ns": float(
            summary["mean_monomer_fraction"]
        ),

        "mean_clustered_fraction_5ns": float(
            summary["mean_clustered_fraction"]
        ),

        "first_largest_cluster": first_largest,

        "mean_largest_cluster_5ns": float(
            summary[
                "mean_largest_cluster_size"
            ]
        ),

        "final_largest_cluster": final_largest,

        "maximum_largest_cluster":
            maximum_largest,

        "late_mean_clustered_fraction": float(
            late_clustered.mean()
        ),

        "late_std_clustered_fraction": float(
            late_clustered.std()
        ),

        "late_min_clustered_fraction": float(
            late_clustered.min()
        ),

        "late_max_clustered_fraction": float(
            late_clustered.max()
        ),

        "late_mean_largest_cluster": float(
            late_largest.mean()
        ),

        "late_std_largest_cluster": float(
            late_largest.std()
        ),

        "late_min_largest_cluster": int(
            late_largest.min()
        ),

        "late_max_largest_cluster": int(
            late_largest.max()
        ),

        "late_largest_range": int(
            late_largest.max()
            - late_largest.min()
        ),

        "late_largest_slope_particles_per_ns":
            linear_slope_per_ns(
                late,
                "largest_cluster_size",
            ),

        "late_clustered_slope_per_ns":
            linear_slope_per_ns(
                late,
                "clustered_fraction",
            ),

        "largest_growth_particles":
            final_largest - first_largest,

        "late_largest_cluster_fraction":
            float(
                late_largest.mean() / 500.0
            ),

        "late_free_particle_fraction":
            float(
                1.0 - late_clustered.mean()
            ),

        "temperature_mean_Tstar": float(
            thermo[
                "kinetic_temperature_kBT"
            ].mean()
        ),

        "temperature_std_Tstar": float(
            thermo[
                "kinetic_temperature_kBT"
            ].std()
        ),

        "potential_energy_change_kBT": float(
            thermo[
                "potential_energy_kBT"
            ].iloc[-1]
            - thermo[
                "potential_energy_kBT"
            ].iloc[0]
        ),

        "coarse_late_clustered_fraction":
            coarse_late_clustered,

        "coarse_late_largest_cluster":
            coarse_late_largest,

        "coarse_late_slope_particles_per_ns":
            coarse_slope,
    }

    rows.append(row)


result = pd.DataFrame(rows)

result["classification_5ns"] = result.apply(
    classify_5ns,
    axis=1,
)


# LLPS只做候选筛选，不代表确认
result["llps_candidate_score"] = 0.0

result["llps_candidate_score"] += np.where(
    result["classification_5ns"]
    == "dynamic_coarsening",
    4.0,
    0.0,
)

result["llps_candidate_score"] += np.where(
    result[
        "late_mean_clustered_fraction"
    ].between(0.20, 0.85),
    2.0,
    0.0,
)

result["llps_candidate_score"] += np.where(
    result[
        "late_free_particle_fraction"
    ].between(0.15, 0.80),
    2.0,
    0.0,
)

result["llps_candidate_score"] += np.where(
    result[
        "late_largest_cluster_fraction"
    ].between(0.02, 0.30),
    1.0,
    0.0,
)

result["llps_candidate_score"] += np.where(
    result[
        "late_largest_slope_particles_per_ns"
    ] >= 1.0,
    1.0,
    0.0,
)

# 接近完全冻结的聚集状态降权
result["llps_candidate_score"] -= np.where(
    result[
        "late_mean_clustered_fraction"
    ] >= 0.90,
    4.0,
    0.0,
)

result = result.sort_values(
    [
        "classification_5ns",
        "pH",
        "nacl_mM",
        "concentration_mg_ml",
    ]
).reset_index(drop=True)

result.to_csv(OUTPUT, index=False)


shortlist = (
    result[
        (
            result["llps_candidate_score"] >= 7
        )
        & (
            result["classification_5ns"]
            == "dynamic_coarsening"
        )
    ]
    .sort_values(
        [
            "llps_candidate_score",
            "late_largest_slope_particles_per_ns",
        ],
        ascending=False,
    )
    .head(4)
)

shortlist.to_csv(
    SHORTLIST_OUTPUT,
    index=False,
)


print("=" * 110)
print("HOOMD 5 ns VALIDATION PANEL SUMMARY")
print("=" * 110)

print("Panel states :", len(panel))
print("Summarized   :", len(result))
print("Missing      :", len(missing))
print("Output       :", OUTPUT)

if missing:
    print("\nMissing states:")
    for state_id in missing:
        print(" ", state_id)

print("\n5 ns classification counts:")
print(
    result[
        "classification_5ns"
    ].value_counts().to_string()
)

columns = [
    "state_id",
    "screening_class_0p5ns",
    "classification_5ns",
    "late_mean_clustered_fraction",
    "late_mean_largest_cluster",
    "late_min_largest_cluster",
    "late_max_largest_cluster",
    "late_largest_slope_particles_per_ns",
    "late_free_particle_fraction",
    "llps_candidate_score",
]

print("\nFull 5 ns classification:")
print(
    result[columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print("\nLLPS candidate shortlist:")
if shortlist.empty:
    print(
        "No state passed the provisional LLPS "
        "candidate filter."
    )
else:
    print(
        shortlist[
            [
                "state_id",
                "pH",
                "nacl_mM",
                "concentration_mg_ml",
                "classification_5ns",
                "late_mean_clustered_fraction",
                "late_mean_largest_cluster",
                "late_largest_slope_particles_per_ns",
                "late_free_particle_fraction",
                "llps_candidate_score",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

print("\nSaved:")
print(" ", OUTPUT)
print(" ", SHORTLIST_OUTPUT)

print(
    "\nSUMMARIZE_HOOMD_VALIDATION_PANEL_5NS: PASS"
)
