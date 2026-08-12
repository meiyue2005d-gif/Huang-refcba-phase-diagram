#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOT = ROOT / "results" / "hoomd_coarse_224_0p5ns"

INPUT = SCAN_ROOT / "hoomd_coarse_224_metrics.csv"
OUTPUT = SCAN_ROOT / "hoomd_coarse_224_screening_classified.csv"
COUNT_OUTPUT = SCAN_ROOT / "screening_class_counts.csv"
BOUNDARY_OUTPUT = SCAN_ROOT / "screening_boundary_transitions.csv"


CLASS_ORDER = [
    "soluble_screen",
    "weak_oligomer_screen",
    "finite_cluster_candidate",
    "dynamic_coarsening_candidate",
    "strong_aggregation_candidate",
    "review_required",
]

CLASS_LABELS = {
    "soluble_screen": "Soluble screen",
    "weak_oligomer_screen": "Weak oligomer",
    "finite_cluster_candidate": "Finite-cluster candidate",
    "dynamic_coarsening_candidate": "Coarsening candidate",
    "strong_aggregation_candidate": "Strong aggregation",
    "review_required": "Review required",
}

SHORT_LABELS = {
    "soluble_screen": "S",
    "weak_oligomer_screen": "O",
    "finite_cluster_candidate": "F",
    "dynamic_coarsening_candidate": "C",
    "strong_aggregation_candidate": "A",
    "review_required": "R",
}


def classify(row: pd.Series) -> str:
    clustered = float(row["late_mean_clustered_fraction"])
    largest = float(row["late_mean_largest_cluster"])
    final_largest = float(row["final_largest_cluster"])
    slope = float(
        row["late_largest_slope_particles_per_ns"]
    )
    growth = float(row["largest_growth_particles"])

    # 非常保守的可溶筛选标准
    if (
        clustered <= 0.05
        and largest <= 2.0
        and final_largest <= 2
    ):
        return "soluble_screen"

    # 高聚集比例优先标为强聚集候选
    if clustered >= 0.70:
        return "strong_aggregation_candidate"

    # 0.5 ns晚期只有约100 ps，因此斜率仅用于筛选
    if (
        clustered >= 0.10
        and slope >= 10.0
        and growth >= 3
    ):
        return "dynamic_coarsening_candidate"

    if clustered <= 0.30 and largest <= 4.5:
        return "weak_oligomer_screen"

    if (
        0.20 <= clustered < 0.70
        and largest <= 15
        and abs(slope) < 10.0
    ):
        return "finite_cluster_candidate"

    return "review_required"


df = pd.read_csv(INPUT)

df["screening_class"] = df.apply(classify, axis=1)
df["class_code"] = df["screening_class"].map(
    {name: i for i, name in enumerate(CLASS_ORDER)}
)

df.to_csv(OUTPUT, index=False)

counts = (
    df["screening_class"]
    .value_counts()
    .reindex(CLASS_ORDER, fill_value=0)
    .rename_axis("screening_class")
    .reset_index(name="count")
)
counts["fraction"] = counts["count"] / len(df)
counts.to_csv(COUNT_OUTPUT, index=False)


# 寻找同一pH和盐浓度下，沿浓度方向发生类别变化的位置
boundary_rows = []

for (ph, salt), group in df.groupby(["pH", "nacl_mM"]):
    group = group.sort_values("concentration_mg_ml").reset_index(
        drop=True
    )

    for index in range(1, len(group)):
        low = group.iloc[index - 1]
        high = group.iloc[index]

        if low["screening_class"] == high["screening_class"]:
            continue

        boundary_rows.append(
            {
                "pH": ph,
                "nacl_mM": salt,
                "low_concentration_mg_ml":
                    low["concentration_mg_ml"],
                "high_concentration_mg_ml":
                    high["concentration_mg_ml"],
                "low_class": low["screening_class"],
                "high_class": high["screening_class"],
                "low_state_id": low["state_id"],
                "high_state_id": high["state_id"],
                "low_clustered_fraction":
                    low["late_mean_clustered_fraction"],
                "high_clustered_fraction":
                    high["late_mean_clustered_fraction"],
                "low_largest_cluster":
                    low["late_mean_largest_cluster"],
                "high_largest_cluster":
                    high["late_mean_largest_cluster"],
                "low_slope":
                    low[
                        "late_largest_slope_particles_per_ns"
                    ],
                "high_slope":
                    high[
                        "late_largest_slope_particles_per_ns"
                    ],
            }
        )

boundaries = pd.DataFrame(boundary_rows)
boundaries.to_csv(BOUNDARY_OUTPUT, index=False)


# 每个盐浓度单独输出一张行为图
ph_values = sorted(df["pH"].unique())
concentrations = sorted(df["concentration_mg_ml"].unique())
salt_values = sorted(df["nacl_mM"].unique())

cmap = ListedColormap([
    "#d9edf7",
    "#c7e9c0",
    "#fee391",
    "#fdae6b",
    "#ef3b2c",
    "#bdbdbd",
])

legend_handles = [
    Patch(
        facecolor=cmap(index),
        edgecolor="black",
        label=f"{SHORT_LABELS[name]}: {CLASS_LABELS[name]}",
    )
    for index, name in enumerate(CLASS_ORDER)
]

for salt in salt_values:
    subset = df[df["nacl_mM"] == salt]

    matrix = (
        subset.pivot(
            index="concentration_mg_ml",
            columns="pH",
            values="class_code",
        )
        .reindex(index=concentrations, columns=ph_values)
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    image = ax.imshow(
        matrix.to_numpy(),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=-0.5,
        vmax=len(CLASS_ORDER) - 0.5,
        cmap=cmap,
    )

    ax.set_xticks(range(len(ph_values)))
    ax.set_xticklabels([f"{value:g}" for value in ph_values])

    ax.set_yticks(range(len(concentrations)))
    ax.set_yticklabels(
        [f"{value:g}" for value in concentrations]
    )

    ax.set_xlabel("pH")
    ax.set_ylabel("Concentration (mg/mL)")
    ax.set_title(
        f"HOOMD 0.5 ns assembly screening, NaCl = {salt:g} mM"
    )

    for row_index, concentration in enumerate(concentrations):
        for column_index, ph in enumerate(ph_values):
            value = matrix.loc[concentration, ph]

            if pd.isna(value):
                text = "?"
            else:
                class_name = CLASS_ORDER[int(value)]
                text = SHORT_LABELS[class_name]

            ax.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    fig.tight_layout()

    output_figure = (
        SCAN_ROOT
        / f"screening_behavior_map_nacl{salt:g}.png"
    )
    fig.savefig(output_figure, dpi=250, bbox_inches="tight")
    plt.close(fig)


print("=" * 90)
print("HOOMD COARSE 224 SCREENING CLASSIFICATION")
print("=" * 90)
print("\nClass counts:")
print(
    counts.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print("\nClass counts by concentration:")
table = pd.crosstab(
    df["concentration_mg_ml"],
    df["screening_class"],
).reindex(columns=CLASS_ORDER, fill_value=0)

print(table.to_string())

print("\nBoundary transitions:", len(boundaries))

print("\nTop dynamic-coarsening candidates:")
coarsening = (
    df[
        df["screening_class"]
        == "dynamic_coarsening_candidate"
    ]
    .sort_values(
        [
            "late_largest_slope_particles_per_ns",
            "late_mean_clustered_fraction",
        ],
        ascending=False,
    )
)

print(
    coarsening[
        [
            "state_id",
            "pH",
            "nacl_mM",
            "concentration_mg_ml",
            "late_mean_clustered_fraction",
            "late_mean_largest_cluster",
            "final_largest_cluster",
            "late_largest_slope_particles_per_ns",
        ]
    ]
    .head(20)
    .to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print("\nStrong-aggregation candidates:")
strong = (
    df[
        df["screening_class"]
        == "strong_aggregation_candidate"
    ]
    .sort_values(
        [
            "late_mean_clustered_fraction",
            "final_largest_cluster",
        ],
        ascending=False,
    )
)

print(
    strong[
        [
            "state_id",
            "pH",
            "nacl_mM",
            "concentration_mg_ml",
            "late_mean_clustered_fraction",
            "late_mean_largest_cluster",
            "final_largest_cluster",
            "late_largest_slope_particles_per_ns",
        ]
    ]
    .head(20)
    .to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print("\nGenerated:")
print(" ", OUTPUT)
print(" ", COUNT_OUTPUT)
print(" ", BOUNDARY_OUTPUT)

for salt in salt_values:
    print(
        " ",
        SCAN_ROOT
        / f"screening_behavior_map_nacl{salt:g}.png",
    )

print("\nCLASSIFY_HOOMD_COARSE_224: PASS")
