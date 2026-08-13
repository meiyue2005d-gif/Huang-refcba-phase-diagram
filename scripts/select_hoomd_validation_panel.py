#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "results" / "hoomd_coarse_224_0p5ns"

INPUT = SCAN / "hoomd_coarse_224_screening_classified.csv"
OUTPUT = SCAN / "validation_panel_5ns.tsv"

df = pd.read_csv(INPUT)

selected = []


def add_rows(frame: pd.DataFrame, reason: str) -> None:
    existing = {row["state_id"] for row in selected}

    for _, row in frame.iterrows():
        if row["state_id"] in existing:
            continue

        selected.append({
            "state_id": row["state_id"],
            "pH": row["pH"],
            "nacl_mM": row["nacl_mM"],
            "concentration_mg_ml":
                row["concentration_mg_ml"],
            "screening_class":
                row["screening_class"],
            "selection_reason": reason,
            "late_mean_clustered_fraction":
                row["late_mean_clustered_fraction"],
            "late_mean_largest_cluster":
                row["late_mean_largest_cluster"],
            "late_slope_particles_per_ns":
                row[
                    "late_largest_slope_particles_per_ns"
                ],
        })
        existing.add(row["state_id"])


# 1. 所有待复核状态
review = df[
    df["screening_class"] == "review_required"
].sort_values(
    "late_mean_clustered_fraction",
    ascending=False,
)
add_rows(review, "all_review_required")


# 2. 所有尚未做过长时间验证的粗化候选
known_long = {
    "pH9_nacl100_c20_seed20260718",
}

coarsening = df[
    (
        df["screening_class"]
        == "dynamic_coarsening_candidate"
    )
    & (~df["state_id"].isin(known_long))
].sort_values(
    "late_largest_slope_particles_per_ns",
    ascending=False,
)
add_rows(coarsening, "unresolved_coarsening_candidate")


# 3. 选择4个有限团簇候选，尽量覆盖不同pH和盐度
finite = df[
    df["screening_class"] == "finite_cluster_candidate"
].copy()

finite = finite[
    finite["state_id"]
    != "pH9_nacl100_c10_seed20260718"
]

finite["selection_score"] = (
    finite["late_mean_clustered_fraction"]
    + finite["late_mean_largest_cluster"] / 20.0
    - finite[
        "late_largest_slope_particles_per_ns"
    ].abs() / 100.0
)

finite_selected = []

for _, row in finite.sort_values(
    "selection_score",
    ascending=False,
).iterrows():
    # 避免全部选到近似相同的pH/盐度
    too_close = any(
        abs(row["pH"] - old["pH"]) < 0.3
        and row["nacl_mM"] == old["nacl_mM"]
        for old in finite_selected
    )

    if too_close:
        continue

    finite_selected.append(row)

    if len(finite_selected) == 4:
        break

add_rows(
    pd.DataFrame(finite_selected),
    "diverse_finite_cluster_candidate",
)


# 4. 选择4个强聚集代表点
strong = df[
    df["screening_class"]
    == "strong_aggregation_candidate"
].copy()

known_strong = {
    "pH5p5_nacl100_c20_seed20260718",
    "pH9_nacl500_c20_seed20260718",
}

strong = strong[
    ~strong["state_id"].isin(known_strong)
]

target_groups = [
    ("acidic_high_salt",
     (strong["pH"] <= 4.5)
     & (strong["nacl_mM"] >= 300)),
    ("near_pI_low_salt",
     (strong["pH"].between(4.7, 5.1))
     & (strong["nacl_mM"] <= 100)),
    ("neutral_high_salt",
     (strong["pH"].between(5.5, 7.5))
     & (strong["nacl_mM"] >= 300)),
    ("low_salt_strong",
     strong["nacl_mM"] == 0),
]

for name, mask in target_groups:
    candidates = strong[mask].sort_values(
        [
            "late_mean_clustered_fraction",
            "late_largest_slope_particles_per_ns",
        ],
        ascending=False,
    )

    if not candidates.empty:
        add_rows(
            candidates.head(1),
            f"strong_representative_{name}",
        )


result = pd.DataFrame(selected)
result.to_csv(OUTPUT, sep="\t", index=False)

print("=" * 90)
print("HOOMD 5 ns VALIDATION PANEL")
print("=" * 90)
print("Selected states:", len(result))

print(
    result[
        [
            "state_id",
            "screening_class",
            "selection_reason",
            "late_mean_clustered_fraction",
            "late_mean_largest_cluster",
            "late_slope_particles_per_ns",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print("\nClass counts:")
print(result["screening_class"].value_counts())

print("\nSaved:", OUTPUT)
print("SELECT_HOOMD_VALIDATION_PANEL: PASS")
