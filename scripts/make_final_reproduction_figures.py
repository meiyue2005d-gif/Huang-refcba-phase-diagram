#!/usr/bin/env python3

from __future__ import annotations

from math import ceil
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT
    / "results"
    / "final_reproduction"
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LJ_FILE = (
    ROOT
    / "results"
    / "lj_nist_coexistence_scan"
    / "lj_nist_coexistence_results.csv"
)

POTENTIAL_FILE = (
    ROOT
    / "results"
    / "liquid_theory_validation"
    / "huang_a1_potential_landmarks.csv"
)

A2_FILE = (
    ROOT
    / "results"
    / "liquid_theory_validation"
    / "huang_a2_sign_split_0p436mgml.csv"
)

STATE_FILE = (
    ROOT
    / "results"
    / "combined_fast_scan_summary.csv"
)

VALIDATION_FILE = (
    ROOT
    / "results"
    / "hoomd_validation_panel_5ns"
    / "validation_panel_5ns_summary.csv"
)

MANIFEST_FILE = (
    ROOT
    / "results"
    / "final_reproduction"
    / "final_figure_manifest.csv"
)

README_FILE = (
    ROOT
    / "results"
    / "final_reproduction"
    / "FINAL_README.md"
)

AGGREGATED_STATE_FILE = (
    ROOT
    / "results"
    / "final_reproduction"
    / "refcba_operational_state_map_aggregated.csv"
)

PHASE_COUNT_FILE = (
    ROOT
    / "results"
    / "final_reproduction"
    / "refcba_operational_phase_counts.csv"
)

VALIDATION_COUNT_FILE = (
    ROOT
    / "results"
    / "final_reproduction"
    / "validation_panel_5ns_classification_counts.csv"
)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input does not exist: {path}"
        )


for required in [
    LJ_FILE,
    POTENTIAL_FILE,
    A2_FILE,
    STATE_FILE,
    VALIDATION_FILE,
]:
    require_file(required)


def save_figure(
    figure: plt.Figure,
    stem: str,
) -> tuple[Path, Path]:
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(figure)

    return png_path, pdf_path


def phase_rank(
    phase: str,
) -> tuple[int, str]:
    text = str(phase).lower()

    rules = [
        (0, ["soluble", "dispersed"]),
        (1, ["weak", "oligomer"]),
        (2, ["finite", "cluster"]),
        (3, ["dynamic", "coarsening", "mobile"]),
        (4, ["strong", "aggregation"]),
        (5, ["arrested", "aggregate"]),
    ]

    for rank, terms in rules:
        if any(term in text for term in terms):
            return rank, text

    return 99, text


# ---------------------------------------------------------------------------
# Figure 1: LJ–NIST coexistence validation
# ---------------------------------------------------------------------------

lj = (
    pd.read_csv(LJ_FILE)
    .sort_values("temperature_reduced")
    .reset_index(drop=True)
)

figure, axis = plt.subplots(
    figsize=(8.0, 5.5)
)

axis.semilogy(
    lj["temperature_reduced"],
    lj["reference_vapor_density"],
    marker="o",
    linestyle="-",
    label="NIST vapor",
)

axis.semilogy(
    lj["temperature_reduced"],
    lj["theory_vapor_density"],
    marker="s",
    linestyle="--",
    label="Theory vapor",
)

axis.semilogy(
    lj["temperature_reduced"],
    lj["reference_liquid_density"],
    marker="o",
    linestyle="-",
    label="NIST liquid",
)

axis.semilogy(
    lj["temperature_reduced"],
    lj["theory_liquid_density"],
    marker="s",
    linestyle="--",
    label="Theory liquid",
)

axis.set_xlabel(
    r"Reduced temperature, $T^*$"
)

axis.set_ylabel(
    r"Reduced coexistence density, $\rho\sigma^3$"
)

axis.set_title(
    "Lennard–Jones coexistence validation against NIST"
)

axis.grid(
    True,
    which="both",
    alpha=0.25,
)

axis.legend(
    frameon=False,
)

maximum_liquid_error = float(
    lj["liquid_relative_error"].max()
    * 100.0
)

maximum_pressure_residual = float(
    lj["pressure_residual"].abs().max()
)

maximum_mu_residual = float(
    lj[
        "chemical_potential_residual"
    ].abs().max()
)

annotation = (
    f"Maximum liquid-density relative error: "
    f"{maximum_liquid_error:.2f}%\n"
    f"max |ΔP| = "
    f"{maximum_pressure_residual:.2e}\n"
    f"max |Δμ| = "
    f"{maximum_mu_residual:.2e}"
)

axis.text(
    0.03,
    0.04,
    annotation,
    transform=axis.transAxes,
    va="bottom",
    ha="left",
)

figure1_png, figure1_pdf = save_figure(
    figure,
    "figure1_lj_nist_validation",
)


# ---------------------------------------------------------------------------
# Figure 2: Huang potential landmarks
# ---------------------------------------------------------------------------

potential = (
    pd.read_csv(POTENTIAL_FILE)
    .sort_values("pH")
    .reset_index(drop=True)
)

figure, axis = plt.subplots(
    figsize=(8.0, 5.5)
)

axis.plot(
    potential["pH"],
    potential["well_depth_kBT"],
    marker="o",
    linewidth=2,
    label="Attractive-well depth",
)

axis.plot(
    potential["pH"],
    potential["barrier_height_kBT"],
    marker="s",
    linewidth=2,
    label="Repulsive-barrier height",
)

axis.set_xlabel("pH")
axis.set_ylabel(r"Potential landmark, $k_{\mathrm{B}}T$")

axis.set_title(
    "Reconstructed Huang A1 potential landmarks"
)

axis.grid(
    True,
    alpha=0.25,
)

axis.legend(
    frameon=False,
)

axis.text(
    0.03,
    0.96,
    (
        "Increasing pH deepens the attractive well\n"
        "and suppresses the electrostatic barrier."
    ),
    transform=axis.transAxes,
    va="top",
    ha="left",
)

figure2_png, figure2_pdf = save_figure(
    figure,
    "figure2_huang_potential_landmarks",
)


# ---------------------------------------------------------------------------
# Figure 3: second-order sign-split diagnosis
# ---------------------------------------------------------------------------

a2 = (
    pd.read_csv(A2_FILE)
    .sort_values("pH")
    .reset_index(drop=True)
)

figure, axes = plt.subplots(
    1,
    2,
    figsize=(12.5, 5.2),
)

fraction_axis = axes[0]

fraction_axis.plot(
    a2["pH"],
    100.0
    * a2["repulsive_fraction_of_I2"],
    marker="o",
    linewidth=2,
    label="Repulsive contribution",
)

fraction_axis.plot(
    a2["pH"],
    100.0
    * a2["attractive_fraction_of_I2"],
    marker="s",
    linewidth=2,
    label="Attractive contribution",
)

fraction_axis.set_xlabel("pH")
fraction_axis.set_ylabel(
    r"Fraction of second moment $I_2$ (%)"
)

fraction_axis.set_ylim(-2.0, 102.0)

fraction_axis.set_title(
    "Composition of the squared-potential moment"
)

fraction_axis.grid(
    True,
    alpha=0.25,
)

fraction_axis.legend(
    frameon=False,
)

pH45_row = (
    a2.loc[
        np.isclose(a2["pH"], 4.5)
    ]
    .iloc[0]
)

fraction_axis.annotate(
    (
        f"pH 4.5:\n"
        f"{100.0 * pH45_row['repulsive_fraction_of_I2']:.1f}% "
        f"repulsive"
    ),
    xy=(
        float(pH45_row["pH"]),
        100.0
        * float(
            pH45_row[
                "repulsive_fraction_of_I2"
            ]
        ),
    ),
    xytext=(5.0, 78.0),
    arrowprops={
        "arrowstyle": "->",
    },
)

free_energy_axis = axes[1]

free_energy_axis.plot(
    a2["pH"],
    a2["beta_a2_full"],
    marker="o",
    linewidth=2,
    label=r"Full $a_2$",
)

free_energy_axis.plot(
    a2["pH"],
    a2["beta_a2_attractive_only"],
    marker="s",
    linewidth=2,
    label=r"Attractive-only $a_2$",
)

free_energy_axis.plot(
    a2["pH"],
    a2["beta_a2_repulsive_only"],
    marker="^",
    linewidth=2,
    label=r"Repulsive-only $a_2$",
)

free_energy_axis.axhline(
    0.0,
    linewidth=1,
)

free_energy_axis.set_xlabel("pH")
free_energy_axis.set_ylabel(
    r"Second-order free-energy correction, $\beta a_2$"
)

free_energy_axis.set_title(
    "Negative correction generated by squared repulsion"
)

free_energy_axis.grid(
    True,
    alpha=0.25,
)

free_energy_axis.legend(
    frameon=False,
)

figure.suptitle(
    (
        "Why the published full-potential second-order reconstruction "
        "reverses the pH ordering"
    )
)

figure.tight_layout()

figure3_png, figure3_pdf = save_figure(
    figure,
    "figure3_huang_a2_sign_split",
)


# ---------------------------------------------------------------------------
# Figure 4: refCBA operational-state map
# ---------------------------------------------------------------------------

states = pd.read_csv(STATE_FILE)

required_state_columns = {
    "pH",
    "NaCl_mM",
    "concentration_mg_ml",
    "coarse_phase",
}

missing_columns = (
    required_state_columns
    - set(states.columns)
)

if missing_columns:
    raise KeyError(
        "State table is missing columns: "
        + ", ".join(sorted(missing_columns))
    )

group_keys = [
    "pH",
    "NaCl_mM",
    "concentration_mg_ml",
]

aggregated_rows: list[dict[str, object]] = []

for key, group in states.groupby(
    group_keys,
    dropna=False,
):
    phase_counts = (
        group["coarse_phase"]
        .astype(str)
        .value_counts()
    )

    modal_phase = str(
        phase_counts.index[0]
    )

    agreement = float(
        phase_counts.iloc[0]
        / phase_counts.sum()
    )

    aggregated_rows.append(
        {
            "pH": float(key[0]),
            "NaCl_mM": float(key[1]),
            "concentration_mg_ml": float(
                key[2]
            ),
            "coarse_phase_mode": modal_phase,
            "classification_agreement": agreement,
            "number_of_observations": int(
                len(group)
            ),
            "mean_clustered_fraction": float(
                group[
                    "mean_clustered_fraction"
                ].median()
            ),
            "mean_largest_cluster_fraction": float(
                group[
                    "mean_largest_cluster_fraction"
                ].median()
            ),
            "percolation_fraction": float(
                group[
                    "percolation_fraction"
                ].median()
            ),
        }
    )

aggregated = (
    pd.DataFrame(aggregated_rows)
    .sort_values(group_keys)
    .reset_index(drop=True)
)

aggregated.to_csv(
    AGGREGATED_STATE_FILE,
    index=False,
)

phase_counts = (
    aggregated[
        "coarse_phase_mode"
    ]
    .value_counts()
    .rename_axis("operational_phase")
    .reset_index(name="state_count")
)

phase_counts.to_csv(
    PHASE_COUNT_FILE,
    index=False,
)

validation = pd.read_csv(
    VALIDATION_FILE
)

validation_counts = (
    validation[
        "classification_5ns"
    ]
    .value_counts()
    .rename_axis("classification_5ns")
    .reset_index(name="state_count")
)

validation_counts.to_csv(
    VALIDATION_COUNT_FILE,
    index=False,
)

salts = sorted(
    aggregated["NaCl_mM"].unique()
)

phase_categories = sorted(
    aggregated[
        "coarse_phase_mode"
    ].unique(),
    key=phase_rank,
)

default_colors = (
    plt.rcParams[
        "axes.prop_cycle"
    ]
    .by_key()["color"]
)

markers = [
    "o",
    "s",
    "^",
    "D",
    "P",
    "X",
    "v",
    "<",
    ">",
]

phase_style = {}

for index, phase in enumerate(
    phase_categories
):
    phase_style[phase] = {
        "color": default_colors[
            index % len(default_colors)
        ],
        "marker": markers[
            index % len(markers)
        ],
    }

number_of_columns = 2
number_of_rows = ceil(
    len(salts)
    / number_of_columns
)

figure, axes = plt.subplots(
    number_of_rows,
    number_of_columns,
    figsize=(
        12.5,
        4.8 * number_of_rows,
    ),
    squeeze=False,
)

legend_handles = {}

for panel_index, salt in enumerate(
    salts
):
    row_index = (
        panel_index
        // number_of_columns
    )

    column_index = (
        panel_index
        % number_of_columns
    )

    axis = axes[
        row_index,
        column_index,
    ]

    subset = aggregated.loc[
        np.isclose(
            aggregated["NaCl_mM"],
            salt,
        )
    ]

    for phase in phase_categories:
        phase_subset = subset.loc[
            subset[
                "coarse_phase_mode"
            ]
            == phase
        ]

        if phase_subset.empty:
            continue

        style = phase_style[phase]

        sizes = (
            35.0
            + 105.0
            * phase_subset[
                "classification_agreement"
            ].to_numpy()
        )

        handle = axis.scatter(
            phase_subset["pH"],
            phase_subset[
                "concentration_mg_ml"
            ],
            s=sizes,
            marker=style["marker"],
            color=style["color"],
            alpha=0.85,
            label=phase,
        )

        legend_handles.setdefault(
            phase,
            handle,
        )

        disputed = phase_subset.loc[
            phase_subset[
                "classification_agreement"
            ]
            < 0.999
        ]

        if not disputed.empty:
            axis.scatter(
                disputed["pH"],
                disputed[
                    "concentration_mg_ml"
                ],
                marker="x",
                s=65,
                linewidths=1.2,
            )

    axis.set_yscale("log")

    axis.set_xlabel("pH")

    axis.set_ylabel(
        "Concentration (mg/mL)"
    )

    salt_label = (
        f"{int(round(salt))}"
        if np.isclose(
            salt,
            round(salt),
        )
        else f"{salt:g}"
    )

    axis.set_title(
        f"NaCl = {salt_label} mM"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.22,
    )

for empty_index in range(
    len(salts),
    number_of_rows
    * number_of_columns,
):
    row_index = (
        empty_index
        // number_of_columns
    )

    column_index = (
        empty_index
        % number_of_columns
    )

    axes[
        row_index,
        column_index,
    ].axis("off")

ordered_handles = [
    legend_handles[phase]
    for phase in phase_categories
    if phase in legend_handles
]

ordered_labels = [
    phase
    for phase in phase_categories
    if phase in legend_handles
]

figure.legend(
    ordered_handles,
    ordered_labels,
    loc="lower center",
    ncol=min(
        4,
        max(1, len(ordered_labels)),
    ),
    frameon=False,
    bbox_to_anchor=(
        0.5,
        -0.01,
    ),
)

figure.suptitle(
    (
        "refCBA operational-state map from the combined fast-scan dataset\n"
        "Marker size indicates agreement among repeated observations; "
        "× denotes a disputed modal classification"
    )
)

figure.tight_layout(
    rect=[
        0.0,
        0.08,
        1.0,
        0.95,
    ]
)

figure4_png, figure4_pdf = save_figure(
    figure,
    "figure4_refcba_operational_state_map",
)


# ---------------------------------------------------------------------------
# Figure manifest and README
# ---------------------------------------------------------------------------

manifest = pd.DataFrame(
    [
        {
            "figure": "Figure 1",
            "png": str(
                figure1_png.relative_to(ROOT)
            ),
            "pdf": str(
                figure1_pdf.relative_to(ROOT)
            ),
            "data_source": str(
                LJ_FILE.relative_to(ROOT)
            ),
            "interpretation": (
                "Numerical validation of the perturbation and "
                "coexistence framework against NIST Lennard–Jones "
                "reference data."
            ),
        },
        {
            "figure": "Figure 2",
            "png": str(
                figure2_png.relative_to(ROOT)
            ),
            "pdf": str(
                figure2_pdf.relative_to(ROOT)
            ),
            "data_source": str(
                POTENTIAL_FILE.relative_to(ROOT)
            ),
            "interpretation": (
                "Reconstructed Huang A1 potential trend: "
                "the attractive well deepens and the repulsive "
                "barrier decreases with increasing pH."
            ),
        },
        {
            "figure": "Figure 3",
            "png": str(
                figure3_png.relative_to(ROOT)
            ),
            "pdf": str(
                figure3_pdf.relative_to(ROOT)
            ),
            "data_source": str(
                A2_FILE.relative_to(ROOT)
            ),
            "interpretation": (
                "At low pH, the squared repulsive potential dominates "
                "the second moment and generates a large negative "
                "second-order correction."
            ),
        },
        {
            "figure": "Figure 4",
            "png": str(
                figure4_png.relative_to(ROOT)
            ),
            "pdf": str(
                figure4_pdf.relative_to(ROOT)
            ),
            "data_source": str(
                STATE_FILE.relative_to(ROOT)
            ),
            "interpretation": (
                "Operational MD classifications across pH, salt and "
                "concentration. This is not an equilibrium LLPS "
                "binodal."
            ),
        },
    ]
)

manifest.to_csv(
    MANIFEST_FILE,
    index=False,
)

unique_state_count = int(
    len(aggregated)
)

raw_state_count = int(
    len(states)
)

repeat_state_count = int(
    (
        aggregated[
            "number_of_observations"
        ]
        > 1
    ).sum()
)

disputed_state_count = int(
    (
        aggregated[
            "classification_agreement"
        ]
        < 0.999
    ).sum()
)

readme = f"""# Final reproduction figure set

Generated from the frozen Huang/refCBA reproduction project.

## Included figures

1. `figure1_lj_nist_validation`
   - Validates the liquid-perturbation and coexistence solver against
     NIST Lennard–Jones reference data.

2. `figure2_huang_potential_landmarks`
   - Shows the reconstructed pH dependence of the Huang A1 attractive
     well and repulsive barrier.

3. `figure3_huang_a2_sign_split`
   - Shows why the publicly described full-potential second-order term
     incorrectly favors low-pH phase separation.

4. `figure4_refcba_operational_state_map`
   - Shows MD-based operational classifications across pH, NaCl and
     concentration.

## Operational-map coverage

- Raw combined-fast-scan rows: {raw_state_count}
- Unique pH–salt–concentration states: {unique_state_count}
- States represented by repeated observations: {repeat_state_count}
- States with classification disagreement: {disputed_state_count}

Repeated states are represented by their modal operational class.
Marker size reflects classification agreement, and an `x` marks states
where repeated observations did not fully agree.

## Interpretation boundary

Figure 4 is an operational kinetic-state map, not a validated equilibrium
phase diagram.

The current reproduction does not provide a strictly validated LLPS
binodal for refCBA.

Required reporting language:

> LLPS binodal unresolved under the publicly specified perturbation theory.

The pH 5.7 attractive-only sensitivity solution is retained as a numerical
diagnostic, not as the strictly reproduced Huang Figure 9 boundary.

## Main scientific result

The potential trend and the numerical liquid-theory framework were
reconstructed and independently validated. However, applying the published
second-order correction to the full potential causes the squared repulsive
barrier to generate a large negative free-energy correction at low pH.
This reverses the reported phase-ordering trend.

Restricting the second-order term to the attractive part restores the
qualitative pH direction but does not quantitatively reproduce the reported
0.436 mg/mL boundary.

Therefore, the publicly specified equations and parameters are insufficient
to uniquely and stably reconstruct the reported Huang Figure 9 LLPS
binodal.
"""

README_FILE.write_text(
    textwrap.dedent(readme),
    encoding="utf-8",
)

print("=" * 100)
print("FINAL REPRODUCTION FIGURES")
print("=" * 100)

for item in manifest.to_dict(
    orient="records"
):
    print(
        f"{item['figure']}: "
        f"{item['png']}"
    )

print()
print(
    "raw combined-fast-scan rows:",
    raw_state_count,
)

print(
    "unique pH-salt-concentration states:",
    unique_state_count,
)

print(
    "states with repeated observations:",
    repeat_state_count,
)

print(
    "states with classification disagreement:",
    disputed_state_count,
)

print()
print("saved:", MANIFEST_FILE)
print("saved:", README_FILE)
print("saved:", AGGREGATED_STATE_FILE)
print("saved:", PHASE_COUNT_FILE)
print("saved:", VALIDATION_COUNT_FILE)
print()
print("MAKE_FINAL_REPRODUCTION_FIGURES: COMPLETE")
