#!/usr/bin/env python3
"""Plot two additional pH-NaCl concentration slices without invented fills.

The long 30 ns evidence table is used when a coordinate exists there.  Missing
coordinates are filled only by the original 0.5 ns screening label and remain
visually marked as screening evidence.  No spatial classifier or global
interpolation is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


COLORS = {
    "soluble": "#4C78A8",
    "llps_candidate": "#54A24B",
    "aggregation": "#E45756",
    "finite_mobile": "#666666",
    "unresolved": "#B279A2",
}

MARKERS = {
    "soluble": "o",
    "llps_candidate": "s",
    "aggregation": "^",
    "finite_mobile": "o",
    "unresolved": "D",
}


def classify_screen(label: object) -> str:
    text = str(label).strip().lower()
    if "soluble" in text or "weak_oligomer" in text:
        return "soluble"
    if "dynamic_coarsening" in text or "llps" in text:
        return "llps_candidate"
    if any(k in text for k in ["aggregation", "aggregate", "arrested", "percolat"]):
        return "aggregation"
    if "finite_cluster" in text:
        return "finite_mobile"
    return "unresolved"


def classify_long(phase: object) -> str:
    text = str(phase).strip().lower()
    if text == "soluble":
        return "soluble"
    if text == "llps":
        return "llps_candidate"
    if text == "aggregation":
        return "aggregation"
    if text == "finite_mobile":
        return "finite_mobile"
    return "unresolved"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--screen",
        type=Path,
        default=Path("results/hoomd_coarse_224_0p5ns/hoomd_coarse_224_screening_classified.csv"),
    )
    parser.add_argument(
        "--long",
        type=Path,
        default=Path("results/FINAL_PUBLICATION_OUTPUTS/final_115_coordinate_master_table.csv"),
    )
    parser.add_argument("--concentrations", nargs=2, type=float, default=[5.0, 10.0])
    parser.add_argument("--outdir", type=Path, default=Path("results/PHASE_FIGURES_ADDITIONAL"))
    args = parser.parse_args()

    screen = pd.read_csv(args.screen)
    long = pd.read_csv(args.long)
    args.outdir.mkdir(parents=True, exist_ok=True)

    screen_label = "screening_class"
    required_screen = {"pH", "nacl_mM", "concentration_mg_ml", screen_label}
    required_long = {"pH", "NaCl_mM", "concentration_mg_ml", "final_phase_conservative"}
    if not required_screen.issubset(screen.columns):
        raise ValueError(f"Screening table is missing: {sorted(required_screen - set(screen.columns))}")
    if not required_long.issubset(long.columns):
        raise ValueError(f"Long table is missing: {sorted(required_long - set(long.columns))}")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.3), dpi=220, sharex=True, sharey=True)
    exported = []

    for ax, concentration, panel in zip(axes, args.concentrations, ["A", "B"]):
        coarse = screen[screen["concentration_mg_ml"].astype(float).sub(concentration).abs() < 1e-8].copy()
        coarse["phase_plot"] = coarse[screen_label].map(classify_screen)
        coarse["evidence_level_plot"] = "0.5 ns screen"

        validated = long[long["concentration_mg_ml"].astype(float).sub(concentration).abs() < 1e-8].copy()
        validated["phase_plot"] = validated["final_phase_conservative"].map(classify_long)
        validated["evidence_level_plot"] = "30 ns evidence"

        long_keys = set(zip(validated["pH"].astype(float), validated["NaCl_mM"].astype(float)))
        coarse = coarse[
            [(float(r.pH), float(r.nacl_mM)) not in long_keys for r in coarse.itertuples(index=False)]
        ]

        for phase in COLORS:
            q = coarse[coarse["phase_plot"] == phase]
            if not q.empty:
                ax.scatter(
                    q["pH"], q["nacl_mM"], marker=MARKERS[phase], s=54,
                    facecolors="none", edgecolors=COLORS[phase], linewidths=1.25,
                    alpha=0.78, zorder=3,
                )
            q = validated[validated["phase_plot"] == phase]
            if not q.empty:
                face = "none" if phase in {"finite_mobile", "unresolved"} else COLORS[phase]
                ax.scatter(
                    q["pH"], q["NaCl_mM"], marker=MARKERS[phase], s=78,
                    facecolors=face, edgecolors=COLORS[phase], linewidths=1.5,
                    zorder=5,
                )

        combined = pd.concat([
            coarse.rename(columns={"nacl_mM": "NaCl_mM"})[
                ["pH", "NaCl_mM", "concentration_mg_ml", "phase_plot", "evidence_level_plot"]
            ],
            validated[["pH", "NaCl_mM", "concentration_mg_ml", "phase_plot", "evidence_level_plot"]],
        ], ignore_index=True).sort_values(["pH", "NaCl_mM"])
        exported.append(combined)
        combined.to_csv(args.outdir / f"slice_c{concentration:g}_evidence.csv", index=False)

        ax.set_title(f"{panel}   c = {concentration:g} mg/mL", loc="left", fontweight="bold")
        ax.set_xlabel("pH")
        ax.set_xlim(2.85, 9.15)
        ax.set_ylim(-10, 510)
        ax.grid(alpha=0.18, linewidth=0.7)

    axes[0].set_ylabel("NaCl (mM)")
    fig.suptitle("refCBA additional concentration slices - evidence only", fontweight="bold")

    phase_handles = [
        Line2D([0], [0], marker=MARKERS[p], linestyle="none", markerfacecolor=(
            "none" if p in {"finite_mobile", "unresolved"} else COLORS[p]
        ), markeredgecolor=COLORS[p], label={
            "soluble": "Soluble",
            "llps_candidate": "LLPS candidate",
            "aggregation": "Aggregation",
            "finite_mobile": "Finite/mobile",
            "unresolved": "Unresolved",
        }[p])
        for p in COLORS
    ]
    evidence_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor="#444444", label="0.5 ns screen (open)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#444444",
               markeredgecolor="#444444", label="30 ns evidence (filled/core)"),
    ]
    fig.legend(
        handles=phase_handles + evidence_handles, loc="lower center", ncol=4,
        bbox_to_anchor=(0.5, -0.01), frameon=True, fontsize=8,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.22, wspace=0.14)

    png = args.outdir / "refCBA_additional_concentration_slices_c5_c10.png"
    pdf = args.outdir / "refCBA_additional_concentration_slices_c5_c10.pdf"
    fig.savefig(png, dpi=350, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    pd.concat(exported, ignore_index=True).to_csv(
        args.outdir / "additional_concentration_slices_all_evidence.csv", index=False
    )
    print(f"PNG: {png}")
    print(f"PDF: {pdf}")


if __name__ == "__main__":
    main()
