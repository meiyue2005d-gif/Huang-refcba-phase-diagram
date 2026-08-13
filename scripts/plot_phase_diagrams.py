#!/usr/bin/env python3
"""Plot honest discrete pH-salt phase slices at selected concentrations."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PHASES = ["soluble", "llps", "aggregate", "unresolved"]
COLORS = ["#4C9BE8", "#F2C14E", "#D95D5D", "#9E9E9E"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concentrations", type=float, nargs="+", default=[5.0, 10.0, 20.0])
    parser.add_argument("--phase-column", default="phase_conservative")
    args = parser.parse_args()
    table = pd.read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {name: index for index, name in enumerate(PHASES)}
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(PHASES) + 0.5), cmap.N)

    fig, axes = plt.subplots(1, len(args.concentrations), figsize=(5.2 * len(args.concentrations), 5.2), squeeze=False)
    for panel, (axis, concentration) in enumerate(zip(axes[0], args.concentrations)):
        subset = table[np.isclose(table["concentration_mg_ml"], concentration)]
        phs = sorted(table["ph"].unique())
        salts = sorted(table["nacl_mM"].unique())
        matrix = np.full((len(salts), len(phs)), mapping["unresolved"], dtype=int)
        for iy, salt in enumerate(salts):
            for ix, ph in enumerate(phs):
                hit = subset[np.isclose(subset["ph"], ph) & np.isclose(subset["nacl_mM"], salt)]
                if not hit.empty:
                    value = str(hit.iloc[0][args.phase_column]).lower()
                    if value == "aggregation":
                        value = "aggregate"
                    matrix[iy, ix] = mapping.get(value, mapping["unresolved"])
        axis.imshow(matrix, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        axis.set_xticks(range(len(phs)), [f"{v:g}" for v in phs], rotation=45, ha="right")
        axis.set_yticks(range(len(salts)), [f"{v:g}" for v in salts])
        axis.set_xlabel("pH")
        if panel == 0:
            axis.set_ylabel("NaCl (mM)")
        axis.set_title(f"{chr(65 + panel)}  {concentration:g} mg/mL", loc="left")
        for iy in range(len(salts)):
            for ix in range(len(phs)):
                axis.text(ix, iy, PHASES[matrix[iy, ix]][0].upper(), ha="center", va="center", fontsize=8)
    handles = [Patch(facecolor=color, label=phase) for phase, color in zip(PHASES, COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=len(PHASES), frameon=False)
    fig.suptitle("refCBA discrete three-phase map (sampled states; no interpolation)")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    stem = "phase_diagram_concentration_slices"
    fig.savefig(args.output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(args.output_dir / f"{stem}.png")


if __name__ == "__main__":
    main()
