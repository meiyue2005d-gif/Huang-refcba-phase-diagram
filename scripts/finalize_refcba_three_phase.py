#!/usr/bin/env python3
"""Finalize a historical refCBA pH-salt-concentration scan into a
soluble / LLPS / aggregation phase table and extract boundary surfaces.

The script is intentionally defensive: it searches under --results-root for the
224-state screening CSV, normalizes common column names, merges available 5 ns
and 30 ns validation summaries, and emits both a conservative label and a
three-class operational label.

Usage
-----
python finalize_refcba_three_phase.py --results-root results

Main outputs are written to results/final_three_phase_boundary/.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


PHASE_ORDER = {"soluble": 0, "llps": 1, "aggregation": 2, "unresolved": 3}


def norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def norm_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [norm_name(c) for c in out.columns]
    aliases = {
        "nacl": "nacl_mm",
        "salt_mm": "nacl_mm",
        "nacl_m_m": "nacl_mm",
        "concentration": "concentration_mg_ml",
        "concentration_mgml": "concentration_mg_ml",
        "c_mg_ml": "concentration_mg_ml",
        "ph_value": "ph",
        "state": "state_id",
        "run": "state_id",
        "task": "state_id",
        "run_token": "state_id",
    }
    out = out.rename(columns={c: aliases.get(c, c) for c in out.columns})
    return out


def numeric_key(df: pd.DataFrame) -> bool:
    return {"ph", "nacl_mm", "concentration_mg_ml"}.issubset(df.columns)


def csv_candidates(root: Path) -> list[tuple[float, Path, pd.DataFrame]]:
    candidates: list[tuple[float, Path, pd.DataFrame]] = []
    class_hints = {
        "screening_class_0p5ns",
        "screening_class",
        "classification_0p5ns",
        "classification",
        "coarse_phase",
        "script_interpretation",
        "interpretation",
    }
    for path in root.rglob("*.csv"):
        try:
            raw = pd.read_csv(path)
        except Exception:
            continue
        df = norm_columns(raw)
        if not numeric_key(df):
            continue
        n = len(df)
        present = class_hints.intersection(df.columns)
        score = 0.0
        if n == 224:
            score += 1000
        elif 180 <= n <= 260:
            score += 300 - abs(n - 224)
        elif n >= 100:
            score += 50
        score += 40 * len(present)
        name = str(path).lower()
        if "224" in name:
            score += 150
        if "screen" in name or "coarse" in name:
            score += 60
        if "transition" in name or "boundary" in name:
            score -= 120
        candidates.append((score, path, df))
    return sorted(candidates, key=lambda x: x[0], reverse=True)


def choose_screening_table(root: Path, explicit: Optional[Path]) -> tuple[Path, pd.DataFrame]:
    if explicit is not None:
        df = norm_columns(pd.read_csv(explicit))
        if not numeric_key(df):
            raise ValueError(f"Explicit screening CSV lacks pH/NaCl/concentration columns: {explicit}")
        return explicit, df
    candidates = csv_candidates(root)
    if not candidates:
        raise FileNotFoundError(f"No suitable screening CSV found below {root}")
    score, path, df = candidates[0]
    return path, df


def first_existing(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def normalize_label(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return norm_name(str(value))


def classify_screen_label(label: object) -> tuple[str, str, str]:
    """Return conservative phase, operational phase, evidence note."""
    s = normalize_label(label)
    if not s:
        return "unresolved", "unresolved", "missing_screen_label"

    if any(k in s for k in ["mostly_dispersed", "soluble_screen", "soluble_like", "dispersed"]):
        return "soluble", "soluble", "short_screen_dispersed"

    if any(k in s for k in ["weak_oligomer", "oligomer"]):
        # User-requested 3-state compression: weak reversible oligomers remain soluble.
        return "soluble", "soluble", "short_screen_weak_oligomer"

    if any(k in s for k in ["arrested", "strong_aggregation", "aggregate", "gel", "percolat"]):
        return "aggregation", "aggregation", "short_screen_aggregation_signal"

    if any(k in s for k in ["dynamic_coarsening", "coarsening", "llps"]):
        # Conservative label remains unresolved until longer-time evidence is merged.
        return "unresolved", "llps", "short_screen_dynamic_coarsening_candidate"

    if any(k in s for k in ["finite_cluster", "cluster_candidate", "clustered_state"]):
        # Huang explicitly identifies an equilibrium cluster fluid.  Without
        # arrest or percolation, it belongs to the soluble public phase.
        return "soluble", "soluble", "short_screen_mobile_finite_cluster_fluid"

    if "unresolved" in s or "review" in s:
        return "unresolved", "unresolved", "screen_unresolved"

    return "unresolved", "unresolved", f"unrecognized_screen_label:{s}"


def key_round(series: pd.Series, ndigits: int = 6) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round(ndigits)


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["ph", "nacl_mm", "concentration_mg_ml"]:
        if col not in out:
            out[col] = np.nan
        out[f"_{col}_key"] = key_round(out[col])
    return out


def merge_validation_panel(base: pd.DataFrame, root: Path) -> pd.DataFrame:
    paths = list(root.rglob("validation_panel_5ns_summary.csv"))
    if not paths:
        return base
    val = add_keys(norm_columns(pd.read_csv(paths[0])))
    label_col = first_existing(val, ["classification_5ns", "classification", "provisional_phase"])
    if label_col is None:
        return base

    keep = ["_ph_key", "_nacl_mm_key", "_concentration_mg_ml_key", label_col]
    for c in ["llps_candidate_score", "late_largest_slope_particles_per_ns",
              "late_clustered_slope_per_ns", "late_largest_cluster_fraction",
              "late_free_particle_fraction", "mean_clustered_fraction_5ns"]:
        if c in val.columns:
            keep.append(c)
    val = val[keep].drop_duplicates(keep[:3], keep="last")
    return base.merge(val, on=keep[:3], how="left", suffixes=("", "_5ns"))


def apply_5ns_overrides(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    label_col = first_existing(out, ["classification_5ns", "classification_5ns_5ns"])
    if label_col is None:
        return out

    for idx, row in out.iterrows():
        s = normalize_label(row.get(label_col))
        if not s:
            continue
        if any(k in s for k in ["mostly_dispersed", "soluble"]):
            out.at[idx, "phase_conservative"] = "soluble"
            out.at[idx, "phase_operational"] = "soluble"
            out.at[idx, "evidence_level"] = "medium"
            out.at[idx, "evidence_source"] = "5ns_validation"
        elif any(k in s for k in ["arrested", "aggregation", "aggregate", "gel"]):
            out.at[idx, "phase_conservative"] = "aggregation"
            out.at[idx, "phase_operational"] = "aggregation"
            out.at[idx, "evidence_level"] = "medium"
            out.at[idx, "evidence_source"] = "5ns_validation"
        elif any(k in s for k in ["dynamic_coarsening", "llps"]):
            out.at[idx, "phase_operational"] = "llps"
            # A homogeneous 5 ns trajectory can nominate a slab test but cannot
            # establish equilibrium coexistence regardless of its screen score.
            out.at[idx, "phase_conservative"] = "unresolved"
            out.at[idx, "evidence_level"] = "low"
            out.at[idx, "evidence_source"] = "5ns_validation"
        elif any(k in s for k in ["finite_cluster", "mobile_cluster", "clustered_state"]):
            out.at[idx, "phase_operational"] = "soluble"
            out.at[idx, "phase_conservative"] = "soluble"
            out.at[idx, "evidence_level"] = "medium"
            out.at[idx, "evidence_source"] = "5ns_finite_or_mobile_cluster"
        elif "unresolved" in s:
            out.at[idx, "phase_conservative"] = "unresolved"
            out.at[idx, "evidence_level"] = "low"
            out.at[idx, "evidence_source"] = "5ns_unresolved"
    return out


def parse_state_id(text: object) -> tuple[float, float, float]:
    s = str(text)
    ph = nacl = conc = np.nan
    m = re.search(r"pH([0-9]+(?:p[0-9]+)?)", s, flags=re.I)
    if m:
        ph = float(m.group(1).replace("p", "."))
    m = re.search(r"nacl([0-9]+(?:p[0-9]+)?)", s, flags=re.I)
    if m:
        nacl = float(m.group(1).replace("p", "."))
    m = re.search(r"_c([0-9]+(?:p[0-9]+)?)", s, flags=re.I)
    if m:
        conc = float(m.group(1).replace("p", "."))
    return ph, nacl, conc


def merge_long_validation(base: pd.DataFrame, root: Path) -> pd.DataFrame:
    records: list[dict] = []
    patterns = ["*30ns*summary.csv", "nine_state_summary.csv", "key5_uniform30ns_summary.csv"]
    seen: set[Path] = set()
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path in seen:
                continue
            seen.add(path)
            try:
                val = norm_columns(pd.read_csv(path))
            except Exception:
                continue
            id_col = first_existing(val, ["state_id"])
            label_col = first_existing(val, [
                "single_seed_readout",
                "provisional_phase",
                "classification_30ns",
                "interpretation",
                "classification",
                "script_interpretation",
            ])
            if label_col is None:
                continue
            for _, row in val.iterrows():
                ph = row.get("ph", np.nan)
                nacl = row.get("nacl_mm", np.nan)
                conc = row.get("concentration_mg_ml", np.nan)
                if id_col and (pd.isna(ph) or pd.isna(nacl) or pd.isna(conc)):
                    p2, n2, c2 = parse_state_id(row.get(id_col))
                    ph = p2 if pd.isna(ph) else ph
                    nacl = n2 if pd.isna(nacl) else nacl
                    conc = c2 if pd.isna(conc) else conc
                # Some direct coexistence refinement tables are salt-specific and
                # omit NaCl. Recover it from parent directory name.
                if pd.isna(nacl):
                    m = re.search(r"nacl([0-9]+(?:p[0-9]+)?)", str(path.parent), flags=re.I)
                    if m:
                        nacl = float(m.group(1).replace("p", "."))
                records.append({
                    "_ph_key": round(float(ph), 6) if pd.notna(ph) else np.nan,
                    "_nacl_mm_key": round(float(nacl), 6) if pd.notna(nacl) else np.nan,
                    "_concentration_mg_ml_key": round(float(conc), 6) if pd.notna(conc) else np.nan,
                    "long_label": row.get(label_col),
                    "long_source": str(path.relative_to(root)),
                    "long_final_initial_bond_survival": row.get("final_initial_bond_survival", row.get("final_retention", np.nan)),
                    "long_mean_consecutive_bond_retention": row.get("mean_consecutive_bond_retention", np.nan),
                    "long_neighbor_jaccard": row.get("neighbor_jaccard", np.nan),
                    "long_exchange": row.get("exchange", np.nan),
                    "long_is_direct_coexistence": (
                        "provisional_phase" in val.columns
                        or "direct_coexistence" in str(path).lower()
                    ),
                })
    if not records:
        return base
    val = pd.DataFrame(records).dropna(subset=["_ph_key", "_concentration_mg_ml_key"])
    # Only exact salt matches can safely override the 3-D grid.
    val = val.dropna(subset=["_nacl_mm_key"]).drop_duplicates(
        ["_ph_key", "_nacl_mm_key", "_concentration_mg_ml_key"], keep="last"
    )
    return base.merge(val, on=["_ph_key", "_nacl_mm_key", "_concentration_mg_ml_key"], how="left")


def apply_long_overrides(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "long_label" not in out.columns:
        return out
    for idx, row in out.iterrows():
        s = normalize_label(row.get("long_label"))
        if not s:
            continue
        survival = pd.to_numeric(pd.Series([row.get("long_final_initial_bond_survival", np.nan)]), errors="coerce").iloc[0]
        retention = pd.to_numeric(pd.Series([row.get("long_mean_consecutive_bond_retention", np.nan)]), errors="coerce").iloc[0]
        exchange = row.get("long_exchange", np.nan)

        direct_value = row.get("long_is_direct_coexistence", False)
        direct_coexistence = (
            direct_value is True
            or str(direct_value).strip().lower() == "true"
        )
        exchange_true = exchange is True or str(exchange).strip().lower() == "true"

        if any(k in s for k in ["mostly_dispersed", "soluble", "dissolving"]):
            conservative_phase = operational_phase = "soluble"
        elif any(k in s for k in ["arrested", "aggregate", "aggregation", "gel"]):
            conservative_phase = operational_phase = "aggregation"
        elif any(k in s for k in ["llps", "dynamic_condensation", "coarsening"]):
            # Very high bond persistence without exchange is aggregation, not LLPS.
            arrested_by_metrics = (
                (pd.notna(survival) and survival >= 0.80) and
                (pd.notna(retention) and retention >= 0.95) and
                (exchange is False or str(exchange).lower() == "false")
            )
            if arrested_by_metrics:
                conservative_phase = operational_phase = "aggregation"
            else:
                operational_phase = "llps"
                # Homogeneous NVT coarsening is an LLPS candidate, not proof of
                # equilibrium coexistence.  Promote conservatively only when a
                # direct-coexistence analysis also demonstrates exchange.
                conservative_phase = (
                    "llps"
                    if direct_coexistence and exchange_true
                    else "unresolved"
                )
        elif any(k in s for k in ["finite_cluster", "mobile_cluster"]):
            conservative_phase = operational_phase = "soluble"
        elif any(k in s for k in ["clustered_unresolved", "unresolved"]):
            conservative_phase = operational_phase = "unresolved"
        else:
            continue

        out.at[idx, "phase_conservative"] = conservative_phase
        out.at[idx, "phase_operational"] = operational_phase
        out.at[idx, "evidence_level"] = (
            "high" if conservative_phase != "unresolved" else "medium"
        )
        out.at[idx, "evidence_source"] = row.get("long_source", "long_validation")
    return out


def smooth_isolated_bins(df: pd.DataFrame, phase_col: str) -> pd.DataFrame:
    """Only repair a single concentration-bin island bracketed by the same phase.
    Original labels are retained in phase_before_topology_cleanup.
    """
    out = df.copy()
    out["phase_before_topology_cleanup"] = out[phase_col]
    for (_, _), idxs in out.groupby(["ph", "nacl_mm"], dropna=False).groups.items():
        order = out.loc[idxs].sort_values("concentration_mg_ml").index.tolist()
        vals = out.loc[order, phase_col].tolist()
        for i in range(1, len(vals) - 1):
            if vals[i - 1] == vals[i + 1] and vals[i] != vals[i - 1] and vals[i] == "unresolved":
                out.at[order[i], phase_col] = vals[i - 1]
                out.at[order[i], "topology_cleanup"] = "filled_single_unresolved_island"
    return out


def extract_boundaries(df: pd.DataFrame, phase_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    transition_rows: list[dict] = []
    envelope_rows: list[dict] = []

    for (ph, salt), g in df.groupby(["ph", "nacl_mm"], dropna=False):
        g = g.sort_values("concentration_mg_ml")
        phases = g[phase_col].tolist()
        concs = g["concentration_mg_ml"].astype(float).tolist()
        ids = g.get("state_id", pd.Series([""] * len(g), index=g.index)).astype(str).tolist()

        for i in range(len(g) - 1):
            if phases[i] != phases[i + 1]:
                transition_rows.append({
                    "ph": ph,
                    "nacl_mm": salt,
                    "low_concentration_mg_ml": concs[i],
                    "high_concentration_mg_ml": concs[i + 1],
                    "low_phase": phases[i],
                    "high_phase": phases[i + 1],
                    "boundary_geometric_mid_mg_ml": math.sqrt(concs[i] * concs[i + 1]) if concs[i] > 0 else (concs[i] + concs[i + 1]) / 2,
                    "low_state_id": ids[i],
                    "high_state_id": ids[i + 1],
                })

        def vals(phase: str) -> list[float]:
            return [c for c, p in zip(concs, phases) if p == phase]

        sv, lv, av, uv = vals("soluble"), vals("llps"), vals("aggregation"), vals("unresolved")
        envelope_rows.append({
            "ph": ph,
            "nacl_mm": salt,
            "soluble_max_mg_ml": max(sv) if sv else np.nan,
            "llps_min_mg_ml": min(lv) if lv else np.nan,
            "llps_max_mg_ml": max(lv) if lv else np.nan,
            "aggregation_min_mg_ml": min(av) if av else np.nan,
            "unresolved_count": len(uv),
            "phase_sequence": " -> ".join(phases),
            "concentration_sequence_mg_ml": " -> ".join(f"{x:g}" for x in concs),
        })

    return pd.DataFrame(transition_rows), pd.DataFrame(envelope_rows)


def validation_manifest(df: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    selected: set[tuple[float, float, float]] = set()

    # Always include conservative unresolved states and all operational LLPS states.
    mask = (df["phase_conservative"] == "unresolved") | (df["phase_operational"] == "llps")
    for _, r in df.loc[mask].iterrows():
        selected.add((float(r.ph), float(r.nacl_mm), float(r.concentration_mg_ml)))

    # Include both sides of all boundaries involving LLPS or unresolved.
    for _, r in transitions.iterrows():
        pair = {r.low_phase, r.high_phase}
        if "llps" in pair or "unresolved" in pair:
            selected.add((float(r.ph), float(r.nacl_mm), float(r.low_concentration_mg_ml)))
            selected.add((float(r.ph), float(r.nacl_mm), float(r.high_concentration_mg_ml)))

    rows = []
    for ph, salt, conc in sorted(selected):
        hit = df[(np.isclose(df.ph, ph)) & (np.isclose(df.nacl_mm, salt)) & (np.isclose(df.concentration_mg_ml, conc))]
        if hit.empty:
            continue
        r = hit.iloc[0]
        reason = []
        if r.phase_conservative == "unresolved":
            reason.append("unresolved")
        if r.phase_operational == "llps":
            reason.append("llps_candidate")
        if not reason:
            reason.append("boundary_neighbor")
        rows.append({
            "ph": ph,
            "nacl_mm": salt,
            "concentration_mg_ml": conc,
            "state_id": r.get("state_id", ""),
            "current_operational_phase": r.phase_operational,
            "current_conservative_phase": r.phase_conservative,
            "evidence_level": r.evidence_level,
            "selection_reason": "+".join(reason),
            "recommended_duration_ns": 30.0,
            "recommended_seeds": "20260723;20260724;20260725",
        })
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, outdir: Path, phase_col: str) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
    except Exception as exc:
        print(f"Plotting skipped: {exc}")
        return

    phase_to_int = {"soluble": 0, "llps": 1, "aggregation": 2, "unresolved": 3}
    cmap = ListedColormap(["#4C9BE8", "#F1C75B", "#D65A5A", "#A0A0A0"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    for salt, g in df.groupby("nacl_mm"):
        pHs = sorted(g.ph.unique())
        concs = sorted(g.concentration_mg_ml.unique())
        matrix = np.full((len(concs), len(pHs)), np.nan)
        for i, c in enumerate(concs):
            for j, ph in enumerate(pHs):
                hit = g[np.isclose(g.ph, ph) & np.isclose(g.concentration_mg_ml, c)]
                if not hit.empty:
                    matrix[i, j] = phase_to_int.get(hit.iloc[0][phase_col], 3)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        im = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, norm=norm,
                       extent=[min(pHs)-0.125, max(pHs)+0.125,
                               math.log10(min(concs))-0.08, math.log10(max(concs))+0.08])
        ax.set_xlabel("pH")
        ax.set_ylabel("log10 concentration (mg/mL)")
        ax.set_title(f"refCBA conservative three-phase map, NaCl = {salt:g} mM")
        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(["soluble", "LLPS", "aggregation", "unresolved"])
        fig.tight_layout()
        fig.savefig(outdir / f"phase_map_nacl_{salt:g}mM.png", dpi=220)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--screening-csv", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    root = args.results_root.resolve()
    outdir = (args.output_dir or root / "final_three_phase_boundary").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    screen_path, screen = choose_screening_table(root, args.screening_csv)
    print(f"Selected screening table: {screen_path}")
    print(f"Rows: {len(screen)}")

    class_col = first_existing(screen, [
        "screening_class_0p5ns", "screening_class", "classification_0p5ns",
        "classification", "coarse_phase", "script_interpretation", "interpretation",
    ])
    if class_col is None:
        raise ValueError(f"No classification column found in {screen_path}; columns={list(screen.columns)}")

    screen = add_keys(screen)
    if "state_id" not in screen.columns:
        screen["state_id"] = [f"pH{p:g}_nacl{s:g}_c{c:g}" for p, s, c in zip(
            screen.ph, screen.nacl_mm, screen.concentration_mg_ml)]

    mapped = screen[class_col].apply(classify_screen_label)
    screen[["phase_conservative", "phase_operational", "classification_note"]] = pd.DataFrame(mapped.tolist(), index=screen.index)
    screen["evidence_level"] = "low"
    screen["evidence_source"] = "0p5ns_screen"

    merged = merge_validation_panel(screen, root)
    merged = apply_5ns_overrides(merged)
    merged = merge_long_validation(merged, root)
    merged = apply_long_overrides(merged)
    # Do not smooth the scientific labels.  Sparse-grid islands are preserved
    # as sampled evidence and handled by targeted refinement instead.

    # Drop internal key columns only in the public table.
    public = merged.sort_values(["nacl_mm", "ph", "concentration_mg_ml"]).copy()
    key_cols = [c for c in public.columns if c.startswith("_")]
    public = public.drop(columns=key_cols)

    # Public phase boundaries use only conservative evidence. Operational
    # candidates are retained in the state table and validation manifest.
    transitions, envelopes = extract_boundaries(public, "phase_conservative")
    manifest = validation_manifest(public, transitions)

    public.to_csv(outdir / "refcba_three_phase_state_table.csv", index=False)
    transitions.to_csv(outdir / "refcba_three_phase_boundary_transitions.csv", index=False)
    envelopes.to_csv(outdir / "refcba_three_phase_boundary_envelopes.csv", index=False)
    manifest.to_csv(outdir / "refcba_targeted_30ns_validation_manifest.csv", index=False)

    counts_oper = public["phase_operational"].value_counts(dropna=False).to_dict()
    counts_cons = public["phase_conservative"].value_counts(dropna=False).to_dict()
    report = {
        "screening_table": str(screen_path),
        "state_count": int(len(public)),
        "operational_phase_counts": {str(k): int(v) for k, v in counts_oper.items()},
        "conservative_phase_counts": {str(k): int(v) for k, v in counts_cons.items()},
        "transition_count": int(len(transitions)),
        "targeted_validation_state_count": int(len(manifest)),
        "interpretation": {
            "soluble": "dispersed plus weak reversible oligomers",
            "llps": "replicated persistent dense/dilute coexistence with dynamic exchange",
            "aggregation": "kinetically arrested finite aggregates or persistent percolated networks",
            "unresolved": "insufficient long-time evidence for a conservative assignment",
        },
    }
    (outdir / "refcba_three_phase_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_plots:
        make_plots(public, outdir, "phase_conservative")

    print("\nOperational counts:", counts_oper)
    print("Conservative counts:", counts_cons)
    print(f"Boundary transitions: {len(transitions)}")
    print(f"Targeted 30 ns states: {len(manifest)}")
    print(f"Outputs: {outdir}")

    if len(public) not in {224, 448}:
        print("WARNING: selected table is not a recognized 224- or 448-state grid. Inspect the selected CSV.")


if __name__ == "__main__":
    main()
