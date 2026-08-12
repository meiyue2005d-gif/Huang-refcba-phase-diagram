#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

RESULTS = Path("results")
INFILE = RESULTS / "active_learning" / "phase_training_records.csv"
OUT = RESULTS / "active_learning_v2"
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 20260821
PHASES = ["soluble", "llps", "aggregation"]


def scaled_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        (df["pH"].to_numpy(float) - 3.0) / 6.0,
        df["NaCl_mM"].to_numpy(float) / 500.0,
        (np.log10(df["concentration_mg_ml"].to_numpy(float)) + 1.0)
        / (math.log10(20.0) + 1.0),
    ])


def nearest_distance(points: np.ndarray, refs: np.ndarray) -> np.ndarray:
    if len(refs) == 0:
        return np.full(len(points), np.inf)
    out = np.full(len(points), np.inf)
    # Chunked to avoid a large temporary distance matrix.
    for start in range(0, len(points), 2000):
        block = points[start : start + 2000]
        dist2 = ((block[:, None, :] - refs[None, :, :]) ** 2).sum(axis=2)
        out[start : start + len(block)] = np.sqrt(dist2.min(axis=1))
    return out


if not INFILE.exists():
    raise SystemExit(
        f"Missing {INFILE}. Run scripts/build_active_learning_candidates.py first."
    )

raw = pd.read_csv(INFILE)
needed = {
    "pH", "NaCl_mM", "concentration_mg_ml", "phase", "seed", "weight", "source_file"
}
missing = needed.difference(raw.columns)
if missing:
    raise SystemExit(f"Training table is missing columns: {sorted(missing)}")

raw = raw[raw["phase"].isin(PHASES)].copy()
raw = raw[np.isfinite(raw["pH"]) & np.isfinite(raw["NaCl_mM"])].copy()
raw = raw[np.isfinite(raw["concentration_mg_ml"]) & (raw["concentration_mg_ml"] > 0)].copy()

# Collapse copied summaries and repeated references to the same physical run.
# For a known seed, one coordinate+seed is one run. For seedless summaries,
# retain the source file in the key so unrelated aggregate tables are not merged.
seed_text = raw["seed"].apply(
    lambda x: str(int(x)) if pd.notna(x) and np.isfinite(float(x)) else "noseed"
)
raw["run_key"] = (
    raw["pH"].round(6).astype(str) + "|"
    + raw["NaCl_mM"].round(6).astype(str) + "|"
    + raw["concentration_mg_ml"].round(6).astype(str) + "|"
    + seed_text
)
raw.loc[seed_text == "noseed", "run_key"] += "|" + raw.loc[
    seed_text == "noseed", "source_file"
].astype(str)

# Prefer the longest/highest-weight source for each physical run.
raw = raw.sort_values(["run_key", "weight"], ascending=[True, False])
maxw = raw.groupby("run_key")["weight"].transform("max")
top = raw[np.isclose(raw["weight"], maxw)].copy()

# If equally trusted copies disagree about the same run, exclude that run rather
# than silently selecting one label.
phase_n = top.groupby("run_key")["phase"].nunique()
conflict_keys = set(phase_n[phase_n > 1].index)
conflicts = top[top["run_key"].isin(conflict_keys)].copy()
clean = top[~top["run_key"].isin(conflict_keys)].drop_duplicates("run_key").copy()

# Trusted-first fit: use 10/30 ns evidence. If a class becomes too sparse,
# relax once to include 5 ns records, but never allow 0.5/1 ns records to drive it.
train = clean[clean["weight"] >= 0.60].copy()
counts = train["phase"].value_counts()
if any(counts.get(p, 0) < 4 for p in PHASES):
    train = clean[clean["weight"] >= 0.30].copy()
    counts = train["phase"].value_counts()

if any(counts.get(p, 0) < 3 for p in PHASES):
    raise SystemExit(
        "Not enough trusted records for all three phases after deduplication: "
        + counts.to_string()
    )

X = scaled_matrix(train)
y = train["phase"].to_numpy(str)
w = train["weight"].to_numpy(float)
groups = (
    train["pH"].round(5).astype(str) + "|"
    + train["NaCl_mM"].round(5).astype(str) + "|"
    + train["concentration_mg_ml"].round(5).astype(str)
).to_numpy(str)

model = ExtraTreesClassifier(
    n_estimators=1000,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model.fit(X, y, sample_weight=w)

# Coordinate-grouped CV prevents seeds/copies of the same state leaking across folds.
unique_groups_by_phase = train.assign(group=groups).groupby("phase")["group"].nunique()
folds = int(min(5, unique_groups_by_phase.min()))
bal_acc = float("nan")
if folds >= 3:
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    pred = np.empty(len(y), dtype=object)
    for tr, te in cv.split(X, y, groups):
        m = ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        m.fit(X[tr], y[tr], sample_weight=w[tr])
        pred[te] = m.predict(X[te])
    bal_acc = balanced_accuracy_score(y, pred)

# Coarse candidate lattice; finer interpolation comes later from the probability model.
ph_grid = np.round(np.arange(3.0, 9.0001, 0.25), 5)
nacl_grid = np.arange(0.0, 500.1, 25.0)
conc_grid = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0])
mesh = np.array(np.meshgrid(ph_grid, nacl_grid, conc_grid, indexing="ij"))
cand = pd.DataFrame({
    "pH": mesh[0].ravel(),
    "NaCl_mM": mesh[1].ravel(),
    "concentration_mg_ml": mesh[2].ravel(),
})
XC = scaled_matrix(cand)
raw_p = model.predict_proba(XC)
p = np.zeros((len(cand), 3), dtype=float)
for j, cls in enumerate(model.classes_):
    p[:, PHASES.index(cls)] = raw_p[:, j]

cand["P_soluble"] = p[:, 0]
cand["P_LLPS"] = p[:, 1]
cand["P_aggregation"] = p[:, 2]
cand["entropy"] = -(p * np.log(p + 1e-12)).sum(axis=1) / math.log(3.0)

# Exclude already sampled coordinates.
keys = set(zip(
    np.round(clean["pH"], 5),
    np.round(clean["NaCl_mM"], 5),
    np.round(clean["concentration_mg_ml"], 5),
))
cand = cand[[
    (round(r.pH, 5), round(r.NaCl_mM, 5), round(r.concentration_mg_ml, 5)) not in keys
    for r in cand.itertuples(index=False)
]].copy()
XC = scaled_matrix(cand)

# Penalize extrapolation: a boundary candidate should be near trusted examples of BOTH phases.
refs = {phase: scaled_matrix(train[train["phase"] == phase]) for phase in PHASES}
for phase in PHASES:
    cand[f"d_{phase}"] = nearest_distance(XC, refs[phase])

# Pairwise ambiguity plus local support from both phases.
def pair_score(pa, pb, pc, da, db):
    ambiguity = 4.0 * pa * pb * (1.0 - pc)
    support = np.exp(-((da / 0.24) ** 2 + (db / 0.24) ** 2))
    return ambiguity * support

cand["score_soluble_aggregation"] = pair_score(
    cand["P_soluble"], cand["P_aggregation"], cand["P_LLPS"],
    cand["d_soluble"], cand["d_aggregation"],
)
cand["score_LLPS_aggregation"] = pair_score(
    cand["P_LLPS"], cand["P_aggregation"], cand["P_soluble"],
    cand["d_llps"], cand["d_aggregation"],
)

# LLPS evidence currently lives at high concentration; do not let the model invent
# a low-concentration LLPS boundary far from every observed LLPS state.
sa_pool = cand[
    (cand["P_soluble"] >= 0.15)
    & (cand["P_aggregation"] >= 0.15)
    & (cand["d_soluble"] <= 0.38)
    & (cand["d_aggregation"] <= 0.38)
].copy()
la_pool = cand[
    (cand["concentration_mg_ml"] >= 10.0)
    & (cand["P_LLPS"] >= 0.12)
    & (cand["P_aggregation"] >= 0.15)
    & (cand["d_llps"] <= 0.38)
    & (cand["d_aggregation"] <= 0.38)
].copy()


def diverse_select(df: pd.DataFrame, score: str, n: int, occupied: list[np.ndarray]):
    chosen = []
    for _, row in df.sort_values(score, ascending=False).iterrows():
        point = scaled_matrix(pd.DataFrame([row]))[0]
        prior = occupied + [scaled_matrix(pd.DataFrame([r]))[0] for r in chosen]
        if any(np.linalg.norm(point - q) < 0.12 for q in prior):
            continue
        chosen.append(row)
        if len(chosen) >= n:
            break
    return chosen

selected_sa = diverse_select(sa_pool, "score_soluble_aggregation", 3, [])
occupied = [scaled_matrix(pd.DataFrame([r]))[0] for r in selected_sa]
selected_la = diverse_select(la_pool, "score_LLPS_aggregation", 3, occupied)

rows = []
for target, selected, score in [
    ("soluble-aggregation", selected_sa, "score_soluble_aggregation"),
    ("LLPS-aggregation", selected_la, "score_LLPS_aggregation"),
]:
    for row in selected:
        d = row.to_dict()
        d["boundary_target"] = target
        d["boundary_score"] = d[score]
        rows.append(d)
ranked = pd.DataFrame(rows)
if not ranked.empty:
    ranked = ranked.sort_values(["boundary_target", "boundary_score"], ascending=[True, False])

clean.to_csv(OUT / "deduplicated_all_records.csv", index=False)
train.to_csv(OUT / "trusted_training_records.csv", index=False)
conflicts.to_csv(OUT / "excluded_conflicting_runs.csv", index=False)
cand.sort_values("entropy", ascending=False).to_csv(OUT / "candidate_probability_grid.csv", index=False)
ranked.to_csv(OUT / "next_batch_ranked.csv", index=False)
with (OUT / "next_batch.tsv").open("w", encoding="utf-8") as f:
    f.write("# ph\tnacl_mM\tconcentration_mg_ml\n")
    for row in ranked.itertuples(index=False):
        f.write(f"{row.pH:.5g}\t{row.NaCl_mM:.5g}\t{row.concentration_mg_ml:.5g}\n")

print("===== DATA AUDIT =====")
print(f"Raw records             : {len(raw)}")
print(f"Deduplicated runs       : {len(clean)}")
print(f"Excluded conflicts      : {len(conflict_keys)}")
print(f"Trusted training runs   : {len(train)}")
print("\nTrusted phase counts:")
print(train["phase"].value_counts().to_string())
print(
    f"Grouped balanced CV accuracy: {bal_acc:.3f}"
    if math.isfinite(bal_acc)
    else "Grouped balanced CV accuracy: unavailable"
)
print("\n===== NEXT BATCH (RUN ONE SEED EACH FIRST) =====")
if ranked.empty:
    print("No supported candidates passed the filters.")
else:
    cols = [
        "boundary_target", "pH", "NaCl_mM", "concentration_mg_ml",
        "P_soluble", "P_LLPS", "P_aggregation",
        "d_soluble", "d_llps", "d_aggregation", "boundary_score",
    ]
    print(ranked[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\nSaved: {OUT / 'next_batch.tsv'}")
print(f"Saved: {OUT / 'next_batch_ranked.csv'}")
