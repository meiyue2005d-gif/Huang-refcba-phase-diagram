#!/usr/bin/env python3
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

RESULTS = Path('results')
OUT = RESULTS / 'active_learning'
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 20260821

PHASES = ['soluble', 'llps', 'aggregation']


def as_num(row: pd.Series, names: Iterable[str]) -> float:
    for name in names:
        if name in row.index:
            try:
                value = float(row[name])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return float('nan')


def token_number(text: str) -> float:
    return float(text.replace('p', '.'))


def parse_state(text: str):
    m = re.search(
        r'pH(?P<ph>\d+(?:p\d+)?)_nacl(?P<nacl>\d+(?:p\d+)?)_c(?P<c>\d+(?:p\d+)?)',
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    return token_number(m['ph']), token_number(m['nacl']), token_number(m['c'])


def explicit_phase(value: object) -> str | None:
    s = str(value).strip().lower()
    if not s or s == 'nan':
        return None
    if any(k in s for k in ['dynamic_condensation', 'llps_candidate', 'liquid_liquid', 'llps']):
        return 'llps'
    if any(k in s for k in ['arrested_aggregation', 'percolated_cluster', 'percolated_aggregation', 'aggregation_support']):
        return 'aggregation'
    if any(k in s for k in ['soluble_or_monomeric', 'finite_mobile_cluster', 'mostly_dispersed', 'soluble_monomeric']):
        return 'soluble'
    return None


def phase_from_metrics(row: pd.Series) -> str | None:
    for col in ['evidence', 'phase', 'final_phase']:
        if col in row.index:
            phase = explicit_phase(row[col])
            if phase:
                return phase

    clustered = as_num(row, ['clustered', 'mean_clustered_fraction'])
    mean_frac = as_num(row, ['mean', 'mean_largest_cluster_fraction'])
    max_frac = as_num(row, ['max', 'maximum_largest_cluster_fraction'])
    final_frac = as_num(row, ['final', 'final_largest_cluster_fraction'])
    dfm = as_num(row, ['dfm'])
    if not math.isfinite(dfm) and math.isfinite(final_frac) and math.isfinite(mean_frac):
        dfm = final_frac - mean_frac
    bond = as_num(row, ['bond', 'final_initial_bond_survival'])
    retain = as_num(row, ['retain', 'mean_consecutive_bond_retention'])
    perc = as_num(row, ['percolation', 'percolation_fraction'])
    slope = as_num(row, ['slope', 'late_slope_particles_per_ns', 'last10_slope_particles_per_ns'])

    # Some old summaries only contain the largest cluster in particles.
    if not math.isfinite(mean_frac):
        mean_size = as_num(row, ['mean_largest_cluster_size'])
        if math.isfinite(mean_size):
            mean_frac = mean_size / 500.0
    if not math.isfinite(final_frac):
        final_size = as_num(row, ['final_largest_cluster_size'])
        if math.isfinite(final_size):
            final_frac = final_size / 500.0
    if not math.isfinite(max_frac):
        max_size = as_num(row, ['maximum_largest_cluster_size', 'max_largest_cluster_size'])
        if math.isfinite(max_size):
            max_frac = max_size / 500.0
        elif math.isfinite(final_frac):
            max_frac = final_frac
    if not math.isfinite(dfm) and math.isfinite(final_frac) and math.isfinite(mean_frac):
        dfm = final_frac - mean_frac

    soluble = (
        math.isfinite(clustered) and clustered < 0.10
        and math.isfinite(max_frac) and max_frac < 0.02
    )
    percolated = math.isfinite(perc) and perc > 0.50
    arrested = (
        math.isfinite(clustered) and clustered >= 0.50
        and math.isfinite(bond) and bond >= 0.80
        and math.isfinite(retain) and retain >= 0.95
    )
    dynamic = (
        math.isfinite(max_frac) and max_frac >= 0.14
        and math.isfinite(retain) and retain < 0.95
        and (
            (math.isfinite(slope) and slope >= 0.001)
            or (math.isfinite(dfm) and dfm >= 0.025)
        )
    )
    if soluble:
        return 'soluble'
    if percolated or arrested:
        return 'aggregation'
    if dynamic:
        return 'llps'
    if math.isfinite(clustered) or math.isfinite(max_frac):
        return 'soluble'  # finite mobile clusters are included in the one-phase soluble class
    return None


def source_weight(path: Path) -> float:
    s = str(path).lower()
    if '0p5ns' in s or '0.5ns' in s:
        return 0.08
    if '1ns' in s:
        return 0.12
    if '5ns' in s:
        return 0.30
    if '10ns' in s:
        return 0.60
    if '30ns' in s or 'metrics.csv' in s:
        return 1.00
    return 0.50


def collect_file(path: Path) -> list[dict]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    rows = []
    for _, row in df.iterrows():
        ph = as_num(row, ['pH', 'ph'])
        nacl = as_num(row, ['NaCl_mM', 'nacl_mM', 'nacl'])
        conc = as_num(row, ['concentration_mg_ml', 'concentration'])
        state_text = ' '.join(str(row.get(c, '')) for c in ['state', 'task', 'run', 'source']) + ' ' + str(path)
        if not (math.isfinite(ph) and math.isfinite(nacl) and math.isfinite(conc)):
            parsed = parse_state(state_text)
            if parsed:
                ph, nacl, conc = parsed
        if not (math.isfinite(ph) and math.isfinite(nacl) and math.isfinite(conc) and conc > 0):
            continue
        phase = phase_from_metrics(row)
        if phase not in PHASES:
            continue
        seed_match = re.search(r'seed(\d+)', state_text, re.IGNORECASE)
        seed = int(seed_match.group(1)) if seed_match else np.nan
        rows.append({
            'pH': ph,
            'NaCl_mM': nacl,
            'concentration_mg_ml': conc,
            'phase': phase,
            'seed': seed,
            'weight': source_weight(path),
            'source_file': str(path),
        })
    return rows


patterns = [
    '**/*metrics.csv',
    'key5_uniform30ns_summary.csv',
    'key_conditions_3seed_replicates.csv',
    'combined_fast_scan_summary.csv',
    'boundary_scan_45_summary.csv',
]
files = []
for pattern in patterns:
    files.extend(RESULTS.glob(pattern))
files = sorted(set(files))

records = []
for file in files:
    records.extend(collect_file(file))

train = pd.DataFrame(records)
if train.empty:
    raise SystemExit('没有收集到可训练数据。请确认已经生成 *_metrics.csv。')

# Remove exact duplicate rows caused by the same table being copied or included twice.
train = train.drop_duplicates(
    subset=['pH', 'NaCl_mM', 'concentration_mg_ml', 'phase', 'seed', 'source_file']
).reset_index(drop=True)

counts = train['phase'].value_counts()
missing = [p for p in PHASES if counts.get(p, 0) < 2]
if missing:
    raise SystemExit(f'以下相态训练样本不足：{missing}；先补可靠锚点再拟合。')

X = np.column_stack([
    train['pH'].to_numpy(float),
    train['NaCl_mM'].to_numpy(float) / 500.0,
    np.log10(train['concentration_mg_ml'].to_numpy(float)),
])
y = train['phase'].to_numpy(str)
w = train['weight'].to_numpy(float)

model = ExtraTreesClassifier(
    n_estimators=800,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model.fit(X, y, sample_weight=w)

# A diagnostic only; it is not used to claim thermodynamic accuracy.
min_class = int(counts.min())
if min_class >= 3:
    folds = min(5, min_class)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    pred = np.empty(len(y), dtype=object)
    for train_idx, test_idx in cv.split(X, y):
        fold_model = ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        fold_model.fit(X[train_idx], y[train_idx], sample_weight=w[train_idx])
        pred[test_idx] = fold_model.predict(X[test_idx])
    bal_acc = balanced_accuracy_score(y, pred)
else:
    bal_acc = float('nan')

# Candidate lattice. This is deliberately coarser than the numerical noise of a 500-particle system.
ph_grid = np.round(np.arange(3.0, 9.0001, 0.25), 5)
nacl_grid = np.arange(0.0, 500.1, 25.0)
conc_grid = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0])
mesh = np.array(np.meshgrid(ph_grid, nacl_grid, conc_grid, indexing='ij'))
candidates = pd.DataFrame({
    'pH': mesh[0].ravel(),
    'NaCl_mM': mesh[1].ravel(),
    'concentration_mg_ml': mesh[2].ravel(),
})
XC = np.column_stack([
    candidates['pH'].to_numpy(float),
    candidates['NaCl_mM'].to_numpy(float) / 500.0,
    np.log10(candidates['concentration_mg_ml'].to_numpy(float)),
])
proba_raw = model.predict_proba(XC)
proba = np.zeros((len(candidates), 3), dtype=float)
for j, cls in enumerate(model.classes_):
    proba[:, PHASES.index(cls)] = proba_raw[:, j]

candidates['P_soluble'] = proba[:, 0]
candidates['P_LLPS'] = proba[:, 1]
candidates['P_aggregation'] = proba[:, 2]
eps = 1e-12
entropy = -(proba * np.log(proba + eps)).sum(axis=1) / math.log(3.0)
candidates['entropy'] = entropy
candidates['score_soluble_aggregation'] = 4.0 * proba[:, 0] * proba[:, 2] * (1.0 - proba[:, 1])
candidates['score_LLPS_aggregation'] = 4.0 * proba[:, 1] * proba[:, 2] * (1.0 - proba[:, 0])

# Exclude points already simulated at the same coordinates.
keys = set(zip(
    np.round(train['pH'], 5),
    np.round(train['NaCl_mM'], 5),
    np.round(train['concentration_mg_ml'], 5),
))
candidates['already_sampled'] = [
    (round(r.pH, 5), round(r.NaCl_mM, 5), round(r.concentration_mg_ml, 5)) in keys
    for r in candidates.itertuples(index=False)
]
candidates = candidates[~candidates['already_sampled']].copy()

# Restrict to points with meaningful aggregation probability; otherwise the selector can waste runs deep in soluble space.
candidates = candidates[candidates['P_aggregation'] >= 0.10].copy()


def scaled_point(row):
    return np.array([
        (float(row['pH']) - 3.0) / 6.0,
        float(row['NaCl_mM']) / 500.0,
        (math.log10(float(row['concentration_mg_ml'])) + 1.0) / (math.log10(20.0) + 1.0),
    ])


def diverse_select(df: pd.DataFrame, score_col: str, n: int, existing: list[np.ndarray]) -> list[pd.Series]:
    chosen = []
    pool = df.sort_values(score_col, ascending=False)
    for _, row in pool.iterrows():
        p = scaled_point(row)
        if any(np.linalg.norm(p - q) < 0.12 for q in existing + [scaled_point(x) for x in chosen]):
            continue
        chosen.append(row)
        if len(chosen) >= n:
            break
    return chosen

selected_sa = diverse_select(candidates, 'score_soluble_aggregation', 4, [])
selected_la = diverse_select(candidates, 'score_LLPS_aggregation', 4, [scaled_point(x) for x in selected_sa])
selected = selected_sa + selected_la

ranked_rows = []
for target, rows in [('soluble-aggregation', selected_sa), ('LLPS-aggregation', selected_la)]:
    for row in rows:
        d = row.to_dict()
        d['boundary_target'] = target
        d['boundary_score'] = d['score_soluble_aggregation'] if target == 'soluble-aggregation' else d['score_LLPS_aggregation']
        ranked_rows.append(d)
ranked = pd.DataFrame(ranked_rows).sort_values(['boundary_target', 'boundary_score'], ascending=[True, False])

train.to_csv(OUT / 'phase_training_records.csv', index=False)
candidates.sort_values('entropy', ascending=False).to_csv(OUT / 'candidate_probability_grid.csv', index=False)
ranked.to_csv(OUT / 'next_batch_ranked.csv', index=False)

with (OUT / 'next_batch.tsv').open('w', encoding='utf-8') as f:
    f.write('# ph\tnacl_mM\tconcentration_mg_ml\n')
    for row in ranked.itertuples(index=False):
        f.write(f'{row.pH:.5g}\t{row.NaCl_mM:.5g}\t{row.concentration_mg_ml:.5g}\n')

print('===== TRAINING DATA =====')
print(f'Rows: {len(train)}')
print(train['phase'].value_counts().to_string())
print(f'Balanced CV accuracy (diagnostic): {bal_acc:.3f}' if math.isfinite(bal_acc) else 'Balanced CV accuracy: unavailable')
print('\n===== NEXT BATCH =====')
show = ranked[[
    'boundary_target', 'pH', 'NaCl_mM', 'concentration_mg_ml',
    'P_soluble', 'P_LLPS', 'P_aggregation', 'entropy', 'boundary_score'
]]
print(show.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print(f'\nSaved: {OUT / "next_batch.tsv"}')
print(f'Saved: {OUT / "next_batch_ranked.csv"}')
