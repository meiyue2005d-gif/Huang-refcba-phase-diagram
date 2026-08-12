#!/usr/bin/env python3

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path("results")
DB = ROOT / "trusted_database_20260731"

INPUT = DB / "pH8p25_nacl100_full_concentration_audit.csv"
OUTPUT = DB / "pH8p25_nacl100_full_concentration_audit_repaired.csv"

with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
    audit_rows = list(csv.DictReader(f))

wanted_run_ids = {row["run_id"] for row in audit_rows}
matches = defaultdict(list)

required_columns = {
    "state",
    "clustered",
    "mean",
    "max",
    "final",
    "dfm",
    "bond",
    "retain",
    "percolation",
    "slope",
    "evidence",
}

for path in ROOT.rglob("*.csv"):
    if DB in path.parents:
        continue

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                continue

            if not required_columns.issubset(set(reader.fieldnames)):
                continue

            for row in reader:
                run_id = str(row.get("state", "")).strip()

                if run_id in wanted_run_ids:
                    matches[run_id].append({
                        "evidence": row.get("evidence", ""),
                        "metric_clustered": row.get("clustered", ""),
                        "metric_mean": row.get("mean", ""),
                        "metric_max": row.get("max", ""),
                        "metric_final": row.get("final", ""),
                        "metric_dfm": row.get("dfm", ""),
                        "metric_bond": row.get("bond", ""),
                        "metric_retain": row.get("retain", ""),
                        "metric_percolation": row.get("percolation", ""),
                        "metric_slope": row.get("slope", ""),
                        "metric_file": str(path),
                    })

    except Exception:
        continue

repaired = []

for row in audit_rows:
    run_id = row["run_id"]
    found = matches.get(run_id, [])

    evidence_values = sorted({
        item["evidence"] for item in found
        if item["evidence"]
    })

    metric_files = sorted({
        item["metric_file"] for item in found
    })

    updated = dict(row)
    updated["exact_evidence"] = (
        "|".join(evidence_values) if evidence_values else ""
    )
    updated["evidence_conflict"] = int(len(evidence_values) > 1)
    updated["matching_metric_rows"] = len(found)
    updated["matching_metric_files"] = "|".join(metric_files)

    if found:
        selected = found[-1]

        for key in [
            "metric_clustered",
            "metric_mean",
            "metric_max",
            "metric_final",
            "metric_dfm",
            "metric_bond",
            "metric_retain",
            "metric_percolation",
            "metric_slope",
        ]:
            updated[key] = selected[key]
    else:
        for key in [
            "metric_clustered",
            "metric_mean",
            "metric_max",
            "metric_final",
            "metric_dfm",
            "metric_bond",
            "metric_retain",
            "metric_percolation",
            "metric_slope",
        ]:
            updated[key] = ""

    repaired.append(updated)

fields = list(repaired[0].keys())

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(repaired)

print(
    "c\tseed\tlargest_frac\tweight_avg\t"
    "old_label\texact_evidence\tmatches\tconflict"
)

for row in repaired:
    print(
        row["concentration_mg_ml"],
        row["seed"],
        row["mean_largest_cluster_fraction"],
        row["mean_weight_average_cluster_size"],
        row["old_evidence"],
        row["exact_evidence"],
        row["matching_metric_rows"],
        row["evidence_conflict"],
        sep="\t",
    )

print()
print("Saved:", OUTPUT)
