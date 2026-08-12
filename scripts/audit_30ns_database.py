#!/usr/bin/env python3

import csv
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path("results")
DB = ROOT / "trusted_database_20260731"

INPUT = DB / "direct_metric_records_deduplicated.csv"
AUDITED = DB / "direct_metric_records_duration_audited.csv"
CONDITIONS = DB / "condition_consensus_30ns.csv"
REPORT = DB / "duration_audit_report.txt"

LONG_THRESHOLD_NS = 29.0


def maximum_numeric_column(path, column):
    maximum = None

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    value = float(row[column])
                except (KeyError, TypeError, ValueError):
                    continue

                if maximum is None or value > maximum:
                    maximum = value

    except Exception:
        return None

    return maximum


# 建立 run_id -> production_thermo.csv 索引
thermo_index = defaultdict(list)

for path in ROOT.rglob("production_thermo.csv"):
    if DB in path.parents:
        continue

    run_id = path.parent.name
    thermo_index[run_id].append(path)


with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
    records = list(csv.DictReader(f))

audited = []

for record in records:
    run_id = record["run_id"]

    durations = []
    thermo_files = thermo_index.get(run_id, [])

    for thermo in thermo_files:
        maximum_time_ps = maximum_numeric_column(thermo, "time_ps")

        if maximum_time_ps is not None:
            durations.append(maximum_time_ps / 1000.0)

    if durations:
        actual_duration_ns = max(durations)
        duration_status = (
            "long_30ns"
            if actual_duration_ns >= LONG_THRESHOLD_NS
            else "short"
        )
    else:
        actual_duration_ns = ""
        duration_status = "missing"

    updated = dict(record)
    updated["actual_duration_ns"] = actual_duration_ns
    updated["duration_status"] = duration_status
    updated["thermo_files_found"] = len(thermo_files)
    updated["thermo_file_paths"] = "|".join(
        str(path) for path in thermo_files
    )

    audited.append(updated)


audited_fields = list(audited[0].keys()) if audited else []

with AUDITED.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=audited_fields)
    writer.writeheader()
    writer.writerows(audited)


# 仅保留实际时长达到约30 ns的直接模拟
long_records = [
    record for record in audited
    if record["duration_status"] == "long_30ns"
]

groups = defaultdict(list)

for record in long_records:
    key = (
        float(record["pH"]),
        float(record["NaCl_mM"]),
        float(record["concentration_mg_ml"]),
    )
    groups[key].append(record)


condition_rows = []

for (pH, nacl, concentration), group in sorted(groups.items()):
    phases = sorted({
        record["phase_normalized"] for record in group
    })

    seeds = sorted({
        int(float(record["seed"])) for record in group
    })

    source_conflict = any(
        str(record.get("source_conflict", "0")) == "1"
        for record in group
    )

    if source_conflict:
        consensus = "uncertain_source_conflict"
        confidence = "low"

    elif len(phases) > 1:
        consensus = "transition_mixed"
        confidence = "boundary_mixed"

    else:
        consensus = phases[0]

        if len(seeds) >= 2:
            confidence = "high_reproducibility"
        else:
            confidence = "provisional_single_seed"

    if consensus == "llps_candidate":
        operational_phase = (
            "LLPS"
            if confidence == "high_reproducibility"
            else "provisional_LLPS"
        )
    elif consensus == "aggregation":
        operational_phase = (
            "aggregation"
            if confidence == "high_reproducibility"
            else "provisional_aggregation"
        )
    elif consensus == "soluble":
        operational_phase = (
            "soluble"
            if confidence == "high_reproducibility"
            else "provisional_soluble"
        )
    elif consensus == "transition_mixed":
        operational_phase = "boundary_transition"
    else:
        operational_phase = "uncertain"

    condition_rows.append({
        "pH": pH,
        "NaCl_mM": nacl,
        "concentration_mg_ml": concentration,
        "n_runs": len(group),
        "n_unique_seeds": len(seeds),
        "seeds": "|".join(str(seed) for seed in seeds),
        "seed_phase_values": "|".join(
            f'{record["seed"]}:{record["phase_normalized"]}'
            for record in sorted(
                group,
                key=lambda x: int(float(x["seed"]))
            )
        ),
        "consensus_phase": consensus,
        "operational_phase": operational_phase,
        "confidence": confidence,
        "minimum_actual_duration_ns": min(
            float(record["actual_duration_ns"])
            for record in group
        ),
        "maximum_actual_duration_ns": max(
            float(record["actual_duration_ns"])
            for record in group
        ),
        "needs_additional_seed": int(
            len(seeds) < 2
            and consensus != "transition_mixed"
        ),
        "evidence_summary": "|".join(sorted({
            record["evidence_raw"] for record in group
        })),
        "run_ids": "|".join(sorted({
            record["run_id"] for record in group
        })),
        "source_rounds": "|".join(sorted({
            record["source_round"] for record in group
        })),
    })


condition_fields = list(condition_rows[0].keys()) if condition_rows else []

with CONDITIONS.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=condition_fields)
    writer.writeheader()
    writer.writerows(condition_rows)


duration_counts = Counter(
    record["duration_status"] for record in audited
)

phase_counts = Counter(
    record["operational_phase"] for record in condition_rows
)

lines = [
    f"All unique direct runs: {len(audited)}",
    f"Runs reaching >= {LONG_THRESHOLD_NS} ns: "
    f"{duration_counts.get('long_30ns', 0)}",
    f"Short runs: {duration_counts.get('short', 0)}",
    f"Runs with missing duration: "
    f"{duration_counts.get('missing', 0)}",
    f"Unique 30 ns conditions: {len(condition_rows)}",
    "",
    "Operational phase counts:",
]

for phase, count in sorted(phase_counts.items()):
    lines.append(f"  {phase}: {count}")

lines.extend([
    "",
    "Short or missing-duration runs:",
])

for record in audited:
    if record["duration_status"] != "long_30ns":
        lines.append(
            f'  {record["run_id"]}\t'
            f'{record["duration_status"]}\t'
            f'{record["actual_duration_ns"]}\t'
            f'{record["source_file"]}'
        )

report = "\n".join(lines) + "\n"

REPORT.write_text(report, encoding="utf-8")

print(report)
