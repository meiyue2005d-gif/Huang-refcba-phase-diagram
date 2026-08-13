#!/usr/bin/env python3

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("results")
OUT = ROOT / "trusted_database_20260731"
OUT.mkdir(parents=True, exist_ok=True)

# 当前仍在运行，暂时排除
RUNNING_MARKERS = ()

METRIC_COLUMNS = {
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


def decode_number(token):
    if token is None:
        return None
    try:
        return float(token.replace("p", "."))
    except ValueError:
        return None


def read_header(path):
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            return next(reader, [])
    except Exception:
        return []


def parse_run_id(run_id, source_path):
    text = str(run_id)
    full_path = str(source_path)

    ph_match = re.search(r"pH(\d+(?:p\d+)?)", text)
    c_match = re.search(r"_c(\d+(?:p\d+)?)", text)
    seed_match = re.search(r"_seed(\d+)", text)

    nacl_match = re.search(r"_nacl(\d+(?:p\d+)?)", text)
    if nacl_match is None:
        nacl_match = re.search(r"nacl(\d+(?:p\d+)?)", full_path)

    pH = decode_number(ph_match.group(1)) if ph_match else None
    nacl = decode_number(nacl_match.group(1)) if nacl_match else None
    concentration = decode_number(c_match.group(1)) if c_match else None
    seed = int(seed_match.group(1)) if seed_match else None

    return pH, nacl, concentration, seed


def infer_duration_ns(path):
    matches = re.findall(r"(\d+(?:p\d+)?)ns", str(path))
    durations = []

    for value in matches:
        parsed = decode_number(value)
        if parsed is not None:
            durations.append(parsed)

    return max(durations) if durations else None


def normalize_phase(evidence):
    text = str(evidence).strip().lower()

    if any(x in text for x in (
        "arrested",
        "aggregation",
        "percolat",
        "gel",
    )):
        return "aggregation"

    if any(x in text for x in (
        "soluble",
        "monomeric",
        "dispersed",
    )):
        return "soluble"

    if any(x in text for x in (
        "finite_mobile",
        "dynamic",
        "llps",
        "coexistence",
    )):
        return "llps_candidate"

    return "uncertain"


all_records = []
skipped_records = []
metric_files = []
header_counts = Counter()

for path in sorted(ROOT.rglob("*.csv")):
    path_text = str(path)

    if OUT in path.parents:
        continue

    if any(marker in path_text for marker in RUNNING_MARKERS):
        continue

    header = read_header(path)
    if not header:
        continue

    header_counts[",".join(header)] += 1

    if not METRIC_COLUMNS.issubset(set(header)):
        continue

    metric_files.append(path)

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)

            for row_number, row in enumerate(reader, start=2):
                run_id = str(row.get("state", "")).strip()

                pH, nacl, concentration, seed = parse_run_id(run_id, path)

                if (
                    not run_id
                    or pH is None
                    or nacl is None
                    or concentration is None
                    or seed is None
                ):
                    skipped_records.append({
                        "source_file": str(path),
                        "row_number": row_number,
                        "run_id": run_id,
                        "reason": "cannot_parse_condition_or_seed",
                    })
                    continue

                evidence = str(row.get("evidence", "")).strip()
                duration = infer_duration_ns(path)

                relative = path.relative_to(ROOT)
                source_round = (
                    relative.parts[0]
                    if len(relative.parts) > 1
                    else path.stem
                )

                all_records.append({
                    "run_id": run_id,
                    "pH": pH,
                    "NaCl_mM": nacl,
                    "concentration_mg_ml": concentration,
                    "seed": seed,
                    "nominal_duration_ns": (
                        duration if duration is not None else ""
                    ),
                    "phase_normalized": normalize_phase(evidence),
                    "evidence_raw": evidence,
                    "clustered": row.get("clustered", ""),
                    "mean": row.get("mean", ""),
                    "max": row.get("max", ""),
                    "final": row.get("final", ""),
                    "dfm": row.get("dfm", ""),
                    "bond": row.get("bond", ""),
                    "retain": row.get("retain", ""),
                    "percolation": row.get("percolation", ""),
                    "slope": row.get("slope", ""),
                    "source_round": source_round,
                    "source_file": str(path),
                })

    except Exception as exc:
        skipped_records.append({
            "source_file": str(path),
            "row_number": "",
            "run_id": "",
            "reason": f"read_error:{type(exc).__name__}",
        })


raw_fields = [
    "run_id",
    "pH",
    "NaCl_mM",
    "concentration_mg_ml",
    "seed",
    "nominal_duration_ns",
    "phase_normalized",
    "evidence_raw",
    "clustered",
    "mean",
    "max",
    "final",
    "dfm",
    "bond",
    "retain",
    "percolation",
    "slope",
    "source_round",
    "source_file",
]

with (OUT / "direct_metric_records_all.csv").open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(f, fieldnames=raw_fields)
    writer.writeheader()
    writer.writerows(all_records)


# 同一 run_id 可能出现在多个汇总文件中，先按 run_id 去重
records_by_run = defaultdict(list)
for record in all_records:
    records_by_run[record["run_id"]].append(record)

deduplicated_runs = []

for run_id, records in sorted(records_by_run.items()):
    phase_values = sorted({
        record["phase_normalized"] for record in records
    })

    def duration_value(record):
        value = record["nominal_duration_ns"]
        return float(value) if value != "" else -1.0

    selected = max(records, key=duration_value).copy()

    selected["phase_normalized"] = (
        phase_values[0] if len(phase_values) == 1 else "uncertain"
    )
    selected["source_conflict"] = int(len(phase_values) > 1)
    selected["all_phase_values"] = "|".join(phase_values)
    selected["all_evidence_values"] = "|".join(sorted({
        record["evidence_raw"] for record in records
    }))
    selected["all_source_files"] = "|".join(sorted({
        record["source_file"] for record in records
    }))
    selected["n_source_files"] = len({
        record["source_file"] for record in records
    })

    deduplicated_runs.append(selected)

dedup_fields = raw_fields + [
    "source_conflict",
    "all_phase_values",
    "all_evidence_values",
    "all_source_files",
    "n_source_files",
]

with (OUT / "direct_metric_records_deduplicated.csv").open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=dedup_fields,
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(deduplicated_runs)


# 按 pH × NaCl × concentration 汇总不同种子
condition_groups = defaultdict(list)

for record in deduplicated_runs:
    key = (
        record["pH"],
        record["NaCl_mM"],
        record["concentration_mg_ml"],
    )
    condition_groups[key].append(record)

condition_rows = []

for condition, records in sorted(condition_groups.items()):
    pH, nacl, concentration = condition

    phases = sorted({
        record["phase_normalized"] for record in records
    })
    seeds = sorted({
        int(record["seed"]) for record in records
    })

    has_source_conflict = any(
        int(record["source_conflict"]) == 1 for record in records
    )

    if has_source_conflict or "uncertain" in phases:
        consensus = "uncertain"
        confidence = "low_source_conflict"

    elif len(phases) > 1:
        consensus = "uncertain"
        confidence = "low_seed_conflict"

    else:
        consensus = phases[0]

        if len(seeds) >= 2:
            confidence = "high_reproducibility"
        else:
            confidence = "provisional_single_seed"

    durations = []
    for record in records:
        value = record["nominal_duration_ns"]
        if value != "":
            durations.append(float(value))

    condition_rows.append({
        "pH": pH,
        "NaCl_mM": nacl,
        "concentration_mg_ml": concentration,
        "n_runs": len(records),
        "n_unique_seeds": len(seeds),
        "seeds": "|".join(str(x) for x in seeds),
        "run_ids": "|".join(sorted(
            record["run_id"] for record in records
        )),
        "seed_phase_values": "|".join(
            f'{record["seed"]}:{record["phase_normalized"]}'
            for record in sorted(records, key=lambda x: int(x["seed"]))
        ),
        "consensus_phase": consensus,
        "confidence": confidence,
        "needs_more_seed": int(
            len(seeds) < 2 or consensus == "uncertain"
        ),
        "needs_llps_validation": int(
            consensus == "llps_candidate"
        ),
        "minimum_nominal_duration_ns": (
            min(durations) if durations else ""
        ),
        "maximum_nominal_duration_ns": (
            max(durations) if durations else ""
        ),
        "evidence_summary": "|".join(sorted({
            record["evidence_raw"] for record in records
        })),
        "source_rounds": "|".join(sorted({
            record["source_round"] for record in records
        })),
    })

condition_fields = [
    "pH",
    "NaCl_mM",
    "concentration_mg_ml",
    "n_runs",
    "n_unique_seeds",
    "seeds",
    "run_ids",
    "seed_phase_values",
    "consensus_phase",
    "confidence",
    "needs_more_seed",
    "needs_llps_validation",
    "minimum_nominal_duration_ns",
    "maximum_nominal_duration_ns",
    "evidence_summary",
    "source_rounds",
]

with (OUT / "condition_consensus_initial.csv").open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(f, fieldnames=condition_fields)
    writer.writeheader()
    writer.writerows(condition_rows)


with (OUT / "metric_files_used.txt").open(
    "w", encoding="utf-8"
) as f:
    for path in metric_files:
        f.write(str(path) + "\n")


with (OUT / "skipped_metric_rows.csv").open(
    "w", encoding="utf-8", newline=""
) as f:
    fields = ["source_file", "row_number", "run_id", "reason"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(skipped_records)


with (OUT / "header_signatures.csv").open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.writer(f)
    writer.writerow(["file_count", "header"])
    for header, count in header_counts.most_common():
        writer.writerow([count, header])


phase_counts = Counter(
    row["consensus_phase"] for row in condition_rows
)
confidence_counts = Counter(
    row["confidence"] for row in condition_rows
)

report_lines = [
    f"CSV files scanned: {sum(header_counts.values())}",
    f"metric summary files used: {len(metric_files)}",
    f"metric rows found: {len(all_records)}",
    f"unique run IDs: {len(deduplicated_runs)}",
    f"unique conditions: {len(condition_rows)}",
    f"skipped metric rows: {len(skipped_records)}",
    "",
    "Condition phase counts:",
]

for key, value in sorted(phase_counts.items()):
    report_lines.append(f"  {key}: {value}")

report_lines.append("")
report_lines.append("Condition confidence counts:")

for key, value in sorted(confidence_counts.items()):
    report_lines.append(f"  {key}: {value}")

report = "\n".join(report_lines) + "\n"

with (OUT / "build_report.txt").open(
    "w", encoding="utf-8"
) as f:
    f.write(report)

print(report, end="")
