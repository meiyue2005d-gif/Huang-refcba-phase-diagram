#!/usr/bin/env python3
"""Run a resume-capable HOOMD direct-coexistence manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def cmd(python: str, script: str, *values: object) -> list[str]:
    return [python, str(ROOT / "scripts" / script), *(str(value) for value in values)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--analysis-python", default=sys.executable)
    parser.add_argument("--hoomd-python", default=sys.executable)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--equil-steps", type=int, default=500000)
    parser.add_argument("--prod-steps", type=int, default=30000000)
    parser.add_argument("--report-interval", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if args.task_index is not None:
        rows = [rows[args.task_index]]
    if args.max_tasks is not None:
        rows = rows[: args.max_tasks]

    for ordinal, row in enumerate(rows, start=1):
        output = Path(row["output_dir"]).resolve()
        done = output / "slab_dynamics_summary.json"
        if done.exists() and not args.force:
            print(f"[{ordinal}/{len(rows)}] skip complete {row['state_id']}")
            continue
        output.mkdir(parents=True, exist_ok=True)
        commands = [
            cmd(
                args.analysis_python, "export_hoomd_slab_input.py",
                "--ph", row["ph"], "--nacl-mM", row["nacl_mM"],
                "--global-concentration-mg-ml", row["global_concentration_mg_ml"],
                "--initial-slab-concentration-mg-ml", row["initial_slab_concentration_mg_ml"],
                "--z-aspect-ratio", row["z_aspect_ratio"], "--seed", row["seed"],
                "--state-config", row["state_config"], "--md-config", row["md_config"],
                "--output-dir", output,
            ),
            cmd(
                args.hoomd_python, "run_direct_coexistence_hoomd.py",
                "--input-dir", output, "--output-dir", output,
                "--equil-steps", args.equil_steps, "--prod-steps", args.prod_steps,
                "--report-interval", args.report_interval,
            ),
            cmd(args.analysis_python, "analyze_direct_coexistence.py", "--input-dir", output),
            cmd(args.analysis_python, "analyze_slab_dynamics.py", "--input-dir", output),
        ]
        print(f"[{ordinal}/{len(rows)}] {row['state_id']}")
        for item in commands:
            print("  ", subprocess.list2cmdline(item))
            if not args.dry_run:
                subprocess.run(item, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
