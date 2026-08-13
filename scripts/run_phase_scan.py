#!/usr/bin/env python3
"""Resume-capable orchestrator around the project's existing HOOMD scripts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huang_md.phase_scan import PhaseScanConfig  # noqa: E402


def command(python: str, script: str, *args: object) -> list[str]:
    return [python, str(ROOT / "scripts" / script), *(str(value) for value in args)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase_scan.yaml")
    parser.add_argument("--analysis-python", default=sys.executable)
    parser.add_argument("--hoomd-python", default=sys.executable)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--equil-steps", type=int, default=None)
    parser.add_argument("--prod-steps", type=int, default=None)
    parser.add_argument("--report-interval", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = PhaseScanConfig.from_yaml(args.config)
    equil_steps = args.equil_steps or config.coarse["equil_steps"]
    prod_steps = args.prod_steps or config.coarse["prod_steps"]
    report_interval = args.report_interval or config.coarse["report_interval_steps"]

    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if args.task_index is not None:
        rows = [rows[args.task_index]]
    if args.max_tasks is not None:
        rows = rows[: args.max_tasks]

    for ordinal, row in enumerate(rows, start=1):
        run_dir = Path(row["output_dir"]).resolve()
        done = run_dir / "dynamics_analysis" / "dynamics_summary.json"
        if done.exists() and not args.force:
            print(f"[{ordinal}/{len(rows)}] skip complete {row['state_id']}")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        commands = [
            command(
                args.analysis_python, "export_hoomd_state_input.py",
                "--ph", row["ph"], "--nacl-mM", row["nacl_mM"],
                "--concentration-mg-ml", row["concentration_mg_ml"],
                "--seed", row["seed"], "--state-config", row["state_config"],
                "--md-config", row["md_config"], "--output-dir", run_dir,
            ),
            command(
                args.hoomd_python, "run_single_state_hoomd.py",
                "--input-dir", run_dir, "--output-dir", run_dir,
                "--equil-steps", equil_steps, "--prod-steps", prod_steps,
                "--report-interval", report_interval,
            ),
            command(args.analysis_python, "analyze_trajectory.py", "--input-dir", run_dir),
            command(args.analysis_python, "analyze_state_dynamics.py", "--input-dir", run_dir),
        ]
        print(f"[{ordinal}/{len(rows)}] {row['state_id']}")
        for item in commands:
            print("  ", subprocess.list2cmdline(item))
            if not args.dry_run:
                subprocess.run(item, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
