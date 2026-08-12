#!/usr/bin/env python3
"""Run and analyze all states in a pilot scan manifest."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            ROOT
            / "results"
            / "pilot_scan_nacl0"
            / "pilot_manifest.csv"
        ),
    )

    parser.add_argument(
        "--equil-steps",
        type=int,
        default=200000,
    )

    parser.add_argument(
        "--prod-steps",
        type=int,
        default=1000000,
    )

    parser.add_argument(
        "--report-interval",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Only run this many pending tasks.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun tasks even if analysis is complete.",
    )

    return parser


def run_command(
    command: list[str],
    log_handle,
) -> None:
    log_handle.write("\n$ " + " ".join(command) + "\n")
    log_handle.flush()

    process = subprocess.run(
        command,
        cwd=ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code "
            f"{process.returncode}: {' '.join(command)}"
        )


def main() -> None:
    args = build_parser().parse_args()

    manifest = args.manifest.resolve()

    if not manifest.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest}"
        )

    table = pd.read_csv(manifest)

    pending_count = 0
    completed_count = 0
    failed_records: list[dict[str, object]] = []

    total_start = time.time()

    for _, row in table.iterrows():
        task_id = int(row["task_id"])
        output_dir = Path(str(row["output_dir"])).resolve()

        dynamics_summary = (
            output_dir
            / "dynamics_analysis"
            / "dynamics_summary.json"
        )

        if dynamics_summary.exists() and not args.force:
            print(
                f"SKIP task {task_id:03d}: already complete"
            )
            completed_count += 1
            continue

        if (
            args.max_tasks is not None
            and pending_count >= args.max_tasks
        ):
            break

        pending_count += 1
        output_dir.mkdir(parents=True, exist_ok=True)

        log_file = output_dir / "run.log"

        print(
            f"\n[{pending_count}] task {task_id:03d}: "
            f"pH={float(row['pH']):.4f}, "
            f"NaCl={float(row['nacl_mM']):.1f} mM, "
            f"c={float(row['concentration_mg_ml']):.4f} mg/mL"
        )

        start = time.time()

        try:
            with log_file.open(
                "a",
                encoding="utf-8",
            ) as log_handle:
                run_command(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "run_single_state.py"),
                        "--ph",
                        str(float(row["pH"])),
                        "--nacl-mM",
                        str(float(row["nacl_mM"])),
                        "--concentration-mg-ml",
                        str(float(row["concentration_mg_ml"])),
                        "--equil-steps",
                        str(args.equil_steps),
                        "--prod-steps",
                        str(args.prod_steps),
                        "--minimize-max-iterations",
                        "1000",
                        "--report-interval",
                        str(args.report_interval),
                        "--seed",
                        str(int(row["seed"])),
                        "--output-dir",
                        str(output_dir),
                    ],
                    log_handle,
                )

                run_command(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "analyze_trajectory.py"),
                        "--input-dir",
                        str(output_dir),
                    ],
                    log_handle,
                )

                run_command(
                    [
                        sys.executable,
                        str(
                            ROOT
                            / "scripts"
                            / "analyze_state_dynamics.py"
                        ),
                        "--input-dir",
                        str(output_dir),
                    ],
                    log_handle,
                )

            elapsed = time.time() - start
            completed_count += 1

            print(
                f"PASS task {task_id:03d}, "
                f"elapsed={elapsed:.1f} s"
            )

        except Exception as error:
            elapsed = time.time() - start

            print(
                f"FAIL task {task_id:03d}, "
                f"elapsed={elapsed:.1f} s: {error}"
            )

            failed_records.append(
                {
                    "task_id": task_id,
                    "pH": float(row["pH"]),
                    "nacl_mM": float(row["nacl_mM"]),
                    "concentration_mg_ml": float(
                        row["concentration_mg_ml"]
                    ),
                    "output_dir": str(output_dir),
                    "error": str(error),
                }
            )

    if failed_records:
        failed_file = (
            manifest.parent / "failed_tasks.csv"
        )

        pd.DataFrame.from_records(
            failed_records
        ).to_csv(
            failed_file,
            index=False,
        )

        print(f"\nFailed task list: {failed_file}")

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 72)
    print("Pilot batch finished")
    print("=" * 72)
    print(f"Tasks started  : {pending_count}")
    print(f"Completed      : {completed_count}")
    print(f"Failed         : {len(failed_records)}")
    print(f"Elapsed        : {total_elapsed:.1f} s")


if __name__ == "__main__":
    main()
