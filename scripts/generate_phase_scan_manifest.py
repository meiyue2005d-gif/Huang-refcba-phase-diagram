#!/usr/bin/env python3
"""Generate the configured pH x NaCl x concentration screening manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huang_md.phase_scan import PhaseScanConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase_scan.yaml")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = PhaseScanConfig.from_yaml(args.config)
    output = (args.output or config.output_root / "coarse_manifest.tsv").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(config.manifest_rows())
    table.to_csv(output, sep="\t", index=False)
    print(f"States: {len(table)}")
    print(f"Grid  : {len(config.pH_values)} x {len(config.nacl_mM_values)} x "
          f"{len(config.concentration_mg_ml_values)} x {len(config.seeds)}")
    print(f"File  : {output}")


if __name__ == "__main__":
    main()
