from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("finalize_phase_scan", ROOT / "scripts" / "finalize_phase_scan.py")
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_consensus_requires_two_matching_seeds() -> None:
    assert MODULE.consensus(["llps", "llps"], 2) == ("llps", "2_seed_consensus")
    assert MODULE.consensus(["llps"], 2)[0] == "unresolved"
    assert MODULE.consensus(["llps", "aggregate"], 2)[0] == "unresolved"


def test_empty_manifest_is_safe(tmp_path: Path) -> None:
    path = tmp_path / "empty.tsv"
    path.write_text("", encoding="utf-8")
    assert MODULE.read_manifest(path).empty
