# Huang–refCBA phase-diagram model

Coarse-grained short-range-attraction/long-range-repulsion simulations for
reflectin-inspired charged colloids, based on the two-Yukawa plus Gaussian-core
framework of Huang et al. (2024), *Biophysical Journal*.

This repository contains source code, versioned model configurations, tests,
scan manifests, and analysis scripts. Production trajectories and bulk result
archives are intentionally not stored in Git.

## Scientific scope

Three state-model configurations are kept separate:

- `configs/refcba_state_model_legacy.yaml`: exact legacy refCBA charge mapping,
  retained only to reproduce existing simulations.
- `configs/refcba_state_model.yaml`: Huang-anchored refCBA sensitivity model
  using the nonlinear Gouy–Chapman mapping in Huang Eq. 10.
- `configs/huang_a1_state_model.yaml`: full-length Reflectin A1 sequence and
  Huang Eq. 10 for the paper-reproduction path. Pair with
  `configs/huang_a1_md.yaml`.

The refCBA configurations reuse A1 diameter and attraction parameters and are
therefore sensitivity models, not experimentally calibrated quantitative refCBA
models. Added NaCl scans beyond the reference buffer are also extrapolations.
See [MODEL_VERSIONS.md](MODEL_VERSIONS.md) for details and data-compatibility
rules.

## Repository layout

```text
huang_md/   Core potentials, electrostatics, thermodynamics, clustering
configs/    Versioned potential, state-model, and MD configurations
scripts/    Simulation, validation, classification, and plotting entry points
tests/      Numerical and classification tests
manifests/  Small tab-separated scan definitions
data/       Reference sequence and Lennard-Jones validation data
results/    Local outputs; only its README is versioned
```

## Installation

The tested production workflow targets Linux with an NVIDIA GPU. Create the
analysis/OpenMM environment with Conda:

```bash
conda env create -f environment.yml
conda activate huang-refcba
```

HOOMD-blue is best isolated in its own environment:

```bash
conda env create -f environment-hoomd.yml
conda activate huang-refcba-hoomd
```

GPU availability and exact package builds depend on the host CUDA driver.

## Quick validation

From the repository root:

```bash
pytest -q
python scripts/validate_potential.py
python scripts/validate_state_model.py
```

The liquid-state perturbation implementation includes an applicability
diagnostic. An `uncontrolled` result must not be interpreted as a quantitative
binodal.

## Running a state

Export a HOOMD input using the default revised refCBA sensitivity model:

```bash
python scripts/export_hoomd_state_input.py \
  --ph 5.5 \
  --nacl-mM 100 \
  --concentration-mg-ml 10 \
  --output-dir results/example_state
```

For the strict A1 path, explicitly select both matching configurations:

```bash
python scripts/export_hoomd_state_input.py \
  --ph 5.5 \
  --nacl-mM 0 \
  --concentration-mg-ml 10 \
  --state-config configs/huang_a1_state_model.yaml \
  --md-config configs/huang_a1_md.yaml \
  --output-dir results/a1_example
```

Every new run records the model identifier, protein identifier, charge mapping,
configuration paths, and extrapolation flags in its metadata. Do not merge runs
with different `state_model_id` values into one phase diagram.

## Phase classification

Ordinary homogeneous NVT clustering is sufficient to identify soluble or
arrested/percolated aggregation candidates. It is not sufficient by itself to
claim equilibrium LLPS. Conservative LLPS classification requires direct
coexistence evidence and dynamic exchange, ideally with multiple seeds and a
finite-size check.

The main aggregation entry point is:

```bash
python scripts/finalize_refcba_three_phase.py \
  --results-root results \
  --screening-csv results/<screen>/screening_classified.csv \
  --output-dir results/final_three_phase
```

## Existing data

The old simulation archive is not included. Locally, run:

```bash
python scripts/audit_legacy_data_against_model_v2.py
```

to determine whether a legacy trajectory exactly matches the revised
Hamiltonian or can only be used as legacy-model evidence.

## Reference

Huang et al., “A colloidal model for the equilibrium assembly and
liquid-liquid phase separation of the reflectin A1 protein,” *Biophysical
Journal* (2024). DOI: [10.1016/j.bpj.2024.05.029](https://doi.org/10.1016/j.bpj.2024.05.029).

## License

No open-source license has been selected yet. Until one is added, the source is
available for viewing but no permission to copy, modify, or redistribute is
granted.
