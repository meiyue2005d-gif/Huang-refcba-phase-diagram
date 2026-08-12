# State-model versions and data compatibility

## `legacy_refcba_abs_charge_v1`

Configuration: `configs/refcba_state_model_legacy.yaml`.

This is the exact pH/salt mapping used to generate the existing refCBA-labelled
simulation archive. It is retained for reproducibility only. Its use of
`abs(net_charge)` creates a repulsive-amplitude rebound after the sequence pI.

Existing trajectories, thermodynamic logs and cluster metrics remain valid
observations of this legacy Hamiltonian. They may be reclassified with the
corrected evidence rules, but they cannot be relabelled as results of another
Hamiltonian.

## `huang_anchored_refcba_eq10_magnitude_v2`

Configuration: `configs/refcba_state_model.yaml`.

This default sensitivity model applies Huang et al. Eq. 10 to the magnitude of
the charge ratio. Two identical proteins therefore remain mutually repulsive
after both reverse charge sign. Crossing the sequence pI is flagged in metadata
as an uncalibrated charge-reversal extrapolation. The Huang A1 diameter and
attraction are still used, so this is not a quantitatively calibrated refCBA
model. Added NaCl remains an explicit extrapolation.

Only old runs whose stored K2/Z2 match this revised model are directly reusable.
Use `scripts/audit_legacy_data_against_model_v2.py` to identify them.

## `huang_a1_eq10_reproduction_v1`

Configurations: `configs/huang_a1_state_model.yaml` and
`configs/huang_a1_md.yaml`.

This route uses the full 350-aa Reflectin A1 sequence, the A1 molecular weight,
and Huang Eq. 10. Its declared reproduction range is pH 4.5-7.5 at the reference
buffer condition. Changing added NaCl is a model extension, not strict paper
reproduction.

## Required provenance for new simulations

The run entry points now write `state_model_id`, `charge_mapping`, `protein_id`,
configuration paths and applicability flags to metadata. Phase aggregation
should never combine rows with different `state_model_id` values.
