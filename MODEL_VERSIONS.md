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

## `huang_anchored_refcba_gouy_chapman_salt_v3` (default)

Configuration: `configs/refcba_state_model.yaml`.

This default sensitivity model applies Huang et al. Eq. 10 to the magnitude of
the charge ratio. Its NaCl extension follows the Gouy-Chapman/Grahame relation
by placing the square-root ionic-strength factor inside the `asinh` argument;
the old post-hoc power-law amplitude multiplier is disabled. Two identical
proteins remain mutually repulsive after both reverse charge sign. Crossing the
sequence pI is flagged as an uncalibrated charge-reversal extrapolation. The
Huang A1 diameter and attraction are still used, so this is not a quantitative
refCBA calibration. Every added-NaCl state remains explicitly marked as an
extrapolation because Huang fitted a fixed reference buffer rather than a
0--500 mM salt series.

The code intentionally preserves Huang's numerical convention in which the
Table S1 value `K2=53.056` is the Yukawa amplitude and the Eq. 10 mapping uses
the thermal-voltage scale. The v3 salt term extends the same convention; it is
not an independent conversion of the fitted amplitude into SI surface volts.

The Gaussian-core width (`sigma=0.35`) follows the paper. Its epsilon is not
tabulated in the article or supplement available to this project; the configured
`1e5 kBT` reproduces the reported small well shift and is therefore a numerical
reconstruction, not a uniquely identified experimental parameter. Sensitivity
to this value should be reported for quantitative claims.

## `huang_anchored_refcba_eq10_magnitude_v2` (historical revised model)

The previous default multiplied Eq. 10 by an empirical ionic-strength power
after applying the charge mapping. It remains represented by old metadata and
must not be mixed with v3. The v2 trajectories are observations of a different
Hamiltonian and need rerunning for a v3 phase diagram.

Only runs whose stored model ID and K2/Z2 match v3 are directly reusable for the
new map. The legacy audit remains useful for inventory, not for silently
relabeling v1/v2 trajectories as v3.

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
