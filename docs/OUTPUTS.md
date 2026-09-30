# Output dictionary

## Step 3 – `$CATD_NCT_RESULTS_DIR/sub-XXXXX/`

| Path | Content |
|---|---|
| `sub-XXXXX_melodic_IC_FSLMNI2mm.nii.gz` | 4-D ICs registered to adult MNI space |
| `ica_components_3d/sub-XXXXX_ICnn.nii.gz` | One 3-D volume per component (`nn` 0-based) |
| `sub-XXXXX_ICnn/network_correspondence.csv` | NCT output for component `nn`; the table used by steps 04 and 06. Columns used: `group` (atlas), `name` (network label), `dice`, `p_value` |
| `sub-XXXXX_all_components_dice.tsv` | Only written if NCT produced `*summary*.csv` files; not used downstream |

## Step 4 – `$CATD_ASSIGNMENTS_DIR/`

**`subject_network_assignments.csv`** – one row per participant

| Column | Meaning |
|---|---|
| `subject_id` | `sub-XXXXX` |
| `DMN`, `Salience`, `Frontoparietal`, `DorsalAttention`, `Visual`, `Somatomotor` | 0-based IC number of the winning component (empty = none). Note that the same IC can win more than one network here; step 06 resolves this for DMN/Salience/Frontoparietal |
| `<network>_is_fallback` | `True` if no IC cleared the primary gate and the permissive fallback gate was used |

**`network_assignment_details.csv`** – one row per participant × network with an assignment

`subject_id, network, ic_num, score, dice_mean, dice_best, p_value_mean, p_value_best, n_atlases, is_fallback`
(`dice_best`/`p_value_best` are the maximum Dice / minimum p across matched atlas networks;
`n_atlases` is the number of atlases, out of 4, with a matching network.)

**`network_fragmentation_counts.csv`** – one row per participant

| Column | Meaning |
|---|---|
| `n_significant_<network>` | Number of candidate ICs (primary gate). 0 = none, 1 = clean, >1 = fragmented |
| `significant_<network>_ICs` | The candidates as `IC<n>(score=<s>);…` |

## Step 6 – `$CATD_MASKS_DIR/sub-XXXXX/` and the metrics table

Masks: `sub-XXXXX_<Network>_mask.nii.gz` (primary), `…_<Network>_b_mask.nii.gz` (secondary),
`…_<Network>_c_mask.nii.gz` (tertiary); binary (uint8), MNIPediatricAsym 2 mm grid. Step 07
additionally writes `sub-XXXXX_<NET>_combined_mask.nii.gz` (primary ∪ secondary).

**Metrics TSV** (`dualreg_nct_metrics.tsv`) – one row per participant × run

`subject_id`, `session`, then for each network *N* and tier *T*:

| Column | Meaning |
|---|---|
| `N_mean`, `N_var`, `N_std` | Mean / variance / SD of the dual-regression stage-2 values inside the mask |
| `N_largest_cluster` | Voxel count of the largest connected cluster of stage-2 values ≥ 2.3 within the mask |

Tiers: no suffix = primary, `_b` = secondary, `_c` = tertiary. Secondary/tertiary exist for
`DMN`, `Salience`, `Frontoparietal` only (e.g. `DMN_b_mean`). Values are empty (NaN) when no
component was available for that network/tier.

## Step 7 – `between_network_connectivity.csv` – one row per participant × run

| Column | Meaning |
|---|---|
| `subject`, `session` | `sub-XXXXX`, `ses-…` |
| `DMN_SN_r`, `DMN_FPN_r`, `SN_FPN_r` | Pearson r between network-mean time series (primary masks) |
| `DMN_SN_Z`, `DMN_FPN_Z`, `SN_FPN_Z` | Fisher z of the above |
| `…_r_comb`, `…_Z_comb` | Same, using combined (primary ∪ secondary) masks |
| `DMN_mask_combined`, `SN_mask_combined`, `FPN_mask_combined` | `True` = a secondary mask existed and the combined mask is a true union; `False` = fell back to the primary mask alone |

## Step 8 – `yeo_coherence_results.csv` – one row per participant × run

`subject_id` (without `sub-`), `session`, `<Net>_coherence` and `<Net>_n_voxels` for
`DMN, Salience, FPN, SMN`, and `file_path`. `n_voxels` is the number of voxels in the atlas
network on the BOLD grid (before excluding non-finite / constant voxels).
