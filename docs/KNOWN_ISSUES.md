# Known issues, limitations and design caveats

These are properties of the code as it was used for the study. They are documented so that
readers can interpret the outputs correctly and decide whether to adapt them.

## 1. DMN / Salience / FPN component choice is made in step 06, not step 04
`04_assign_networks.py` picks each network's best IC independently, so the same IC can win
several networks in `subject_network_assignments.csv`, and it can apply a *fallback* when no
IC clears the gate. `06_extract_dr_metrics.py` instead re-derives DMN/Salience/Frontoparietal
assignments (primary/secondary/tertiary) by global matching directly from the NCT tables,
which guarantees distinct components, and it does **not** apply step 04's fallback: a
participant with no gate-passing IC for a network gets empty metrics for that network. The
step-04 columns for these three networks are therefore informational. Dorsal Attention,
Visual and Somatomotor are taken from step 04.

## 2. Resampling of registered ICs to 91×109×91 (step 03)
NCT expects data on FSL's MNI152 2 mm grid (91×109×91 voxels). After applying the ANTs
transform, if the result is not already 91×109×91 (for example because the TemplateFlow
target template has a larger field of view), the code calls `ants.resample_image(...,
use_voxels=True)` with those voxel counts. This keeps the field of view and changes the
voxel size slightly; it does not crop to the FSL grid. Users who need the ICs on FSL's exact
grid should resample onto FSL's `MNI152_T1_2mm` (or crop) instead and check the resulting
affine.

## 3. Dual-regression cluster threshold (step 06)
`largest_cluster` counts voxels ≥ 2.3 in the un-suffixed `dr_stage2_subjectNNNNN.nii.gz`
maps (FSL's stage-2 regression outputs). The `_Z` maps written by `dual_regression` are not
used. Interpret 2.3 as a threshold on those values, not as a z-score threshold.

## 4. Yeo-atlas comparison is resampled, not registered (step 08)
The Yeo 2011 atlas (adult MNI152) is resampled onto the BOLD grid with nearest-neighbour
interpolation. No spatial normalisation is performed between adult MNI152 and
MNIPediatricAsym, unlike the template-to-template registration in step 03.

## 5. Network masks are not constrained to be disjoint
Masks for different networks come from different ICs and can overlap. Between-network
connectivity (step 07) uses the mean time series of each mask as-is, so voxels shared by two
masks contribute to both.

## 6. Nuisance handling happens in step P2 only
Steps 07 and 08 read the files in `subject_level_files.txt` with no extra detrending,
filtering, standardisation or confound regression, so the denoising in
`scripts/preprocessing/P2_denoise_bold.py` is the only nuisance handling. See also §10.

## 7. Permutation-count patch depends on the NCT version (step 03)
`--n-permutations` overrides NCT's permutation count by rewriting the 5th positional argument
of `compute_overlap_with_atlases.permute_overlap_data_all`. If NCT changes that signature the
override will not apply (a warning is printed) and NCT's own default is used.

## 8. Registration is recomputed for every participant (step 03)
The pediatric→adult template registration does not depend on the participant, but it is run
again for each one. This costs time but has no other effect.

## 9. Testing
`tests/smoke_test.py` exercises steps 04–07 on synthetic data. Steps 00–03 and 08 depend on
FSL, ANTs, TemplateFlow, NCT and nilearn's atlas download and are not covered by automated
tests.

## 10. Preprocessing choices to be aware of (steps P1–P2)
* **No distortion correction.** fMRIPrep was run with `--ignore fieldmaps` and without
  fieldmap-less SyN correction.
* **Regression, then filtering.** Nuisance regressors are removed first and the band-pass filter
  is applied afterwards to the data only, not to the regressors. This sequential order can
  reintroduce variance related to the regressors (Hallquist et al., 2013, *NeuroImage* 82).
* **Global signal regression** is optional (`--gsr`). It changes correlation-based measures
  (between-network connectivity, coherence), so the variant used must be reported.
* **No motion censoring or scrubbing** is performed by any script in this repository; run and
  participant exclusion criteria are not scripted.
* **Scheduler resources vs. fMRIPrep resources.** The original fMRIPrep job requested 2 slots
  and 12 GB per slot from the scheduler but told fMRIPrep to use 24 threads and 64 GB. The job
  script now derives both from the allocation (override with `FMRIPREP_NPROCS`,
  `FMRIPREP_MEM_MB`).
