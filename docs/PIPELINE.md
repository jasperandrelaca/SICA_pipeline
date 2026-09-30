# Pipeline guide

This document explains what each step does, why, what it needs and what it produces.
Paths are given relative to the directories configured in `config.example.env`
(`$CATD_ROOT` and friends).

## Conventions used throughout

| Item | Convention |
|---|---|
| Participant ID | `sub-XXXXX` in file/folder names; the numeric part (`XXXXX`) in `subjects.txt` and in dual-regression folder names |
| Component (IC) numbering | **0-based** everywhere: `IC00 … IC24` for 25 components. IC *n* is volume *n* of `melodic_IC.nii.gz` **and** volume *n* of every dual-regression stage-2 map, so no re-indexing is ever needed |
| Run order | `subject_level_files.txt` lists one BOLD file per line. The *same file in the same order* is given to MELODIC and dual regression. Dual-regression output `dr_stage2_subjectKKKKK.nii.gz` corresponds to line *K+1* of that list, which is how metrics are linked back to sessions |
| Session label | Parsed from the BOLD path with the pattern `ses-[A-Za-z0-9]+` (e.g. `ses-v1`, `ses-v2o1`) |
| Space | All functional data and MELODIC outputs are in **MNIPediatricAsym:cohort-1, 2 mm** |

---

## Decisions to make before you start

Two choices shape everything downstream. Make them explicitly, and report them in any write-up.

| # | Decision | Where it is set | What the study used |
|---|---|---|---|
| D1 | **Global signal: remove or keep?** Regressing the global signal (GSR) changes correlation-based measures such as between-network connectivity and coherence. | `--gsr y\|n` in P2 (`GSR=y\|n` for the array wrapper) and again in P3, so that both refer to the same output files | **Removed** (`--gsr y`; files tagged `removed_GS`) |
| D2 | **Minimum number of scans per participant** (inclusion floor). Participants below the floor are left out of MELODIC, dual regression and everything after. | `--min-scans N` in P3 (look at the scan-count table first) | **3 scans** |

Neither script has a silent default: `--gsr` is required in P2 and P3. D1 must be decided
before P2 is run (the two variants are written side by side and never overwrite each other,
but downstream steps use only one of them). D2 can only be made after P2, because it depends
on how many runs survived preprocessing; P3 shows the distribution to decide from.

---

## Steps P1–P3 – Preprocessing and inclusion (`scripts/preprocessing/`)

These steps produce the BOLD files, the participant list and the run lists that every numbered step below starts from.

### Step P1 – fMRIPrep (`P1_run_fmriprep.sh`)
One array task per participant; runs **fMRIPrep 22.0.2** from its container image.

| Setting | Value | Why it matters |
|---|---|---|
| `--output-spaces` | `MNIPediatricAsym:res-2:cohort-1` | Data are normalised to the pediatric asymmetric MNI template (cohort 1, 2 mm); all downstream masks and templates are in this space |
| `--ignore fieldmaps` | on | Fieldmaps are not used. Fieldmap-less SyN correction (`--use-syn-sdc`) is not enabled either, so **no susceptibility-distortion correction** is applied |
| `--skip_bids_validation` | on | |
| `--fs-license-file` | your FreeSurfer license | FreeSurfer surface reconstruction is run (default fMRIPrep behaviour); keep the license file out of the repository |

The participant list for P1 contains **BIDS labels** (e.g. `901`, no `sub-` prefix), which are
not necessarily the same identifiers as the ones used from step 01 onwards
(`examples/bids_participants.example.txt` vs `examples/subjects.example.txt`).

Output used later, per run, in `<fmriprep_out>/sub-<label>/[ses-<label>/]func/`:
`*_space-MNIPediatricAsym_cohort-1_res-2_desc-preproc_bold.nii.gz`,
`*_desc-confounds_timeseries.tsv` and `*_space-MNIPediatricAsym_cohort-1_res-2_desc-brain_mask.nii.gz`.

### Step P2 – Denoising (`P2_denoise_bold.py`, array wrapper `P2_run_denoise.sh`)
Applied to every run, in this order:

| # | Operation | FSL command | Parameters |
|---|---|---|---|
| 1 | Nuisance regression | `fsl_regfilt -f 1,2,…,N` | 6 motion parameters, white matter and CSF, each with derivative, square and squared derivative (**32 regressors**); with `--gsr y` also the global signal with the same expansions (**36**) |
| 2 | Spatial smoothing | `fslmaths -s 2.123` | Gaussian σ = 2.123 mm (5 mm FWHM) |
| 3 | Temporal band-pass | `fslmaths -bptf hp lp` | 0.009–0.08 Hz; σ in volumes = 1 / (2 · TR · f), TR read from the NIfTI header |
| 4 | De-meaning | `-Tmean`, `-sub` | voxel-wise temporal mean removed |
| 5 | Brain mask | `-mas` | fMRIPrep MNIPediatricAsym brain mask of the same run |

Final file: `<run>_space-MNIPediatricAsym_cohort-1_res-2_desc-preproc_removed_GS_smoothed_filtered_demeaned_masked.nii.gz`
(`removed_GS` → `with_GS` if `--gsr n`). `--gsr` has no default because global signal
regression is a methods choice (decision D1; the study used `y`, i.e. `removed_GS`).

**Differences from the script used for the study** (the processing steps, regressors and
parameters are unchanged):

* Each BOLD run is paired with the confounds file and mask carrying the same BIDS prefix, and
  all runs found in the folder are processed. The original took the first `*_bold.nii.gz` and
  the first confounds file in directory order, which could pair different runs.
* The regression design matrix is written **without a header row**, with the empty first-row
  values of the derivative columns set to 0. (The original wrote the column names into the
  file passed to `fsl_regfilt -d` and left those first-row values empty.) Column names are
  saved separately in `*_filtered_confounds_columns.txt`, and the number of values set to 0
  is written to the log.
* The number of confound rows must equal the number of BOLD volumes, otherwise the run fails
  (see `--confound-skip-rows`).
* The "output already exists" check now looks for the output of the *selected* GSR variant. The
  original always looked for the `removed_GS` file, so running `n` after `y` did nothing.
* Failure at any step stops that run (the original carried on with missing inputs), a missing
  brain mask is an error (the original skipped masking but logged success), and the exit code
  is non-zero if any run failed.
* Intermediate images are deleted after success. In the original the cleanup lines sat inside
  the masking `except:` block, so they ran only when masking failed, and then removed the
  data that had just been produced.
* The log is `preproc_denoising_log.tsv` (one row per run and step).
* `bptf` sigmas are computed in double precision (values agree with the original to ~1e-7).

---

### Step P3 – Count scans per participant and build the run lists (`P3_count_scans_build_lists.py`)
Scans the denoised files of the chosen GSR variant (`--gsr`, same as in P2) under the fMRIPrep
output directory and counts **one scan per denoised run file** for every participant. Runs in
two stages:

1. **Report** (always). Prints how many participants have 1, 2, 3 and 4+ scans and, for each
   possible floor (≥1 … ≥4), how many participants and scans would remain. With
   `--subject-list` (the P1 list) it also names participants with **zero** scans, which normally
   means a failed or never-submitted P1/P2 job. Use this table to make decision D2.
   `--out-csv` saves the per-participant counts (label, ID, scans, sessions, included).
2. **Build** (only with `--min-scans N --write-lists`). Keeps participants with at least *N*
   scans and writes `$CATD_FMRIPREP_LIST_DIR/sub-<ID>/subject_level_files.txt` (one run per
   line, sorted by path) and the participant list `subjects.txt` used by steps 01–02, then
   prints the array range to submit (`1-<number of participants>`). It refuses to overwrite
   existing files unless `--overwrite` is given.

```bash
python scripts/preprocessing/P3_count_scans_build_lists.py --gsr y \
    --fmriprep-dir "$CATD_FMRIPREP_OUT_DIR" --subject-list "$BIDS_SUBJECT_LIST"      # look first
python scripts/preprocessing/P3_count_scans_build_lists.py --gsr y --min-scans 3 --write-lists
```

If the identifiers used from step 01 on differ from the BIDS labels, give `--id-map`, a CSV
with columns `bids_label,study_id`; otherwise the BIDS label is used as the ID. Only non-empty
files count, and the other GSR variant is ignored. The floor applies to the number of scans, not
to sessions, and P3 does not look at motion or image quality: any additional run-level QC
exclusion has to be applied before running P3 (for instance by removing those files).

---

## Step 0 – Inputs to the numbered steps

### Preprocessed BOLD
Produced by steps P1–P2 above (fMRIPrep, then denoising).

### `subject_level_files.txt`
Written by step P3. One file per participant at `$CATD_FMRIPREP_LIST_DIR/sub-<ID>/subject_level_files.txt`,
listing the absolute paths of that participant's included BOLD runs (across sessions), one
per line:

```
/path/to/sub-10001/ses-v1/func/sub-10001_ses-v1_task-rest_..._desc-preproc_removed_GS_smoothed_filtered_demeaned_masked.nii.gz
/path/to/sub-10001/ses-v2o1/func/sub-10001_ses-v2o1_task-rest_..._desc-preproc_removed_GS_smoothed_filtered_demeaned_masked.nii.gz
```

This list defines which runs enter every downstream step. Steps 07 and 08 use the listed
files exactly as they are (no further filtering, smoothing or confound regression), so the
list must point at the fully denoised files from step P2.

### `subjects.txt`
Written by step P3 (participants at or above the scan floor). One numeric participant ID per line (`examples/subjects.example.txt`). Keep it out of version
control. The job scripts take line *N* of this file for array task *N*; the array range is the number of lines.

### Templates – `00_fetch_templates.py`
MELODIC needs the brain mask and T1w background of the space the data live in. The script
copies `tpl-MNIPediatricAsym_cohort-1_res-2_desc-brain_mask.nii.gz` and
`tpl-MNIPediatricAsym_cohort-1_res-2_T1w.nii.gz` from TemplateFlow into `$CATD_TEMPLATES_DIR`.

---

## Step 1 – Participant-level MELODIC ICA (`01_run_melodic.sh`)

Runs FSL MELODIC once per participant on **all** of that participant's runs
(multi-input → temporal concatenation), giving participant-specific spatial components rather than
projecting a group template onto everyone.

| Parameter | Value | Where |
|---|---|---|
| Model order | 25 (`-d 25`) | `N_COMPONENTS` |
| TR | 2.5 s | `TR` |
| Mask / background | MNIPediatricAsym brain mask / T1w | `$CATD_TEMPLATES_DIR` |
| Extra outputs | `--Oall`, `--report`, `--verbose`, `--debug` | script |

Output: `$CATD_MELODIC_DIR/sub-<ID>_subject_level_ICA.gica/` (notably `melodic_IC.nii.gz`,
25 volumes, in MNIPediatricAsym space, and the HTML `report/`).

**Submitting.** Set the array range to the number of participants:

```bash
N=$(wc -l < "$CATD_ROOT/subjects.txt")
qsub -t 1-${N} scripts/01_run_melodic.sh          # SGE
```

Slurm equivalent (the script reads `SLURM_ARRAY_TASK_ID` automatically):

```bash
sbatch --array=1-${N} --cpus-per-task=2 --mem=16G --time=20:00:00 \
       --output=joblogs/melodic.%A_%a.log --export=ALL scripts/01_run_melodic.sh
```

The `#$` lines are SGE directives and are ignored by Slurm. Adjust the `module` lines
(`FSL_MODULE`, `MODULES_INIT`, `EXTRA_MODULEPATH`) for your site.

## Step 2 – Dual regression (`02_run_dual_regression.sh`)

Uses each participant's **own** `melodic_IC.nii.gz` as the spatial template and back-projects
it into each of that participant's runs (FSL `dual_regression`: stage 1 spatial regression
→ time courses, stage 2 temporal regression → run-specific spatial maps). Time courses are
variance-normalised (`des_norm=1`).

Output: `$CATD_DUALREG_DIR/<ID>/dr_stage2_subject00000.nii.gz`, `…00001.nii.gz`, … – one
4-D file per run, one volume per component. Downstream metrics use these un-suffixed
stage-2 maps. The randomise step (`N_PERM`, default 100) is not used downstream.

```bash
qsub -t 1-${N} scripts/02_run_dual_regression.sh
sbatch --array=1-${N} --mem=64G --time=20:00:00 --output=joblogs/dualreg.%A_%a.log \
       --export=ALL scripts/02_run_dual_regression.sh          # Slurm
```

## Step 3 – Network correspondence (`03_nct_network_correspondence.py`)

Decides which canonical networks each participant-specific component resembles.

1. **Register ICs to adult MNI space.** NCT's atlases live in FSL MNI152 2 mm space, so the
   4-D `melodic_IC.nii.gz` is moved from MNIPediatricAsym to MNI152 by registering the two
   *templates* with ANTs (`SyN` by default) and applying that transform to the ICs (linear
   interpolation, time-series mode). The result is saved as
   `sub-XXXXX_melodic_IC_FSLMNI2mm.nii.gz`. See KNOWN_ISSUES §2 regarding the final
   resampling to 91×109×91.
2. **Split** into one 3-D file per component (`ica_components_3d/`).
3. **Threshold and compare.** Each component is thresholded at **z > 2.5** (`--threshold`) by
   NCT and compared, as a soft map, with the atlases `TY7, TY17, AS400Y17, EG17`
   (CBIG NCT identifiers; see the NCT documentation for full descriptions). NCT reports Dice
   overlap and a permutation-based p-value per atlas network in
   `sub-XXXXX_ICnn/network_correspondence.csv`.
4. **Permutations.** 1000 permutations by default (`--n-permutations`); `--no-permutations`
   skips p-values entirely (then steps 04+ cannot be used, since they rely on p-values).

Memory: the script frees figures/objects after each component. Process participants in
batches with `--start/--end` (0-based indices into the sorted list of participants), e.g.
`--start 0 --end 10`, then `--start 10 --end 20`, and so on.

## Step 4 – Assigning components to networks (`04_assign_networks.py`)

For each participant and network, each IC is scored from its NCT table (`nct_common.py`):

```
weight(row) = (1 − p) × Dice                       for each atlas network belonging to the target network
score(IC)   = mean(weight) + 0.1 × (number of atlases with a match)
```

The mapping of atlas labels to networks (e.g. DMN = `Default` in TY7/EG17; `DefaultA/B/C` in
TY17/AS400Y17) is `NETWORK_MAPPINGS` in `nct_common.py`.

**Candidate gate.** An IC is a *candidate* for a network if its best Dice ≥ the network's
floor **and** its best p ≤ 0.05:

| Network | Dice floor |
|---|---|
| DMN, Frontoparietal | 0.3 |
| Salience | 0.2 |
| Dorsal Attention, Visual, Somatomotor | 0.2 |

Salience keeps the lower floor because its winning components have lower Dice than DMN/FPN
winners and a higher floor would exclude genuine matches; DMN/FPN floors were raised to cut
low-quality secondary candidates.

**Winner** = candidate with the highest score. **Fallback:** if a participant has *no*
candidate, the best IC under a permissive gate (Dice ≥ 0, p ≤ 0.05) is taken and flagged in
`<network>_is_fallback`.

**Fragmentation.** The number of candidates per network is recorded
(`n_significant_<network>`); >1 indicates the network is split across several components.

Outputs are described in `OUTPUTS.md`.

## Optional diagnostic – choosing the Dice floor (`scripts/diagnostics/diagnose_min_dice_floor.py`)

Run after step 03. It explains, and lets you re-check, the network-specific floors in
`nct_common.CANDIDATE_MIN_DICE` (DMN 0.3, Salience 0.2, Frontoparietal 0.3, others 0.2).

For every participant and network it re-derives the candidate ICs that clear the current gate
(same scoring and gate as steps 04 and 06, imported from `nct_common.py`), calls the
highest-scoring one the **winner** and every other candidate a **secondary** candidate (the
source of fragmentation counts above 1). For each network it then reports the Dice
distribution of secondary candidates and winners, how many secondary candidates a stricter floor
(0.20 … 0.70) would remove, and how many **winners** the same floor would also remove.

How to read it: a floor that removes many secondary candidates and no winners is removing
noise. If the winners' Dice range overlaps the secondary range (as it does for Salience), a
higher floor would discard real matches, so the floor is left permissive.

```bash
python scripts/diagnostics/diagnose_min_dice_floor.py                   # uses $CATD_NCT_RESULTS_DIR
python scripts/diagnostics/diagnose_min_dice_floor.py --floors 0.2 0.3 0.4 --out-prefix floor_diag
```

`--out-prefix` also writes `<prefix>_winners.csv` and `<prefix>_secondary.csv`. The script only
reports; changing a floor means editing `CANDIDATE_MIN_DICE` in `nct_common.py` and re-running
steps 04–07.

---

## Step 5 – QA plots (`05_qa_network_assignment.py`)

Plots coverage (% of participants with an identified component per network), the Dice and
p-value distributions of the winning components, agreement across atlases, and composite
score vs Dice, written to `<assignments dir>/quality_plots/`. Use these to sanity-check the
gates in step 4 before extracting metrics.

## Step 6 – Ranked components, masks and dual-regression metrics (`06_extract_dr_metrics.py`)

**Component assignment (DMN, Salience, Frontoparietal).** Scores are recomputed from the NCT
tables with the same rules as step 04, then components are allocated by three sequential
rounds of global maximum-weight bipartite matching (Hungarian algorithm):

| Round | Column suffix | Meaning |
|---|---|---|
| 1 | *(none)* | primary – jointly optimal, one distinct IC per network |
| 2 | `_b` | secondary – best remaining ICs after removing round-1 winners |
| 3 | `_c` | tertiary – best remaining ICs after removing rounds 1–2 |

Matching jointly (rather than picking each network's best IC independently, as step 04 does)
prevents one IC from being claimed by two networks. A network gets no component in a round if
no unclaimed candidate clears the gate. Dorsal Attention, Visual and Somatomotor use the
step-04 winner only.

**Masks.** For each selected IC, take its positive voxels and keep those at or above the
**95th percentile** (top 5%) in the original MNIPediatricAsym-space `melodic_IC.nii.gz`. Masks
are written as `$CATD_MASKS_DIR/sub-XXXXX/sub-XXXXX_<Network><suffix>_mask.nii.gz`. Existing
masks are reused unless `--overwrite-masks` is passed – pass it whenever thresholds or
assignment logic change.

**Metrics.** For every run and every network/tier, inside the mask on the dual-regression
stage-2 volume of the matching IC: mean, variance, SD, and the size of the largest connected
cluster of voxels ≥ 2.3 (`--dr-threshold`).

## Step 7 – Between-network connectivity (`07_between_network_connectivity.py`)

Requires the masks from step 6. For each run: average BOLD time series in each network mask,
Pearson r between DMN–SN, DMN–FPN, SN–FPN, Fisher z (r clipped to ±0.9999). Done twice:

- **primary** masks (top-ranked IC per network), and
- **combined** masks: union of primary and secondary (`_b`) masks. If a participant has no
  secondary mask for a network, the primary mask is used for that network and the
  `<NET>_mask_combined` flag is `False`.

## Step 8 – Yeo-atlas coherence (`08_yeo_coherence.py`)

Non-precision comparison: the Yeo 2011 7-network atlas (`thick_7`, from nilearn) is resampled
(nearest neighbour) onto the BOLD grid, and for DMN (7), Salience/Ventral attention (4),
FPN (6) and Somatomotor (2) coherence is the mean correlation between each in-network voxel's
time series and the network mean time series (voxels that are non-finite or constant are
excluded).

---

## Reproducibility notes

- **Run order matters:** 04 → 06 → 07. Step 07 reads masks written by step 06.
- Step 06 is the authority for DMN/Salience/FPN component choice (see Step 6); the
  corresponding columns in step 04's CSV are informational.
- Random elements: NCT permutation p-values and ANTs registration are not seeded here.
- `tests/smoke_test.py` runs steps 04–07 on synthetic data to verify an installation.
