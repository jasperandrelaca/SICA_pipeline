# CAT-D precision resting-state network pipeline

Code for identifying and quantifying **participant-specific ("precision") resting-state
networks** in the CAT-D (*Characterizing Adolescent Trajectories of Depression*)
longitudinal neuroimaging study, alongside a conventional atlas-based comparison.

For every participant we:

1. decompose all of their resting-state runs with **MELODIC ICA** (participant-level),
2. decide which components correspond to the **DMN, Salience and Frontoparietal**
   networks (and DAN / Visual / Somatomotor) by comparing them with several canonical
   atlases using the **CBIG Network Correspondence Toolbox (NCT)**,
3. back-project those components into every individual run with **dual regression**,
4. extract per-run **network coherence/strength metrics** and **between-network
   connectivity**, and
5. compute a **non-precision comparison** using the standard Yeo 7-network atlas.

> **Scope.** This repository covers the neuroimaging feature-extraction pipeline
> (steps 00-08). The downstream statistical modelling is not included here.
> This repo contains **no study data**; see [Data](#data).

## Pipeline at a glance

```
P1_run_fmriprep.sh   (fMRIPrep 22.0.2 → MNIPediatricAsym:cohort-1, 2 mm)
        │
        ▼
P2_denoise_bold.py   (confound regression, 5 mm smoothing, 0.009–0.08 Hz band-pass, mask)
        │
        ▼
P3_count_scans_build_lists.py   (scan counts per participant → inclusion floor → run lists)
        │
        ▼
subject_level_files.txt (per participant: the denoised runs)
        │
        ▼
01_run_melodic.sh ─────────► participant-level ICA (25 components, all runs concatenated)
        │                               │
        │                               ▼
        │                    03_nct_network_correspondence.py   (register ICs → adult MNI, NCT vs atlases)
        │                               │
        │                               ▼
        │                    04_assign_networks.py  ──► 05_qa_network_assignment.py
        │                               │
        ▼                               ▼
02_run_dual_regression.sh ──► 06_extract_dr_metrics.py ──► masks + per-run DR metrics
                                        │
                                        ▼
                             07_between_network_connectivity.py ──► BNC (r, Fisher Z)

08_yeo_coherence.py  ──►  atlas-based (non-precision) coherence, for comparison
```

| Step | Script | What it does |
|---|---|---|
| P1 | `scripts/preprocessing/P1_run_fmriprep.sh` | fMRIPrep 22.0.2, one participant per array task |
| P2 | `scripts/preprocessing/P2_denoise_bold.py`, `P2_run_denoise.sh` | Post-fMRIPrep denoising of every run (FSL) |
| P3 | `scripts/preprocessing/P3_count_scans_build_lists.py` | Counts denoised scans per participant, applies the inclusion floor, writes `subjects.txt` and the run lists |
| 00 | `scripts/00_fetch_templates.py` | Copies the MNIPediatricAsym cohort-1 2 mm brain mask + T1w from TemplateFlow |
| 01 | `scripts/01_run_melodic.sh` | Participant-level MELODIC ICA (SGE/Slurm array job) |
| 02 | `scripts/02_run_dual_regression.sh` | Dual regression using each participant's own ICs (array job) |
| 03 | `scripts/03_nct_network_correspondence.py` | Registers ICs to adult MNI and runs NCT on every component |
| 04 | `scripts/04_assign_networks.py` | Scores components, picks the winning IC per network, counts fragmentation |
| 05 | `scripts/05_qa_network_assignment.py` | QA plots for the assignments |
| 06 | `scripts/06_extract_dr_metrics.py` | Ranked component assignment, network masks, dual-regression metrics |
| 07 | `scripts/07_between_network_connectivity.py` | Between-network connectivity from the masks |
| 08 | `scripts/08_yeo_coherence.py` | Yeo-atlas coherence (non-precision comparison) |
| – | `scripts/diagnostics/diagnose_min_dice_floor.py` | Optional: shows what a stricter per-network Dice floor would remove (secondary candidates) or cost (winners) |
| – | `scripts/catd_config.py`, `scripts/nct_common.py` | Shared paths/helpers; shared NCT scoring rules and thresholds |

## Two decisions to make first

| Decision | Setting | Study |
|---|---|---|
| Remove or keep the global signal | `--gsr y\|n` (P2, P3) | removed (`y`) |
| Minimum scans per participant | `--min-scans N` (P3) | 3 |

Both are required arguments with no silent default; see
[`docs/PIPELINE.md`](docs/PIPELINE.md#decisions-to-make-before-you-start).

Detailed, annotated documentation of each step, its parameters and outputs:

- [`docs/PIPELINE.md`](docs/PIPELINE.md) – step-by-step guide and rationale
- [`docs/OUTPUTS.md`](docs/OUTPUTS.md) – file and column dictionary
- [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) – limitations and design caveats to be aware of

## Requirements

- Python ≥ 3.8 and the packages in [`requirements.txt`](requirements.txt)
- [FSL](https://fsl.fmrib.ox.ac.uk/) ≥ 6.0 (`melodic`, `dual_regression`, `fsl_regfilt`, `fslmaths`); the study used 6.0.7.15
- For step P1: Singularity/Apptainer, the fMRIPrep 22.0.2 image and a FreeSurfer license
- BOLD data in **MNIPediatricAsym:cohort-1 (2 mm)** space (steps P1–P2 produce them)
- A cluster with SGE or Slurm for steps 01-03 (all steps also run locally on small samples)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```bash
# 1. Point the scripts at your data (see config.example.env for every option)
cp config.example.env config.env      # edit CATD_ROOT etc.
source config.env

# 2. Preprocessing (skip if you already have denoised MNIPediatricAsym runs)
#    needs CATD_BIDS_DIR, CATD_FMRIPREP_OUT_DIR, CATD_FMRIPREP_SIF, CATD_FS_LICENSE, BIDS_SUBJECT_LIST
mkdir -p joblogs
NB=$(wc -l < "$BIDS_SUBJECT_LIST")
qsub -t 1-${NB} scripts/preprocessing/P1_run_fmriprep.sh
GSR=y qsub -V -t 1-${NB} scripts/preprocessing/P2_run_denoise.sh     # GSR=y removes the global signal, n keeps it

# 3. Inclusion: look at how many scans each participant has, then choose the floor
python scripts/preprocessing/P3_count_scans_build_lists.py --gsr y --subject-list "$BIDS_SUBJECT_LIST"
python scripts/preprocessing/P3_count_scans_build_lists.py --gsr y --min-scans 3 --write-lists
#    writes $CATD_ROOT/subjects.txt and $CATD_FMRIPREP_LIST_DIR/sub-<ID>/subject_level_files.txt
#    (skip P3 if you build these two inputs yourself: one participant ID per line, and one
#    preprocessed 4-D BOLD path per line, in the same order for MELODIC and dual regression)

# 4. Run the numbered steps in order
python scripts/00_fetch_templates.py
mkdir -p joblogs
N=$(wc -l < "$CATD_ROOT/subjects.txt")      # number of participants kept by P3
qsub -t 1-${N} scripts/01_run_melodic.sh            # Slurm: sbatch --array=1-${N} ... (docs/PIPELINE.md)
qsub -t 1-${N} scripts/02_run_dual_regression.sh
python scripts/03_nct_network_correspondence.py     # use --start/--end to process in batches
python scripts/04_assign_networks.py
python scripts/05_qa_network_assignment.py
python scripts/06_extract_dr_metrics.py \
    --network-assignments "$CATD_ASSIGNMENTS_DIR/subject_network_assignments.csv" \
    --overwrite-masks
python scripts/07_between_network_connectivity.py --output-csv between_network_connectivity.csv
python scripts/08_yeo_coherence.py --output yeo_coherence_results.csv
```

Check your installation without any data:

```bash
python tests/smoke_test.py               # synthetic data; exercises steps 04-07
python tests/test_denoise_stubbed.py     # steps P2 (stubbed FSL)
python tests/test_scan_counts.py         # step P3
python tests/test_diagnose_floor.py      # Dice-floor diagnostic
```

## Data

No participant data, identifiers or derived participant-level outputs are included, and
`.gitignore` blocks NIfTI files and tabular outputs. Participant lists must be kept outside
version control (`subjects.txt` is git-ignored).

<!-- TODO(authors): add data-availability statement / OpenNeuro accession and access instructions. -->

## Citation and contact

<!-- TODO(authors): add citation (manuscript/preprint DOI), license, and maintainer contact. -->

If you use this code, please also cite the tools it builds on: FSL MELODIC / dual
regression, fMRIPrep, TemplateFlow, ANTs, the CBIG Network Correspondence Toolbox, and the
Yeo et al. (2011) parcellation.
