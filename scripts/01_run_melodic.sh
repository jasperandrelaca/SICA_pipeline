#!/bin/bash
# =============================================================================
# 01_run_melodic.sh -- subject-level MELODIC ICA (one SGE array task / subject)
# =============================================================================
# For each participant, runs FSL MELODIC on ALL of that participant's
# preprocessed BOLD runs together (temporal concatenation across sessions),
# producing a participant-specific ("precision") set of spatial components.
#
# Inputs
#   $SUBJECT_LIST                       text file, one participant ID per line
#                                       (numeric part only, e.g. 12345)
#   $FMRIPREP_LIST_DIR/sub-<ID>/subject_level_files.txt
#                                       one preprocessed 4-D BOLD path per line
#   $TEMPLATES_DIR/tpl-MNIPediatricAsym_cohort-1_res-2_desc-brain_mask.nii.gz
#   $TEMPLATES_DIR/tpl-MNIPediatricAsym_cohort-1_res-2_T1w.nii.gz
#                                       (see 00_fetch_templates.py)
# Output
#   $MELODIC_DIR/sub-<ID>_subject_level_ICA.gica/   (melodic_IC.nii.gz, report/, ...)
#
# Usage (SGE) -- set the array range to the number of lines in the subject list:
#   mkdir -p joblogs
#   export CATD_ROOT=/path/to/project
#   N=$(wc -l < "$CATD_ROOT/subjects.txt")
#   qsub -t 1-${N} scripts/01_run_melodic.sh
#
# NOTE: cluster directives and `module` lines below target an SGE cluster with
# Environment Modules. Adapt them (or see docs/PIPELINE.md for Slurm hints) for
# your site.
# =============================================================================

#$ -cwd
#$ -o joblogs/melodic.$JOB_ID.$TASK_ID
#$ -j y
#$ -pe shared 2
#$ -l h_rt=20:00:00,h_data=16G
#$ -V
## Optional: abort-only e-mail notification
##$ -M you@example.edu
##$ -m a

set -euo pipefail

# --- Configuration (override via environment) --------------------------------
: "${CATD_ROOT:?CATD_ROOT must be set (see config.example.env)}"
SUBJECT_LIST="${SUBJECT_LIST:-${CATD_ROOT}/subjects.txt}"
FMRIPREP_LIST_DIR="${CATD_FMRIPREP_LIST_DIR:-${CATD_ROOT}/fmriprep_lists}"
MELODIC_DIR="${CATD_MELODIC_DIR:-${CATD_ROOT}/melodic}"
TEMPLATES_DIR="${CATD_TEMPLATES_DIR:-${CATD_ROOT}/templates}"

N_COMPONENTS="${N_COMPONENTS:-25}"   # model order (-d)
TR="${TR:-2.5}"                      # repetition time in seconds (--tr)
FSL_MODULE="${FSL_MODULE:-fsl/6.0.7.15}"

# --- Pick this task's participant --------------------------------------------
# Works under SGE (SGE_TASK_ID) and Slurm (SLURM_ARRAY_TASK_ID).
TASK_ID="${SGE_TASK_ID:-${SLURM_ARRAY_TASK_ID:-}}"
[[ -n "${TASK_ID}" ]] || { echo "ERROR: no array task ID (SGE_TASK_ID / SLURM_ARRAY_TASK_ID) set" >&2; exit 1; }
sid="$(sed -n "${TASK_ID}p" "${SUBJECT_LIST}" | tr -d '[:space:]')"
if [[ -z "${sid}" ]]; then
    echo "ERROR: task ${TASK_ID} is beyond the end of ${SUBJECT_LIST}" >&2
    exit 1
fi
echo "Task ${TASK_ID}: participant ${sid}"

# --- Environment -------------------------------------------------------------
# Module init scripts often reference unset variables, so relax `set -u` here.
set +u
. "${MODULES_INIT:-/etc/profile.d/modules.sh}"   # site-specific
[[ -n "${EXTRA_MODULEPATH:-}" ]] && module use "${EXTRA_MODULEPATH}"
module load "${FSL_MODULE}"
set -u

mkdir -p "${MELODIC_DIR}"

# --- Run MELODIC --------------------------------------------------------------
melodic -i "${FMRIPREP_LIST_DIR}/sub-${sid}/subject_level_files.txt" \
        -o "${MELODIC_DIR}/sub-${sid}_subject_level_ICA.gica" \
        -m "${TEMPLATES_DIR}/tpl-MNIPediatricAsym_cohort-1_res-2_desc-brain_mask.nii.gz" \
        --bgimage="${TEMPLATES_DIR}/tpl-MNIPediatricAsym_cohort-1_res-2_T1w.nii.gz" \
        -d "${N_COMPONENTS}" \
        --tr="${TR}" \
        --Oall \
        --verbose \
        --debug \
        --report

echo "MELODIC completed for sub-${sid}"
