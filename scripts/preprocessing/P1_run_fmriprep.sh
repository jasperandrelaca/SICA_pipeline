#!/bin/bash
# =============================================================================
# P1_run_fmriprep.sh -- fMRIPrep 22.0.2, one participant per array task
# =============================================================================
# Runs fMRIPrep (in its Singularity/Apptainer image) on one participant of a
# BIDS dataset and writes the functional data in MNIPediatricAsym:cohort-1, 2 mm
# space -- the space every downstream step of this repository assumes.
#
# Settings that define the study's preprocessing (do not change casually):
#   fMRIPrep 22.0.2
#   --output-spaces MNIPediatricAsym:res-2:cohort-1
#   --ignore fieldmaps            (fieldmaps are NOT used; no distortion correction)
#   --skip_bids_validation
#
# Inputs (environment variables)
#   CATD_BIDS_DIR            BIDS dataset root
#   CATD_FMRIPREP_OUT_DIR    fMRIPrep output (derivatives) directory
#   CATD_FMRIPREP_SIF        path to fmriprep-22.0.2.sif
#   CATD_FS_LICENSE          FreeSurfer license file (keep it OUT of the repo)
#   BIDS_SUBJECT_LIST        text file, one BIDS participant label per line
#                            WITHOUT the "sub-" prefix (see examples/)
# Optional
#   FMRIPREP_NPROCS          threads   (default: scheduler allocation, else 2)
#   FMRIPREP_MEM_MB          memory MB (default: 22000, i.e. h_data=12G x 2 slots
#                            minus headroom)
#   FMRIPREP_WORK_DIR        work dir  (default: $SCRATCH/fmriprep_work/sub-<label>)
#   CONTAINER_CMD            singularity (default) or apptainer
#   CONTAINER_MODULE         module to load (default: singularity)
#   SINGULARITY_BIND         bind mounts if your site does not bind project storage
#   MODULES_INIT, EXTRA_MODULEPATH   as in 01_run_melodic.sh
#
# Usage (SGE) -- the array range must equal the number of participants:
#   mkdir -p joblogs
#   N=$(wc -l < "$BIDS_SUBJECT_LIST")
#   qsub -t 1-${N} scripts/preprocessing/P1_run_fmriprep.sh
# Slurm:
#   sbatch --array=1-${N} --cpus-per-task=2 --mem=24G --time=10:00:00 \
#          --export=ALL scripts/preprocessing/P1_run_fmriprep.sh
# =============================================================================

#$ -cwd
#$ -o joblogs/fmriprep.$JOB_ID.$TASK_ID
#$ -j y
#$ -pe shared 2
#$ -l h_rt=10:00:00,h_data=12G
#$ -V
## Optional e-mail notification (begin/end/abort):
##$ -M you@example.edu
##$ -m bea

set -euo pipefail

: "${CATD_BIDS_DIR:?CATD_BIDS_DIR must be set}"
: "${CATD_FMRIPREP_OUT_DIR:?CATD_FMRIPREP_OUT_DIR must be set}"
: "${CATD_FMRIPREP_SIF:?CATD_FMRIPREP_SIF must be set}"
: "${CATD_FS_LICENSE:?CATD_FS_LICENSE must be set}"
: "${BIDS_SUBJECT_LIST:?BIDS_SUBJECT_LIST must be set}"

NPROCS="${FMRIPREP_NPROCS:-${NSLOTS:-${SLURM_CPUS_PER_TASK:-2}}}"
MEM_MB="${FMRIPREP_MEM_MB:-22000}"
CONTAINER_CMD="${CONTAINER_CMD:-singularity}"

# --- Pick this task's participant --------------------------------------------
TASK_ID="${SGE_TASK_ID:-${SLURM_ARRAY_TASK_ID:-}}"
[[ -n "${TASK_ID}" ]] || { echo "ERROR: no array task ID (SGE_TASK_ID / SLURM_ARRAY_TASK_ID) set" >&2; exit 1; }
label="$(sed -n "${TASK_ID}p" "${BIDS_SUBJECT_LIST}" | tr -d '[:space:]')"
label="${label#sub-}"
if [[ -z "${label}" ]]; then
    echo "ERROR: task ${TASK_ID} is beyond the end of ${BIDS_SUBJECT_LIST}" >&2
    exit 1
fi
echo "Task ${TASK_ID}: participant sub-${label}"

# --- Environment -------------------------------------------------------------
set +u
if [[ "${CONTAINER_CMD}" != "none" && -r "${MODULES_INIT:-/etc/profile.d/modules.sh}" ]]; then
    . "${MODULES_INIT:-/etc/profile.d/modules.sh}"
    [[ -n "${EXTRA_MODULEPATH:-}" ]] && module use "${EXTRA_MODULEPATH}"
    module load "${CONTAINER_MODULE:-singularity}"
fi
set -u

WORK_DIR="${FMRIPREP_WORK_DIR:-${SCRATCH:-${TMPDIR:-/tmp}}/fmriprep_work/sub-${label}}"
mkdir -p "${CATD_FMRIPREP_OUT_DIR}" "${WORK_DIR}"

# --- Run fMRIPrep ------------------------------------------------------------
"${CONTAINER_CMD}" run --cleanenv "${CATD_FMRIPREP_SIF}" \
    "${CATD_BIDS_DIR}" \
    "${CATD_FMRIPREP_OUT_DIR}" \
    participant --participant-label "${label}" \
    -w "${WORK_DIR}" \
    --output-spaces MNIPediatricAsym:res-2:cohort-1 \
    --ignore fieldmaps \
    --fs-license-file "${CATD_FS_LICENSE}" \
    --skip_bids_validation \
    --nprocs "${NPROCS}" \
    --mem "${MEM_MB}" \
    --verbose

echo "fMRIPrep completed for sub-${label}"
