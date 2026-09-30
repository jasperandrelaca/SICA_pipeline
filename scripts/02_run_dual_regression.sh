#!/bin/bash
# =============================================================================
# 02_run_dual_regression.sh -- subject-level dual regression (SGE array job)
# =============================================================================
# Uses each participant's OWN MELODIC components (melodic_IC.nii.gz from
# 01_run_melodic.sh) as the spatial template and back-projects them into each
# of that participant's individual BOLD runs, yielding run-level spatial maps
# for every participant-specific component.
#
# Inputs
#   $SUBJECT_LIST                       same subject list used for MELODIC
#   $MELODIC_DIR/sub-<ID>_subject_level_ICA.gica/melodic_IC.nii.gz
#   $FMRIPREP_LIST_DIR/sub-<ID>/subject_level_files.txt
#                                       SAME file (and order) as used for MELODIC
# Output
#   $DUALREG_DIR/<ID>/dr_stage2_subject00000.nii.gz, dr_stage2_subject00001...
#       4-D maps, one 3-D volume per component; subject index k corresponds to
#       line k+1 of subject_level_files.txt (i.e. one file per run/session).
#       Downstream code (06_extract_dr_metrics.py) relies on this ordering.
#
# dual_regression positional arguments used below:
#   <group_IC_maps> <des_norm=1> <design=-1 (group-mean model)> <n_perm> <outdir> <inputs...>
# The randomise step (n_perm) is not used by downstream analyses; set N_PERM=0
# to skip it if your FSL version supports that (check `dual_regression` usage).
#
# Usage (SGE):
#   mkdir -p joblogs
#   export CATD_ROOT=/path/to/project
#   N=$(wc -l < "$CATD_ROOT/subjects.txt")
#   qsub -t 1-${N} scripts/02_run_dual_regression.sh
# =============================================================================

#$ -cwd
#$ -o joblogs/dualreg.$JOB_ID.$TASK_ID
#$ -j y
#$ -l h_rt=20:00:00,h_data=64G
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
DUALREG_DIR="${CATD_DUALREG_DIR:-${CATD_ROOT}/dual_regression}"

N_PERM="${N_PERM:-100}"
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

# --- Inputs / outputs ---------------------------------------------------------
ic_maps="${MELODIC_DIR}/sub-${sid}_subject_level_ICA.gica/melodic_IC.nii.gz"
run_list="${FMRIPREP_LIST_DIR}/sub-${sid}/subject_level_files.txt"
outdir="${DUALREG_DIR}/${sid}"

[[ -f "${ic_maps}" ]]  || { echo "ERROR: missing ${ic_maps}" >&2; exit 1; }
[[ -f "${run_list}" ]] || { echo "ERROR: missing ${run_list}" >&2; exit 1; }

mkdir -p "${outdir}"

# --- Run dual regression ------------------------------------------------------
# $(cat ...) is intentionally unquoted: each line becomes a separate argument.
dual_regression "${ic_maps}" \
    1 \
    -1 \
    "${N_PERM}" \
    "${outdir}" \
    $(cat "${run_list}")

echo "Dual regression completed for sub-${sid}"
