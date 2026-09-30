#!/bin/bash
# =============================================================================
# P2_run_denoise.sh -- array-job wrapper around P2_denoise_bold.py
# =============================================================================
# One array task per participant (same BIDS_SUBJECT_LIST as P1). For every
# session directory of that participant in the fMRIPrep output
# (sub-<label>/ses-*/, or sub-<label>/ itself if there are no sessions) it runs
# P2_denoise_bold.py.
#
# Required environment
#   CATD_FMRIPREP_OUT_DIR   fMRIPrep output directory (as in P1)
#   BIDS_SUBJECT_LIST       participant labels, one per line, without "sub-"
#   GSR                     y = regress global signal, n = keep it (no default:
#                           this is a methods choice, state it explicitly)
# Optional
#   FSL_MODULE (default fsl/6.0.7.15), MODULES_INIT, EXTRA_MODULEPATH
#   P2_EXTRA_ARGS           extra arguments for P2_denoise_bold.py (e.g. --overwrite)
#
# Usage (SGE):
#   N=$(wc -l < "$BIDS_SUBJECT_LIST")
#   GSR=y qsub -V -t 1-${N} scripts/preprocessing/P2_run_denoise.sh
# =============================================================================

#$ -cwd
#$ -o joblogs/denoise.$JOB_ID.$TASK_ID
#$ -j y
#$ -pe shared 1
#$ -l h_rt=4:00:00,h_data=16G
#$ -V

set -euo pipefail

: "${CATD_FMRIPREP_OUT_DIR:?CATD_FMRIPREP_OUT_DIR must be set}"
: "${BIDS_SUBJECT_LIST:?BIDS_SUBJECT_LIST must be set}"
: "${GSR:?GSR must be set to y or n}"
[[ "${GSR}" == "y" || "${GSR}" == "n" ]] || { echo "ERROR: GSR must be y or n" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASK_ID="${SGE_TASK_ID:-${SLURM_ARRAY_TASK_ID:-}}"
[[ -n "${TASK_ID}" ]] || { echo "ERROR: no array task ID set" >&2; exit 1; }
label="$(sed -n "${TASK_ID}p" "${BIDS_SUBJECT_LIST}" | tr -d '[:space:]')"
label="${label#sub-}"
[[ -n "${label}" ]] || { echo "ERROR: task ${TASK_ID} is beyond the end of ${BIDS_SUBJECT_LIST}" >&2; exit 1; }

set +u
if [[ -r "${MODULES_INIT:-/etc/profile.d/modules.sh}" ]]; then
    . "${MODULES_INIT:-/etc/profile.d/modules.sh}"
    [[ -n "${EXTRA_MODULEPATH:-}" ]] && module use "${EXTRA_MODULEPATH}"
    module load "${FSL_MODULE:-fsl/6.0.7.15}"
fi
set -u

subdir="${CATD_FMRIPREP_OUT_DIR}/sub-${label}"
[[ -d "${subdir}" ]] || { echo "ERROR: ${subdir} not found (did P1 finish?)" >&2; exit 1; }

shopt -s nullglob
dirs=("${subdir}"/ses-*/)
[[ ${#dirs[@]} -gt 0 ]] || dirs=("${subdir}/")

status=0
for d in "${dirs[@]}"; do
    echo "=== sub-${label}: ${d}"
    # shellcheck disable=SC2086
    python3 "${HERE}/P2_denoise_bold.py" "${d%/}" --gsr "${GSR}" ${P2_EXTRA_ARGS:-} || status=1
done
exit ${status}
