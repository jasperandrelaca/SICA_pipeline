"""
catd_config.py -- shared path configuration and small I/O helpers.

No study data or cluster-specific paths are hard-coded in this repository.
All locations are resolved from environment variables so that the same code
runs on any cluster / workstation:

    CATD_ROOT                 top-level project directory (required unless every
                              directory below is set individually)

    Optional overrides (each defaults to a sub-directory of CATD_ROOT):
    CATD_FMRIPREP_LIST_DIR    {root}/fmriprep_lists      sub-XXXXX/subject_level_files.txt
    CATD_FMRIPREP_OUT_DIR     {root}/fmriprep            fMRIPrep output; denoised runs from step P2 (used by P3)
    CATD_MELODIC_DIR          {root}/melodic             sub-XXXXX_subject_level_ICA.gica/
    CATD_DUALREG_DIR          {root}/dual_regression     {XXXXX}/dr_stage2_subject*.nii.gz
    CATD_TEMPLATES_DIR        {root}/templates           MNIPediatricAsym brain mask + T1w
    CATD_DERIVED_DIR          {root}/derived             all downstream outputs
    CATD_NCT_RESULTS_DIR      {derived}/nct_results      per-subject / per-IC NCT output
    CATD_ASSIGNMENTS_DIR      {derived}/network_assignments
    CATD_MASKS_DIR            {derived}/network_masks    subject-specific RSN masks

See config.example.env for a template.
"""

import os
import re
import glob
from pathlib import Path


# --------------------------------------------------------------------------- #
# Directory resolution
# --------------------------------------------------------------------------- #
def _root() -> Path:
    root = os.environ.get("CATD_ROOT")
    if not root:
        raise EnvironmentError(
            "CATD_ROOT is not set. Export it (see config.example.env) or set "
            "each CATD_*_DIR variable individually."
        )
    return Path(root).expanduser()


def _dir(env_name: str, default_rel: str, base: Path = None) -> Path:
    val = os.environ.get(env_name)
    if val:
        return Path(val).expanduser()
    return (base if base is not None else _root()) / default_rel


def fmriprep_list_dir() -> Path:
    return _dir("CATD_FMRIPREP_LIST_DIR", "fmriprep_lists")


def fmriprep_out_dir() -> Path:
    return _dir("CATD_FMRIPREP_OUT_DIR", "fmriprep")


def melodic_dir() -> Path:
    return _dir("CATD_MELODIC_DIR", "melodic")


def dualreg_dir() -> Path:
    return _dir("CATD_DUALREG_DIR", "dual_regression")


def templates_dir() -> Path:
    return _dir("CATD_TEMPLATES_DIR", "templates")


def derived_dir() -> Path:
    return _dir("CATD_DERIVED_DIR", "derived")


def nct_results_dir() -> Path:
    return _dir("CATD_NCT_RESULTS_DIR", "nct_results", base=derived_dir())


def assignments_dir() -> Path:
    return _dir("CATD_ASSIGNMENTS_DIR", "network_assignments", base=derived_dir())


def masks_dir() -> Path:
    return _dir("CATD_MASKS_DIR", "network_masks", base=derived_dir())


# --------------------------------------------------------------------------- #
# Subject / run helpers (shared by several scripts)
# --------------------------------------------------------------------------- #
def subject_num(subject_id: str) -> str:
    """'sub-10001' -> '10001' (first run of digits)."""
    m = re.search(r"(\d+)", subject_id)
    return m.group(1) if m else subject_id


def find_subject_ids(melodic_directory: Path = None) -> list:
    """
    Subject IDs (e.g. 'sub-10001') inferred from the MELODIC output folders
    named 'sub-XXXXX_subject_level_ICA.gica'.
    """
    melodic_directory = Path(melodic_directory) if melodic_directory else melodic_dir()
    folders = glob.glob(str(melodic_directory / "sub-*_subject_level_ICA.gica"))
    return sorted(os.path.basename(f).split("_")[0] for f in folders)


def read_run_list(subject_id: str, list_dir: Path = None) -> list:
    """
    Read {list_dir}/{subject_id}/subject_level_files.txt -- one preprocessed
    4-D BOLD path per line, in the same order that was passed to MELODIC and
    to dual_regression (this ordering is what links DR outputs back to runs).
    Returns [] if the list does not exist.
    """
    list_dir = Path(list_dir) if list_dir else fmriprep_list_dir()
    path = list_dir / subject_id / "subject_level_files.txt"
    if not path.exists():
        print(f"  WARNING: run list not found: {path}")
        return []
    with open(path, "r") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def parse_session(filepath: str, default: str = "ses-unknown") -> str:
    """Extract the BIDS session label ('ses-XXXX') from a file path."""
    m = re.search(r"(ses-[A-Za-z0-9]+)", str(filepath))
    return m.group(1) if m else default
