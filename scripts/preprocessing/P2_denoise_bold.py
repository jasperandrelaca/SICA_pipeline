#!/usr/bin/env python3
"""
P2_denoise_bold.py -- post-fMRIPrep denoising of resting-state BOLD runs (FSL).

For every fMRIPrep MNIPediatricAsym BOLD run found in <input_dir>/func the script
applies, in this order (same as the study):

  1. nuisance regression   fsl_regfilt with 32 confounds (+4 if --gsr y)
  2. spatial smoothing     fslmaths -s 2.123   (sigma in mm = 5 mm FWHM)
  3. temporal band-pass    fslmaths -bptf      (0.009 - 0.08 Hz)
  4. de-meaning            subtract the voxel-wise temporal mean
  5. brain masking         fMRIPrep MNIPediatricAsym brain mask

Confound set (from fMRIPrep's *_desc-confounds_timeseries.tsv)
  6 motion parameters + their temporal derivatives, squares and squared derivatives
  (= 24 regressors), white matter and CSF, each with derivative, square and squared
  derivative (= 8 regressors), and, only with --gsr y, the global signal with the
  same four expansions (= 4 regressors).

Output (next to the input BOLD file, same naming as the study):
  <prefix>_..._desc-preproc_removed_GS_smoothed_filtered_demeaned_masked.nii.gz  (--gsr y)
  <prefix>_..._desc-preproc_with_GS_smoothed_filtered_demeaned_masked.nii.gz     (--gsr n)
  <run>_filtered_confounds.tsv          the design matrix given to fsl_regfilt (no header)
  <run>_filtered_confounds_columns.txt  the column names of that matrix, in order
  preproc_denoising_log.tsv             one row per run and step (status + message)

Intermediate images (smoothed, filtered, mean, demeaned) are deleted after a run
succeeds unless --keep-intermediates is given. The un-smoothed regression output
(..._removed_GS.nii.gz / ..._with_GS.nii.gz) is always kept.

Usage
  python P2_denoise_bold.py /path/to/fmriprep/sub-901/ses-1 --gsr y

Requires FSL on PATH (fsl_regfilt, fslmaths), nibabel and pandas.
Exit status is non-zero if any run failed.

Changes relative to the original script are listed in docs/PIPELINE.md (Step P2).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import pandas as pd

# Same regressors and order as the original script.
BASE_CONFOUNDS = [
    "trans_x", "trans_x_derivative1", "trans_x_derivative1_power2", "trans_x_power2",
    "trans_y", "trans_y_derivative1", "trans_y_derivative1_power2", "trans_y_power2",
    "trans_z", "trans_z_derivative1", "trans_z_derivative1_power2", "trans_z_power2",
    "rot_x", "rot_x_derivative1", "rot_x_power2", "rot_x_derivative1_power2",
    "rot_y", "rot_y_derivative1", "rot_y_power2", "rot_y_derivative1_power2",
    "rot_z", "rot_z_derivative1", "rot_z_power2", "rot_z_derivative1_power2",
    "white_matter", "white_matter_derivative1", "white_matter_power2",
    "white_matter_derivative1_power2",
    "csf", "csf_derivative1", "csf_power2", "csf_derivative1_power2",
]
GSR_CONFOUNDS = [
    "global_signal", "global_signal_derivative1",
    "global_signal_power2", "global_signal_derivative1_power2",
]

DEFAULT_SPACE_TAG = "space-MNIPediatricAsym_cohort-1_res-2"


def sh(cmd):
    """Run an FSL command, echoing it first; raises CalledProcessError on failure."""
    print("Running command:", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def with_suffix(path, tag):
    """'x.nii.gz' + '_tag' -> 'x_tag.nii.gz'."""
    return Path(str(path)[: -len(".nii.gz")] + tag + ".nii.gz")


def find_runs(func_dir, space_tag):
    """
    fMRIPrep preprocessed BOLD runs in `func_dir`, each paired by its BIDS prefix
    (everything before '_space-') with its own confounds file and brain mask, so
    that a run can never be denoised with another run's confounds.
    """
    runs = []
    for bold in sorted(func_dir.glob(f"*_{space_tag}_desc-preproc_bold.nii.gz")):
        prefix = bold.name.split("_space-")[0]
        runs.append({
            "prefix": prefix,
            "bold": bold,
            "confounds": func_dir / f"{prefix}_desc-confounds_timeseries.tsv",
            "mask": func_dir / f"{prefix}_{space_tag}_desc-brain_mask.nii.gz",
        })
    return runs


def denoise_run(run, args, log):
    """Denoise one run. Appends one entry per step to `log`; raises on failure."""
    prefix = run["prefix"]
    bold, mask = run["bold"], run["mask"]
    gs_tag = "_removed_GS" if args.gsr == "y" else "_with_GS"

    regfilt_out = Path(str(bold).replace("_bold.nii.gz", gs_tag + ".nii.gz"))
    smoothed = with_suffix(regfilt_out, "_smoothed")
    filtered = with_suffix(smoothed, "_filtered")
    mean_img = with_suffix(filtered, "_mean_image")
    demeaned = with_suffix(filtered, "_demeaned")
    masked = with_suffix(demeaned, "_masked")

    if masked.exists() and not args.overwrite:
        print(f"[{prefix}] final output exists, skipping (use --overwrite to redo): {masked.name}")
        log.append({"run": prefix, "step": "all", "status": "skipped",
                    "detail": "final output already exists"})
        return

    def record(step, detail="ok"):
        log.append({"run": prefix, "step": step, "status": "ok", "detail": detail})

    def fail(step, exc):
        log.append({"run": prefix, "step": step, "status": "FAILED", "detail": str(exc)})
        print(f"[{prefix}] FAILED at {step}: {exc}", file=sys.stderr)
        raise exc

    # ---- 0. inputs and design matrix ---------------------------------------
    step = "prepare_confounds"
    try:
        for key in ("confounds", "mask"):
            if not run[key].exists():
                raise FileNotFoundError(f"{key} file not found: {run[key]}")
        variables = BASE_CONFOUNDS + (GSR_CONFOUNDS if args.gsr == "y" else [])
        conf = pd.read_csv(run["confounds"], sep="\t")
        if args.confound_skip_rows:
            conf = conf.iloc[args.confound_skip_rows:].reset_index(drop=True)
        missing = [v for v in variables if v not in conf.columns]
        if missing:
            raise KeyError(f"confounds missing columns: {missing}")
        design = conf[variables].copy()

        # fMRIPrep leaves the first row of every derivative column empty (n/a).
        n_nan = int(design.isna().sum().sum())
        design = design.fillna(0.0)

        n_vols = nib.load(str(bold)).shape[3]
        if len(design) != n_vols:
            raise ValueError(f"{len(design)} confound rows but {n_vols} BOLD volumes "
                             f"(check --confound-skip-rows)")

        design_path = run["confounds"].parent / f"{prefix}_filtered_confounds.tsv"
        design.to_csv(design_path, sep="\t", header=False, index=False)
        (run["confounds"].parent / f"{prefix}_filtered_confounds_columns.txt").write_text(
            "\n".join(variables) + "\n")
        col_idx = ",".join(str(i + 1) for i in range(len(variables)))   # 1-based
        record(step, f"{len(variables)} regressors, {n_vols} volumes, "
                     f"{n_nan} empty values set to 0")
    except Exception as exc:
        fail(step, exc)

    # ---- 1. nuisance regression --------------------------------------------
    step = "regfilt"
    try:
        sh(["fsl_regfilt", "-i", bold, "-d", design_path, "-o", regfilt_out, "-f", col_idx])
        record(step)
    except Exception as exc:
        fail(step, exc)

    # ---- 2. smoothing ------------------------------------------------------
    step = "smooth"
    try:
        sh(["fslmaths", regfilt_out, "-s", args.smooth_sigma, smoothed])
        record(step, f"sigma={args.smooth_sigma} mm")
    except Exception as exc:
        fail(step, exc)

    # ---- 3. band-pass (sigmas in volumes, high-pass first) -----------------
    step = "bandpass"
    try:
        tr = float(nib.load(str(smoothed)).header.get_zooms()[3])
        hp_sigma = 1 / (2 * tr * args.highpass)
        lp_sigma = 1 / (2 * tr * args.lowpass)
        sh(["fslmaths", smoothed, "-bptf", hp_sigma, lp_sigma, filtered])
        record(step, f"TR={tr}s, {args.highpass}-{args.lowpass} Hz")
    except Exception as exc:
        fail(step, exc)

    # ---- 4. de-mean --------------------------------------------------------
    step = "demean"
    try:
        sh(["fslmaths", filtered, "-Tmean", mean_img])
        sh(["fslmaths", filtered, "-sub", mean_img, demeaned])
        record(step)
    except Exception as exc:
        fail(step, exc)

    # ---- 5. mask -----------------------------------------------------------
    step = "mask"
    try:
        sh(["fslmaths", demeaned, "-mas", mask, masked])
        record(step, mask.name)
    except Exception as exc:
        fail(step, exc)

    # ---- clean up ----------------------------------------------------------
    if not args.keep_intermediates:
        for tmp in (smoothed, filtered, mean_img, demeaned):
            tmp.unlink(missing_ok=True)
        record("cleanup", "intermediates removed")
    print(f"[{prefix}] done: {masked.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path,
                    help="fMRIPrep directory that contains func/ "
                         "(e.g. .../sub-901/ses-1)")
    ap.add_argument("--gsr", choices=["y", "n"], required=True,
                    help="y = also regress the global signal (output tag '_removed_GS'); "
                         "n = keep it ('_with_GS'). Required: this is a methods choice.")
    ap.add_argument("--space-tag", default=DEFAULT_SPACE_TAG,
                    help="output-space tag in the fMRIPrep file names "
                         f"(default: {DEFAULT_SPACE_TAG})")
    ap.add_argument("--smooth-sigma", default="2.123",
                    help="Gaussian sigma in mm (default 2.123 = 5 mm FWHM)")
    ap.add_argument("--highpass", type=float, default=0.009, help="Hz (default 0.009)")
    ap.add_argument("--lowpass", type=float, default=0.08, help="Hz (default 0.08)")
    ap.add_argument("--confound-skip-rows", type=int, default=0,
                    help="drop this many leading rows of the confounds file, if that many "
                         "volumes were removed from the BOLD data (default 0)")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo runs whose final output already exists")
    ap.add_argument("--keep-intermediates", action="store_true")
    args = ap.parse_args()

    for tool in ("fsl_regfilt", "fslmaths"):
        if shutil.which(tool) is None:
            sys.exit(f"ERROR: '{tool}' not found on PATH (load FSL first)")

    func_dir = args.input_dir / "func"
    if not func_dir.is_dir():
        sys.exit(f"ERROR: no func/ directory in {args.input_dir}")
    runs = find_runs(func_dir, args.space_tag)
    if not runs:
        sys.exit(f"ERROR: no '*_{args.space_tag}_desc-preproc_bold.nii.gz' files in {func_dir}")
    print(f"{len(runs)} run(s) found in {func_dir}; global signal regression: {args.gsr}")

    log, n_failed = [], 0
    for run in runs:
        try:
            denoise_run(run, args, log)
        except Exception:
            n_failed += 1          # already logged; continue with the next run

    log_path = args.input_dir / "preproc_denoising_log.tsv"
    pd.DataFrame(log).to_csv(log_path, sep="\t", index=False)
    print(f"Log written to {log_path}")
    print(f"preproc over for {args.input_dir}: {len(runs) - n_failed}/{len(runs)} run(s) succeeded")
    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()
