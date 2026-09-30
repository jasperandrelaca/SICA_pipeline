#!/usr/bin/env python3
"""
07_between_network_connectivity.py
==================================
Between-network connectivity (BNC) for DMN, Salience (SN) and Frontoparietal (FPN)
using the participant-specific masks created by 06_extract_dr_metrics.py.

Two analyses per participant x session (one row per BOLD run):

  PRIMARY masks
      The single top-ranked IC mask per network:
      {sub}_DMN_mask.nii.gz, {sub}_Salience_mask.nii.gz, {sub}_Frontoparietal_mask.nii.gz

  COMBINED masks (primary OR secondary)
      Union of the primary and secondary (``_b``) masks, capturing the network's
      full distributed footprint when it is split across two components. If a
      participant has no secondary mask for a network, that network falls back to
      the primary mask alone for the combined analysis (recorded in the
      ``*_mask_combined`` columns).

For each mask set and run:
  1. average BOLD time series within each network mask
  2. Pearson r between network pairs (DMN-SN, DMN-FPN, SN-FPN)
  3. Fisher z-transform (r clipped to +/-0.9999 before arctanh)

Prerequisite: run 06_extract_dr_metrics.py first (it writes the masks). Use its
--overwrite-masks flag if masks from an earlier version of the pipeline exist.

Input BOLD: the files listed in each participant's subject_level_files.txt,
used exactly as listed (no additional filtering, smoothing or confound
regression is applied here).

Output columns
--------------
subject | session
DMN_SN_r,  DMN_SN_Z,  DMN_FPN_r,  DMN_FPN_Z,  SN_FPN_r,  SN_FPN_Z        (primary)
DMN_SN_r_comb, DMN_SN_Z_comb, DMN_FPN_r_comb, DMN_FPN_Z_comb,
SN_FPN_r_comb, SN_FPN_Z_comb                                              (combined)
DMN_mask_combined, SN_mask_combined, FPN_mask_combined
    True  = a secondary mask existed and the combined mask is a true union
    False = secondary mask missing; the "combined" analysis used the primary mask only
"""

import argparse
import os
import sys
import traceback
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import maskers
from nilearn.image import resample_to_img
from scipy import stats

import catd_config

warnings.filterwarnings("ignore", category=UserWarning)

# Primary mask file suffixes (created by step 06)
PRIMARY_SUFFIXES = {
    "DMN": "_DMN_mask.nii.gz",
    "SN": "_Salience_mask.nii.gz",
    "FPN": "_Frontoparietal_mask.nii.gz",
}
# Secondary mask file suffixes (created by step 06)
SECONDARY_SUFFIXES = {
    "DMN": "_DMN_b_mask.nii.gz",
    "SN": "_Salience_b_mask.nii.gz",
    "FPN": "_Frontoparietal_b_mask.nii.gz",
}
NETWORK_PAIRS = [("DMN", "SN"), ("DMN", "FPN"), ("SN", "FPN")]


# ── Logging ───────────────────────────────────────────────────────────────────
def log_exception(context_msg, exc):
    print(f"  ERROR ({context_msg}):")
    print(f"    Exception type: {type(exc).__name__}")
    print(f"    str(e):         {str(exc).strip() or '(empty)'}")
    print(f"    repr(e):        {repr(exc)}")
    print("    Full traceback:")
    for block in traceback.format_exception(type(exc), exc, exc.__traceback__):
        for line in block.rstrip().split("\n"):
            print(f"      {line}")
    sys.stdout.flush()


# ── Mask helpers ──────────────────────────────────────────────────────────────
def load_primary_masks(subject_id, masks_dir):
    """{net: path} of primary masks, or None if any is missing."""
    mask_dir = Path(masks_dir) / subject_id
    masks = {}
    for net, suffix in PRIMARY_SUFFIXES.items():
        path = mask_dir / f"{subject_id}{suffix}"
        if not path.exists():
            print(f"  WARNING: Missing primary {net} mask: {path}")
            return None
        masks[net] = str(path)
    return masks


def build_combined_masks(subject_id, primary_masks, masks_dir):
    """
    Per network, OR the primary and secondary masks and save the union next to them.

    Returns (combined_paths, fallback_flags): fallback_flags[net] is True when the
    secondary mask was missing (or could not be combined) and the primary was used.
    """
    mask_dir = Path(masks_dir) / subject_id
    combined_paths, fallback_flags = {}, {}

    for net, primary_path in primary_masks.items():
        secondary_path = mask_dir / f"{subject_id}{SECONDARY_SUFFIXES[net]}"
        combined_out = mask_dir / f"{subject_id}_{net}_combined_mask.nii.gz"

        if not secondary_path.exists():
            print(f"  NOTE: No secondary {net} mask found - using primary only for combined analysis")
            combined_paths[net] = primary_path
            fallback_flags[net] = True
            continue

        try:
            primary_img = nib.load(primary_path)
            secondary_img = nib.load(str(secondary_path))
            primary_data = primary_img.get_fdata().astype(bool)
            secondary_data = secondary_img.get_fdata().astype(bool)

            # If the grids differ, resample the secondary mask onto the primary's grid
            if (primary_data.shape != secondary_data.shape
                    or not np.allclose(primary_img.affine, secondary_img.affine, atol=1e-3)):
                secondary_img = resample_to_img(secondary_img, primary_img, interpolation="nearest")
                secondary_data = secondary_img.get_fdata().astype(bool)

            combined = (primary_data | secondary_data).astype(np.uint8)
            img = nib.Nifti1Image(combined, affine=primary_img.affine, header=primary_img.header)
            img.set_data_dtype(np.uint8)
            nib.save(img, str(combined_out))

            print(f"  Combined {net} mask: {primary_data.sum()} + {secondary_data.sum()} voxels "
                  f"-> {combined.sum()} voxels (union) -> {combined_out}")
            combined_paths[net] = str(combined_out)
            fallback_flags[net] = False

        except Exception as exc:  # noqa: BLE001
            log_exception(f"{subject_id} building combined {net} mask "
                          f"[primary={primary_path}, secondary={secondary_path}]", exc)
            combined_paths[net] = primary_path
            fallback_flags[net] = True

    return combined_paths, fallback_flags


# ── Time series / BNC ─────────────────────────────────────────────────────────
def extract_avg_timeseries(func_path, mask_path):
    """
    Mean time series over all in-mask voxels of a 4-D file (float32 to save memory).

    Masks are created on the same grid as the BOLD data (both MNIPediatricAsym 2 mm),
    so no resampling is needed. (An earlier version passed ``resampling_target="data"``
    to NiftiMasker; that argument does not exist for NiftiMasker, was silently ignored
    by older nilearn releases, and raises a TypeError on current ones.)
    """
    masker = maskers.NiftiMasker(
        mask_img=mask_path,
        standardize=False,
        memory_level=0,
        dtype="float32",
    )
    return masker.fit_transform(func_path).mean(axis=1)


def pearson_to_fisher_z(ts_a, ts_b):
    r, _ = stats.pearsonr(ts_a, ts_b)
    return float(r), float(np.arctanh(np.clip(r, -0.9999, 0.9999)))


def extract_timeseries_for_masks(func_path, masks, img_shape, subject_id, session):
    """{net: mean time series} for every mask in ``masks``, or None on failure."""
    ts = {}
    for net, mask_path in masks.items():
        try:
            ts[net] = extract_avg_timeseries(func_path, mask_path)
            if ts[net].size == 0:
                print(f"  ERROR: {net} mask produced empty timeseries: {mask_path}")
                return None
            if np.all(ts[net] == 0) or np.isnan(ts[net]).all():
                print(f"  ERROR: {net} timeseries is all-zero/NaN: {mask_path}")
                return None
        except Exception as exc:  # noqa: BLE001
            try:
                mask_shape = nib.load(mask_path).shape
            except Exception:  # noqa: BLE001
                mask_shape = "(unreadable)"
            log_exception(f"{subject_id} {session} extracting {net} timeseries "
                          f"[mask shape={mask_shape}, func shape={img_shape}, mask path={mask_path}]", exc)
            return None
    return ts


def compute_bnc(ts, col_suffix=""):
    """Pairwise r and Fisher Z for all NETWORK_PAIRS; col_suffix '' (primary) or '_comb'."""
    result = {}
    for net_a, net_b in NETWORK_PAIRS:
        try:
            r, z = pearson_to_fisher_z(ts[net_a], ts[net_b])
        except Exception as exc:  # noqa: BLE001
            log_exception(f"computing {net_a}-{net_b} correlation{col_suffix}", exc)
            return None
        result[f"{net_a}_{net_b}_r{col_suffix}"] = round(r, 6)
        result[f"{net_a}_{net_b}_Z{col_suffix}"] = round(z, 6)
    return result


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run(melodic_dir, list_dir, masks_dir, output_csv):
    subject_ids = catd_config.find_subject_ids(melodic_dir)
    print(f"Found {len(subject_ids)} subjects\n")
    rows, skipped = [], []

    for sub in subject_ids:
        print(f"Processing {sub}...")

        primary_masks = load_primary_masks(sub, masks_dir)
        if primary_masks is None:
            skipped.append((sub, "missing_primary_mask"))
            continue

        combined_masks, fallback_flags = build_combined_masks(sub, primary_masks, masks_dir)

        func_files = catd_config.read_run_list(sub, list_dir)
        if not func_files:
            skipped.append((sub, "no_func_files"))
            continue

        for func_path in func_files:
            if not os.path.exists(func_path):
                print(f"  WARNING: Functional file not found: {func_path}")
                continue

            session = catd_config.parse_session(func_path)
            print(f"  Session {session}: {os.path.basename(func_path)}")

            # Sanity checks
            try:
                if os.path.getsize(func_path) == 0:
                    print(f"  ERROR: 0-byte file: {func_path}")
                    skipped.append((sub, f"{session}_zero_byte_file"))
                    continue
                if not os.access(func_path, os.R_OK):
                    print(f"  ERROR: Not readable: {func_path}")
                    skipped.append((sub, f"{session}_permission_denied"))
                    continue
            except OSError as exc:
                log_exception(f"{sub} {session} stat check", exc)
                skipped.append((sub, f"{session}_stat_error"))
                continue

            try:
                img_shape = nib.load(func_path).shape
            except Exception as exc:  # noqa: BLE001
                log_exception(f"{sub} {session} header read", exc)
                skipped.append((sub, f"{session}_header_error"))
                continue

            if len(img_shape) != 4 or img_shape[3] < 2:
                print(f"  ERROR: Unexpected shape {img_shape}: {func_path}")
                skipped.append((sub, f"{session}_bad_shape"))
                continue

            ts_primary = extract_timeseries_for_masks(func_path, primary_masks, img_shape, sub, session)
            if ts_primary is None:
                skipped.append((sub, f"{session}_primary_extraction_error"))
                continue
            ts_combined = extract_timeseries_for_masks(func_path, combined_masks, img_shape, sub, session)
            if ts_combined is None:
                skipped.append((sub, f"{session}_combined_extraction_error"))
                continue

            bnc_primary = compute_bnc(ts_primary, col_suffix="")
            if bnc_primary is None:
                skipped.append((sub, f"{session}_primary_bnc_error"))
                continue
            bnc_combined = compute_bnc(ts_combined, col_suffix="_comb")
            if bnc_combined is None:
                skipped.append((sub, f"{session}_combined_bnc_error"))
                continue

            rows.append({
                "subject": sub,
                "session": session,
                **bnc_primary,
                **bnc_combined,
                # True = secondary mask existed (true union); False = primary-only fallback
                "DMN_mask_combined": not fallback_flags["DMN"],
                "SN_mask_combined": not fallback_flags["SN"],
                "FPN_mask_combined": not fallback_flags["FPN"],
            })
            print(f"    Primary  - DMN-SN Z={bnc_primary['DMN_SN_Z']:.3f}  "
                  f"DMN-FPN Z={bnc_primary['DMN_FPN_Z']:.3f}  SN-FPN Z={bnc_primary['SN_FPN_Z']:.3f}")
            print(f"    Combined - DMN-SN Z={bnc_combined['DMN_SN_Z_comb']:.3f}  "
                  f"DMN-FPN Z={bnc_combined['DMN_FPN_Z_comb']:.3f}  SN-FPN Z={bnc_combined['SN_FPN_Z_comb']:.3f}")

    if not rows:
        print("\nNo data collected - check warnings above.")
        return

    columns = [
        "subject", "session",
        "DMN_SN_r", "DMN_SN_Z", "DMN_FPN_r", "DMN_FPN_Z", "SN_FPN_r", "SN_FPN_Z",
        "DMN_SN_r_comb", "DMN_SN_Z_comb", "DMN_FPN_r_comb", "DMN_FPN_Z_comb",
        "SN_FPN_r_comb", "SN_FPN_Z_comb",
        "DMN_mask_combined", "SN_mask_combined", "FPN_mask_combined",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"\nDone. {len(df)} rows written to: {output_csv}")

    print("\nCombined-mask summary (True = true union, False = primary-only fallback):")
    for net in ["DMN", "SN", "FPN"]:
        col = f"{net}_mask_combined"
        print(f"  {net}: {df[col].sum()} obs used the union mask, {(~df[col]).sum()} fell back to primary")

    if skipped:
        print(f"\nSkipped {len(skipped)} subject x session entries:")
        for sub, reason in skipped:
            print(f"  {sub} - {reason}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-csv", default="between_network_connectivity.csv",
                        help="Output CSV (default: ./between_network_connectivity.csv)")
    parser.add_argument("--melodic-dir", default=None, help="default: $CATD_MELODIC_DIR")
    parser.add_argument("--fmriprep-list-dir", default=None, help="default: $CATD_FMRIPREP_LIST_DIR")
    parser.add_argument("--masks-dir", default=None, help="default: $CATD_MASKS_DIR")
    args = parser.parse_args()

    run(
        melodic_dir=Path(args.melodic_dir) if args.melodic_dir else catd_config.melodic_dir(),
        list_dir=Path(args.fmriprep_list_dir) if args.fmriprep_list_dir else catd_config.fmriprep_list_dir(),
        masks_dir=Path(args.masks_dir) if args.masks_dir else catd_config.masks_dir(),
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
