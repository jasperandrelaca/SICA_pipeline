#!/usr/bin/env python3
"""
08_yeo_coherence.py
===================
Network coherence from the standard (non-precision) Yeo 2011 7-network atlas,
for comparison with the participant-specific MELODIC approach.

Steps
-----
1. Fetch the Yeo 2011 7-network atlas (``thick_7``) from nilearn.
2. Resample it onto the grid of our preprocessed BOLD data (MNIPediatricAsym,
   2 mm) with nearest-neighbour interpolation. NOTE: this is a resampling onto
   the BOLD grid only. No spatial normalisation is applied between adult MNI152
   and MNIPediatricAsym (contrast with 03_nct_network_correspondence.py, which
   registers the two templates with ANTs).
3. For every BOLD run in each participant's subject_level_files.txt and for every
   network: coherence = mean Pearson correlation between each in-network voxel's
   time series and the network's mean time series.

Voxels containing NaN/Inf or with zero variance are excluded; if the network's mean
time series is non-finite or constant, the coherence is NaN.

BOLD files are used exactly as listed (no additional filtering, smoothing or
confound regression is applied here).

Usage:
    python scripts/08_yeo_coherence.py --output yeo_coherence_results.csv
"""

import argparse
import os
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.image import resample_to_img, index_img
from tqdm import tqdm

import catd_config

warnings.filterwarnings("ignore")

# Yeo 2011 7-network label values used in this study
YEO_NETWORKS = {
    "DMN": 7,        # Default Mode
    "Salience": 4,   # Ventral Attention / Salience
    "FPN": 6,        # Frontoparietal Control
    "SMN": 2,        # Somatomotor
}
# Full label set: 1 Visual, 2 Somatomotor, 3 Dorsal Attention,
# 4 Ventral Attention/Salience, 5 Limbic, 6 Frontoparietal, 7 Default Mode


# ============================================================================
# Atlas preparation
# ============================================================================
def fetch_and_prepare_yeo_atlas(reference_img_path, output_dir=None):
    """
    Fetch the Yeo atlas and resample it onto the grid of ``reference_img_path``.

    Returns (atlas_data ndarray, atlas nibabel image). The resampled atlas is
    cached as Yeo2011_7Networks_MNIPediatricAsym_2mm.nii.gz in ``output_dir``.
    """
    output_dir = output_dir or os.getcwd()
    cached = os.path.join(output_dir, "Yeo2011_7Networks_MNIPediatricAsym_2mm.nii.gz")
    if os.path.exists(cached):
        print(f"Found existing resampled atlas: {cached}")
        img = nib.load(cached)
        return img.get_fdata(), img

    print("\n" + "=" * 80 + "\nPREPARING YEO ATLAS\n" + "=" * 80)

    print("\n1. Fetching Yeo 2011 7-network atlas from nilearn...")
    yeo_img = nib.load(datasets.fetch_atlas_yeo_2011()["thick_7"])
    print(f"   Original atlas shape: {yeo_img.shape}")
    if len(yeo_img.shape) == 4:
        yeo_img = index_img(yeo_img, 0)
        print(f"   Squeezed to 3D: {yeo_img.shape}")

    print(f"\n2. Loading reference image: {reference_img_path}")
    if not os.path.exists(reference_img_path):
        raise FileNotFoundError(f"Reference image not found: {reference_img_path}")
    reference = nib.load(reference_img_path)
    if len(reference.shape) == 4:
        reference = index_img(reference, 0)
    print(f"   Reference shape: {reference.shape}")

    print("\n3. Resampling atlas onto the reference grid (nearest neighbour)...")
    resampled = resample_to_img(yeo_img, reference, interpolation="nearest")  # preserves labels
    data = resampled.get_fdata()
    print(f"   Resampled shape: {resampled.shape}")
    print(f"   Unique labels: {np.unique(data)}  (expected 0 = background, 1-7 = networks)")
    for label in range(1, 8):
        print(f"   Network {label}: {int(np.sum(data == label)):,} voxels")

    os.makedirs(output_dir, exist_ok=True)
    nib.save(resampled, cached)
    print(f"\n4. Saved: {cached}\n" + "=" * 80 + "\n")
    return data, resampled


def get_reference_image(subject_ids, list_dir):
    """First existing BOLD file among the first five participants (defines the grid)."""
    print("Finding reference image for atlas resampling...")
    for sid in subject_ids[:5]:
        files = catd_config.read_run_list(sid, list_dir)
        if files and os.path.exists(files[0]):
            print(f"Using reference image from {sid}: {files[0]}\n")
            return files[0]
    raise RuntimeError("Could not find a valid reference image!")


# ============================================================================
# Coherence
# ============================================================================
def calculate_network_coherence(fmri_data, network_mask):
    """
    Mean voxel-to-network-mean Pearson correlation.

    fmri_data    : ndarray (x, y, z, t)
    network_mask : ndarray (x, y, z), non-zero inside the network

    Returns (coherence, n_voxels) where n_voxels counts all voxels in the mask
    (before NaN / zero-variance exclusions).
    """
    mask = np.asarray(network_mask).astype(bool)
    n_voxels = int(mask.sum())
    if n_voxels == 0:
        return np.nan, 0

    ts = fmri_data[mask]                 # (n_voxels, t)
    mean_ts = ts.mean(axis=0)

    # A single non-finite voxel makes the network mean non-finite -> NaN result.
    if not np.all(np.isfinite(mean_ts)) or mean_ts.std() == 0:
        return np.nan, n_voxels

    keep = np.all(np.isfinite(ts), axis=1) & (ts.std(axis=1) > 0)
    ts = ts[keep]
    if ts.shape[0] == 0:
        return np.nan, n_voxels

    ts_c = ts - ts.mean(axis=1, keepdims=True)
    m_c = mean_ts - mean_ts.mean()
    r = (ts_c @ m_c) / (np.sqrt((ts_c ** 2).sum(axis=1)) * np.sqrt((m_c ** 2).sum()))
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.nan, n_voxels
    return float(np.mean(r)), n_voxels


def process_single_scan(fmri_path, yeo_atlas, network_labels):
    """Coherence and voxel count for each network in one BOLD file."""
    nan_result = {
        **{f"{n}_coherence": np.nan for n in network_labels},
        **{f"{n}_n_voxels": 0 for n in network_labels},
    }
    try:
        fmri_data = nib.load(fmri_path).get_fdata()
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading {fmri_path}: {exc}")
        return nan_result
    if fmri_data.ndim != 4:
        print(f"Warning: expected 4D data, got {fmri_data.ndim}D for {fmri_path}")
        return nan_result

    result = {}
    for name, label in network_labels.items():
        coherence, n_vox = calculate_network_coherence(fmri_data, yeo_atlas == label)
        result[f"{name}_coherence"] = coherence
        result[f"{name}_n_voxels"] = n_vox
    return result


# ============================================================================
# Driver
# ============================================================================
def process_all_subjects(subject_ids, list_dir, yeo_atlas, network_labels, output_csv):
    results = []
    print(f"\n{'=' * 80}\nPROCESSING {len(subject_ids)} SUBJECTS\n{'=' * 80}\n")

    for sid in tqdm(subject_ids, desc="Processing subjects"):
        files = catd_config.read_run_list(sid, list_dir)
        if not files:
            print(f"No files found for {sid}")
            continue
        for path in files:
            if not os.path.exists(path):
                print(f"Warning: File not found: {path}")
                continue
            row = {
                "subject_id": sid.replace("sub-", ""),
                "session": catd_config.parse_session(path, default=None),
                "file_path": path,
            }
            row.update(process_single_scan(path, yeo_atlas, network_labels))
            results.append(row)

    df = pd.DataFrame(results)
    coh_cols = [f"{n}_coherence" for n in network_labels]
    vox_cols = [f"{n}_n_voxels" for n in network_labels]
    df = df[["subject_id", "session"] + coh_cols + vox_cols + ["file_path"]]
    df.to_csv(output_csv, index=False)

    print(f"\n{'=' * 80}\nRESULTS SAVED\n{'=' * 80}")
    print(f"Output file: {output_csv}\nTotal scans processed: {len(df)}")
    print("\nSUMMARY STATISTICS")
    for net in network_labels:
        c, v = df[f"{net}_coherence"], df[f"{net}_n_voxels"]
        print(f"\n{net}:")
        print(f"  Coherence - mean {c.mean():.4f}, SD {c.std():.4f}, range [{c.min():.4f}, {c.max():.4f}]")
        print(f"  Voxels    - mean {v.mean():.0f}, SD {v.std():.0f}")
        print(f"  Missing: {c.isna().sum()} scans")
    return df


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="yeo_coherence_results.csv",
                        help="Output CSV (default: yeo_coherence_results.csv)")
    parser.add_argument("--atlas-dir", default=None,
                        help="Where to save/load the resampled atlas (default: current directory)")
    parser.add_argument("--melodic-dir", default=None,
                        help="Used only to enumerate subjects (default: $CATD_MELODIC_DIR)")
    parser.add_argument("--fmriprep-list-dir", default=None, help="default: $CATD_FMRIPREP_LIST_DIR")
    args = parser.parse_args()

    melodic_dir = Path(args.melodic_dir) if args.melodic_dir else catd_config.melodic_dir()
    list_dir = Path(args.fmriprep_list_dir) if args.fmriprep_list_dir else catd_config.fmriprep_list_dir()

    print("\n" + "=" * 80 + "\nYEO NETWORK COHERENCE\nNon-precision method for comparison with MELODIC\n" + "=" * 80)
    print("\nNetworks analysed:")
    for name, label in YEO_NETWORKS.items():
        print(f"  {name}: Yeo label {label}")

    subject_ids = catd_config.find_subject_ids(melodic_dir)
    print(f"\nFound {len(subject_ids)} subjects")
    if not subject_ids:
        print("ERROR: No subjects found!")
        return

    reference = get_reference_image(subject_ids, list_dir)
    atlas_data, _ = fetch_and_prepare_yeo_atlas(reference, output_dir=args.atlas_dir)

    process_all_subjects(subject_ids, list_dir, atlas_data, YEO_NETWORKS, args.output)
    print("\n" + "=" * 80 + "\nPROCESSING COMPLETE\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
