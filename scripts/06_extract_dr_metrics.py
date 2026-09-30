#!/usr/bin/env python3
"""
06_extract_dr_metrics.py
========================
Create participant-specific network masks from MELODIC components and extract
metrics from dual-regression stage-2 maps.

Component selection (DMN, Salience, Frontoparietal)
---------------------------------------------------
Every IC is scored against each network from the NCT output (see nct_common.py).
Components are then assigned to networks in three sequential rounds of *global*
maximum-weight bipartite matching (Hungarian algorithm,
``scipy.optimize.linear_sum_assignment``):

    tier 1 (primary)   : one IC per network, chosen jointly so that no IC is
                         claimed by two networks and the summed score is maximal
    tier 2 (secondary) : same, over the ICs not used in tier 1  -> suffix "_b"
    tier 3 (tertiary)  : same, over the ICs not used in tiers 1-2 -> suffix "_c"

This captures fragmented networks: when a canonical network is split across
several ICs in a participant's decomposition, the additional meaningful
components are retained rather than only the top scorer. A network gets ``None``
for a tier when no unclaimed candidate that clears the candidate gate remains.

NOTE: for these three networks the assignments are re-derived here directly from
the NCT tables; the corresponding columns of subject_network_assignments.csv
(step 04) are not used, and step 04's *fallback* assignments are NOT applied
(participants with no gate-passing IC get NaN metrics for that network).
Other networks (Dorsal Attention, Visual, Somatomotor) use the step-04 winner only.

Mask creation
-------------
For each selected IC, the mask is the top ``--thresh-percentile`` (default 5%,
i.e. >= 95th percentile) of the component's positive voxels in the ORIGINAL
MNIPediatricAsym-space melodic_IC.nii.gz. Masks are saved as
``{masks_root}/{sub}/{sub}_{Network}{suffix}_mask.nii.gz`` (suffix '', '_b', '_c').
Existing masks are re-used unless ``--overwrite-masks`` is given, so pass that
flag whenever the assignment logic or thresholds change.

Metrics (per run/session, per network and tier)
-----------------------------------------------
Taken from the dual-regression stage-2 volume of the matching component, inside
the mask: mean, variance, standard deviation, and the size (voxels) of the largest
connected cluster of voxels >= ``--dr-threshold`` (default 2.3).

Output: TSV with columns
    subject_id, session, {Net}_{mean,var,std,largest_cluster},
    {Net}_b_*, {Net}_c_*  (DMN/Salience/Frontoparietal),
    {Net}_*               (other networks; primary only)

Usage:
    python scripts/06_extract_dr_metrics.py \\
        --network-assignments $CATD_ASSIGNMENTS_DIR/subject_network_assignments.csv \\
        --overwrite-masks
"""

import os
import re
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

try:
    from nilearn.image import resample_to_img
except Exception:  # noqa: BLE001
    resample_to_img = None

import catd_config
from nct_common import (
    RANKED_NETWORKS,
    CANDIDATE_MIN_DICE,
    CANDIDATE_MAX_P,
    calculate_ic_score,
    passes_gate,
    load_subject_ic_tables,
)

# =======================
# CONFIGURABLE PARAMETERS
# =======================
THRESH_PERCENTILE = 95.0  # percentile threshold for mask creation
POSITIVE_ONLY = True      # percentile computed over positive voxels only
DR_THRESHOLD = 2.3        # voxel threshold for the largest-cluster metric

# Tier suffixes: primary keeps un-suffixed column names; secondary = _b; tertiary = _c
TIER_SUFFIXES = ["", "_b", "_c"]


# =============================
# Ranked (primary / secondary / tertiary) component assignment
# =============================
def load_subject_ic_scores(subject_id, networks, nct_results_dir):
    """
    Score every IC of one subject against each network in ``networks``.

    Returns {ic_num: {network: score_dict}} containing only (IC, network) pairs
    that clear the candidate gate; ICs that clear it for no network are omitted.
    """
    tables = load_subject_ic_tables(Path(nct_results_dir) / subject_id)

    ic_scores = {}
    for ic_num, ic_df in tables.items():
        net_scores = {}
        for net in networks:
            s = calculate_ic_score(ic_df, net)
            if passes_gate(s, net, CANDIDATE_MIN_DICE, CANDIDATE_MAX_P):
                net_scores[net] = s
        if net_scores:
            ic_scores[ic_num] = net_scores
    return ic_scores


def global_assign_tier(ic_scores, networks, excluded_ics):
    """
    Assign one IC per network for one tier by maximum-weight bipartite matching.

    ic_scores    : {ic_num: {network: score_dict}}  (full candidate pool)
    networks     : networks to assign
    excluded_ics : ICs consumed by earlier tiers

    Returns {network: ic_num or None}. ``None`` means no qualifying candidate.
    """
    available = {ic: sc for ic, sc in ic_scores.items() if ic not in excluded_ics}
    if not available:
        return {net: None for net in networks}

    ic_list = sorted(available)
    n_nets, n_ics = len(networks), len(ic_list)

    # Cost = -score (linear_sum_assignment minimises). ICs that did not clear the
    # gate for a network contribute 0, i.e. "no match".
    cost = np.zeros((n_nets, n_ics))
    for i, net in enumerate(networks):
        for j, ic in enumerate(ic_list):
            cost[i, j] = -available[ic].get(net, {}).get("score", 0.0)

    # Pad with dummy ICs (score 0) if there are fewer ICs than networks.
    if n_ics < n_nets:
        cost = np.hstack([cost, np.zeros((n_nets, n_nets - n_ics))])
        ic_list_padded = ic_list + [None] * (n_nets - n_ics)
    else:
        ic_list_padded = ic_list

    row_ind, col_ind = linear_sum_assignment(cost)

    assignments = {}
    for i, net in enumerate(networks):
        if i < len(row_ind):
            ic = ic_list_padded[col_ind[i]]
            if ic is None or available[ic].get(net, {}).get("score", 0.0) == 0.0:
                assignments[net] = None
            else:
                assignments[net] = ic
        else:
            assignments[net] = None
    return assignments


def resolve_ranked_assignments(subject_id, networks, nct_results_dir):
    """
    Primary, secondary and tertiary IC assignments via three sequential rounds
    of ``global_assign_tier``. Returns a list of 3 dicts {network: ic or None}.
    """
    ic_scores = load_subject_ic_scores(subject_id, networks, nct_results_dir)

    tiers, excluded = [], set()
    for _ in range(3):
        assignment = global_assign_tier(ic_scores, networks, excluded)
        tiers.append(assignment)
        excluded.update(ic for ic in assignment.values() if ic is not None)
    return tiers


# =============================
# NIfTI / mask utilities
# =============================
def load_nifti(path: Path) -> nib.Nifti1Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing NIfTI: {path}")
    return nib.load(str(path))


def save_mask_like(ref_img, mask_bool, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(mask_bool.astype(np.uint8), affine=ref_img.affine, header=ref_img.header)
    img.set_data_dtype(np.uint8)
    nib.save(img, str(out_path))


def find_largest_cluster_size(mask_bool) -> int:
    """Voxel count of the largest connected component (scipy default connectivity)."""
    if mask_bool.sum() == 0:
        return 0
    labeled, n_features = ndimage.label(mask_bool)
    if n_features == 0:
        return 0
    sizes = ndimage.sum(mask_bool, labeled, range(1, n_features + 1))
    return int(np.max(sizes))


def make_mask_for_network(subject_id, network_name, ic_index, masks_root, melodic_template,
                          thresh_percentile, positive_only, mask_suffix="", overwrite=False):
    """
    Binary mask of the strongest voxels of one MELODIC IC.
    ``mask_suffix``: '' (primary), '_b' (secondary), '_c' (tertiary).
    """
    ic_path = Path(melodic_template.format(subject_id=subject_id))
    img = load_nifti(ic_path)
    data = img.get_fdata()

    if data.ndim != 4:
        raise ValueError(f"Expected 4D MELODIC IC image for {subject_id}, got {data.shape}")
    if ic_index < 0 or ic_index >= data.shape[-1]:
        raise ValueError(f"IC index {ic_index} out of range for {subject_id} "
                         f"(max {data.shape[-1] - 1})")

    vol = data[..., ic_index]
    if positive_only:
        pos_vals = vol[vol > 0]
        thr = np.percentile(pos_vals, thresh_percentile) if pos_vals.size > 0 else np.inf
    else:
        thr = np.percentile(vol, thresh_percentile)
    mask_bool = vol >= thr

    out_dir = masks_root / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subject_id}_{network_name}{mask_suffix}_mask.nii.gz"

    if out_path.exists() and not overwrite:
        return out_path

    save_mask_like(img, mask_bool, out_path)
    tier = {"": "primary", "_b": "secondary", "_c": "tertiary"}.get(mask_suffix, mask_suffix)
    print(f"  Created {tier} mask: {network_name} (IC{ic_index:02d}) -> {out_path.name}")
    return out_path


# =============================
# Dual-regression utilities
# =============================
def list_dr_stage2_files(dr_dir: Path) -> list:
    """dr_stage2_subjectNNNNN.nii.gz files, ordered by run index NNNNN."""
    pat = re.compile(r"^dr_stage2_subject(\d{5})\.nii\.gz$")
    indexed = []
    for entry in os.scandir(dr_dir):
        m = pat.match(entry.name) if entry.is_file() else None
        if m:
            indexed.append((int(m.group(1)), Path(entry.path)))
    indexed.sort(key=lambda t: t[0])
    return [p for _, p in indexed]


def maybe_resample_mask_to_img(mask_img, target_img):
    if mask_img.shape == target_img.shape[:3] and np.allclose(mask_img.affine, target_img.affine):
        return mask_img
    if resample_to_img is None:
        raise RuntimeError("Mask and DR image grids differ and nilearn isn't available.")
    return resample_to_img(mask_img, target_img, interpolation="nearest")


def extract_metrics_for_run(dr_img_path: Path, network_masks: dict, component_map: dict,
                            dr_threshold: float = DR_THRESHOLD) -> dict:
    """
    Metrics from one dual-regression stage-2 4-D map.

    network_masks : {mask_key: mask path}   e.g. 'DMN', 'DMN_b', 'DMN_c'
    component_map : {mask_key: IC index}    volume of the stage-2 map to use
    """
    img = load_nifti(dr_img_path)
    data = img.get_fdata()
    if data.ndim != 4:
        raise ValueError(f"Expected 4D DR stage2 image at {dr_img_path}, got {data.shape}")

    results = {}
    for mask_key, ic_idx in component_map.items():
        for stat in ("mean", "var", "std", "largest_cluster"):
            results[f"{mask_key}_{stat}"] = np.nan

        if ic_idx is None or ic_idx < 0 or ic_idx >= data.shape[-1]:
            continue
        mask_path = network_masks.get(mask_key)
        if mask_path is None or not Path(mask_path).exists():
            continue

        vol = data[..., ic_idx]
        mask_img = maybe_resample_mask_to_img(load_nifti(Path(mask_path)), img)
        mask = mask_img.get_fdata().astype(bool)
        if mask.sum() == 0:
            continue

        vals = vol[mask]
        results[f"{mask_key}_mean"] = float(np.mean(vals))
        results[f"{mask_key}_var"] = float(np.var(vals))
        results[f"{mask_key}_std"] = float(np.std(vals))
        results[f"{mask_key}_largest_cluster"] = find_largest_cluster_size((vol >= dr_threshold) & mask)

    return results


# =============================
# Main
# =============================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--network-assignments", required=True,
                        help="subject_network_assignments.csv from step 04")
    parser.add_argument("--out-tsv", default=None,
                        help="Output TSV (default: $CATD_DERIVED_DIR/dualreg_nct_metrics.tsv)")
    parser.add_argument("--make-masks", dest="make_masks", action="store_true", default=True,
                        help="Create masks (default)")
    parser.add_argument("--no-make-masks", dest="make_masks", action="store_false",
                        help="Use existing masks only")
    parser.add_argument("--overwrite-masks", action="store_true", default=False,
                        help="Recreate masks even if they already exist")
    parser.add_argument("--masks-root", default=None, help="default: $CATD_MASKS_DIR")
    parser.add_argument("--melodic-dir", default=None, help="default: $CATD_MELODIC_DIR")
    parser.add_argument("--dualreg-dir", default=None, help="default: $CATD_DUALREG_DIR")
    parser.add_argument("--fmriprep-list-dir", default=None,
                        help="default: $CATD_FMRIPREP_LIST_DIR")
    parser.add_argument("--nct-results-dir", default=None,
                        help="default: $CATD_NCT_RESULTS_DIR")
    parser.add_argument("--thresh-percentile", type=float, default=THRESH_PERCENTILE)
    parser.add_argument("--dr-threshold", type=float, default=DR_THRESHOLD)
    parser.add_argument("--all-voxels", action="store_true", default=False,
                        help="Compute the mask percentile over all voxels, not just positive ones")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Restrict to these subject IDs (e.g. sub-12345)")
    args = parser.parse_args()

    masks_root = Path(args.masks_root) if args.masks_root else catd_config.masks_dir()
    melodic_dir = Path(args.melodic_dir) if args.melodic_dir else catd_config.melodic_dir()
    dualreg_dir = Path(args.dualreg_dir) if args.dualreg_dir else catd_config.dualreg_dir()
    list_dir = Path(args.fmriprep_list_dir) if args.fmriprep_list_dir else catd_config.fmriprep_list_dir()
    nct_dir = Path(args.nct_results_dir) if args.nct_results_dir else catd_config.nct_results_dir()
    out_tsv = Path(args.out_tsv) if args.out_tsv else catd_config.derived_dir() / "dualreg_nct_metrics.tsv"
    melodic_template = str(melodic_dir / "{subject_id}_subject_level_ICA.gica" / "melodic_IC.nii.gz")

    df_assign = pd.read_csv(args.network_assignments)
    if "subject_id" not in df_assign.columns:
        raise ValueError("Network assignments CSV must contain a 'subject_id' column")

    if args.subjects:
        df_assign = df_assign[df_assign["subject_id"].isin(set(args.subjects))].copy()
        if df_assign.empty:
            print("[WARN] No subjects match --subjects filter.", file=sys.stderr)
            sys.exit(0)

    network_cols = [c for c in df_assign.columns
                    if c != "subject_id" and not c.endswith("_is_fallback")]

    print("=" * 70)
    print("Dual-regression metric extraction - primary + secondary + tertiary ICs")
    print("=" * 70)
    print(f"Networks (primary only):  {[n for n in network_cols if n not in RANKED_NETWORKS]}")
    print(f"Networks (primary+_b+_c): {RANKED_NETWORKS}")
    print(f"Subjects: {len(df_assign)}")
    print("=" * 70)

    rows = []
    for _, row in df_assign.iterrows():
        subject_id = row["subject_id"]
        sidnum = catd_config.subject_num(subject_id)
        print(f"\n[{subject_id}]")

        # Step 1: ranked assignments for the three networks of interest
        print("  Resolving ranked IC assignments for DMN/Salience/Frontoparietal...")
        ranked_tiers = resolve_ranked_assignments(subject_id, RANKED_NETWORKS, nct_dir)
        for tier_idx, tier_name in enumerate(["primary", "secondary", "tertiary"]):
            txt = ", ".join(f"{net}=IC{ic}" if ic is not None else f"{net}=None"
                            for net, ic in ranked_tiers[tier_idx].items())
            print(f"  {tier_name.capitalize()}: {txt}")

        # Step 2: component_map / network_masks for all tiers
        #   primary -> 'DMN', secondary -> 'DMN_b', tertiary -> 'DMN_c'
        component_map, network_masks = {}, {}
        for tier_idx, suffix in enumerate(TIER_SUFFIXES):
            for net in RANKED_NETWORKS:
                ic_idx = ranked_tiers[tier_idx].get(net)
                key = f"{net}{suffix}"
                component_map[key] = ic_idx
                if ic_idx is None:
                    continue
                if args.make_masks:
                    try:
                        network_masks[key] = make_mask_for_network(
                            subject_id, net, ic_idx, masks_root, melodic_template,
                            args.thresh_percentile, not args.all_voxels,
                            mask_suffix=suffix, overwrite=args.overwrite_masks)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  ERROR creating mask {key}: {exc}", file=sys.stderr)
                else:
                    p = masks_root / subject_id / f"{subject_id}_{net}{suffix}_mask.nii.gz"
                    if p.exists():
                        network_masks[key] = p

        # Step 3: other networks, primary only (from the step-04 CSV)
        for net in network_cols:
            if net in RANKED_NETWORKS:
                continue
            if pd.isna(row[net]):
                component_map[net] = None
                continue
            ic_idx = int(row[net])
            component_map[net] = ic_idx
            if args.make_masks:
                try:
                    network_masks[net] = make_mask_for_network(
                        subject_id, net, ic_idx, masks_root, melodic_template,
                        args.thresh_percentile, not args.all_voxels,
                        mask_suffix="", overwrite=args.overwrite_masks)
                except Exception as exc:  # noqa: BLE001
                    print(f"  ERROR creating mask {net}: {exc}", file=sys.stderr)
            else:
                p = masks_root / subject_id / f"{subject_id}_{net}_mask.nii.gz"
                if p.exists():
                    network_masks[net] = p

        # Step 4: dual-regression maps and the matching run list
        dr_dir = dualreg_dir / sidnum
        if not dr_dir.exists():
            print(f"  WARN: DR directory not found: {dr_dir}", file=sys.stderr)
            continue
        dr_files = list_dr_stage2_files(dr_dir)
        if not dr_files:
            print(f"  WARN: No DR files found at {dr_dir}", file=sys.stderr)
            continue

        run_paths = catd_config.read_run_list(subject_id, list_dir)
        if not run_paths:
            continue
        if len(run_paths) != len(dr_files):
            print(f"  WARN: run list ({len(run_paths)}) != DR files ({len(dr_files)}); "
                  f"using the smaller count.", file=sys.stderr)

        # Step 5: metrics per run (k-th DR file <-> k-th line of the run list)
        n_runs = min(len(run_paths), len(dr_files))
        print(f"  Extracting metrics from {n_runs} runs...")
        for k in range(n_runs):
            session = catd_config.parse_session(run_paths[k])
            try:
                metrics = extract_metrics_for_run(dr_files[k], network_masks, component_map,
                                                  dr_threshold=args.dr_threshold)
                rows.append({"subject_id": subject_id, "session": session, **metrics})
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR on run {k}: {exc}", file=sys.stderr)
        print(f"  Done: {n_runs} runs")

    if not rows:
        print("\n[WARN] No rows produced. Check inputs.", file=sys.stderr)
        sys.exit(1)

    out_df = pd.DataFrame(rows)

    # Column order: ids, ranked primary, ranked _b/_c, other networks
    stats = ["mean", "var", "std", "largest_cluster"]
    cols = ["subject_id", "session"]
    cols += [f"{net}_{s}" for net in RANKED_NETWORKS for s in stats]
    cols += [f"{net}{suf}_{s}" for suf in ("_b", "_c") for net in RANKED_NETWORKS for s in stats]
    cols += [f"{net}_{s}" for net in network_cols if net not in RANKED_NETWORKS for s in stats]
    out_df = out_df[[c for c in cols if c in out_df.columns]]

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_tsv, sep="\t", index=False)

    print("\n" + "=" * 70)
    print(f"Wrote {len(out_df)} rows to {out_tsv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
