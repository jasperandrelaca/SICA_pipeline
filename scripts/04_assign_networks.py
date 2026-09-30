#!/usr/bin/env python3
"""
04_assign_networks.py
=====================
Identify which participant-specific ICA component best represents each
canonical network (DMN, Salience, Frontoparietal, Dorsal Attention, Visual,
Somatomotor), using the NCT results from 03_nct_network_correspondence.py.

For every participant and network
---------------------------------
1. Score each IC against the network with the composite score defined in
   ``nct_common.calculate_ic_score`` (Dice x significance, plus a bonus for
   agreement across atlases).
2. An IC is a *candidate* if it clears the network-specific gate
   (``nct_common.CANDIDATE_MIN_DICE`` / ``CANDIDATE_MAX_P``).
3. The candidate with the highest composite score is the *winner*.
4. If NO IC clears the gate, fall back to the best IC under a permissive gate
   (Dice >= 0, p <= 0.05) and flag it (``*_is_fallback``), so that these
   assignments stay visible downstream.
5. The number of candidates is also recorded: more than one candidate suggests
   the network is split across several components ("fragmented").

Outputs (in --output-dir)
-------------------------
subject_network_assignments.csv   subjects x networks (winning IC number, 0-based)
                                  + '<network>_is_fallback' columns
network_assignment_details.csv    score / Dice / p-value / n_atlases per winner
network_fragmentation_counts.csv  n_significant_<network> (candidate counts) and
                                  significant_<network>_ICs (which ICs)

Important: for DMN, Salience and Frontoparietal, the masks used for dual-regression
metrics (06_extract_dr_metrics.py) are re-derived there with a global (Hungarian)
matching that also yields secondary/tertiary components, so those three columns of
subject_network_assignments.csv are informational; see docs/PIPELINE.md.
"""

import argparse
import os
from pathlib import Path

import pandas as pd

import catd_config
from nct_common import (
    NETWORK_MAPPINGS,
    CANDIDATE_MIN_DICE,
    CANDIDATE_MAX_P,
    FALLBACK_MIN_DICE,
    FALLBACK_MAX_P,
    calculate_ic_score,
    passes_gate,
    load_subject_ic_tables,
)


def load_all_subject_data(nct_results_dir):
    """
    Load NCT tables for every subject and IC.

    Returns {subject_id: {ic_num: DataFrame}}.
    """
    results_dir = Path(nct_results_dir)
    subject_dirs = sorted(d for d in results_dir.iterdir()
                          if d.is_dir() and d.name.startswith("sub-"))
    print(f"Found {len(subject_dirs)} subjects\nLoading data...")

    all_data = {}
    for subject_dir in subject_dirs:
        tables = load_subject_ic_tables(subject_dir)
        all_data[subject_dir.name] = tables
        if tables:
            print(f"  {subject_dir.name}: {len(tables)} ICs loaded")
        else:
            print(f"  {subject_dir.name}: No ICs found")
    return all_data


def find_candidate_ics_for_network(subject_data, target_network,
                                   min_dice=CANDIDATE_MIN_DICE, max_p_value=CANDIDATE_MAX_P):
    """
    All ICs of one subject that clear the candidate gate for ``target_network``.

    ``min_dice`` / ``max_p_value`` may be floats or {network: float} dicts.
    Returns {ic_num: score_dict}.
    """
    candidates = {}
    for ic_num, ic_data in subject_data.items():
        score_dict = calculate_ic_score(ic_data, target_network)
        if passes_gate(score_dict, target_network, min_dice, max_p_value):
            candidates[ic_num] = score_dict
    return candidates


def find_best_ic_for_network(subject_data, target_network,
                             min_dice=CANDIDATE_MIN_DICE, max_p_value=CANDIDATE_MAX_P,
                             fallback_min_dice=FALLBACK_MIN_DICE,
                             fallback_max_p_value=FALLBACK_MAX_P):
    """
    Winning IC for a network in one subject (highest composite score among
    candidates), falling back to the permissive gate if there are none.

    Returns (ic_num, score_dict) with score_dict['is_fallback'] set, or
    (None, None) if nothing qualifies even under the fallback gate.
    """
    candidates = find_candidate_ics_for_network(
        subject_data, target_network, min_dice, max_p_value)
    is_fallback = False

    if not candidates:
        candidates = find_candidate_ics_for_network(
            subject_data, target_network, fallback_min_dice, fallback_max_p_value)
        is_fallback = True
        if not candidates:
            return None, None

    best_ic, score_dict = max(candidates.items(), key=lambda kv: kv[1]["score"])
    score_dict = dict(score_dict)
    score_dict["is_fallback"] = is_fallback
    return best_ic, score_dict


def create_summary_table(all_data, networks_of_interest=None,
                         min_dice=CANDIDATE_MIN_DICE, max_p_value=CANDIDATE_MAX_P,
                         fallback_min_dice=FALLBACK_MIN_DICE,
                         fallback_max_p_value=FALLBACK_MAX_P,
                         include_details=False):
    """
    Build the assignment, details and fragmentation tables.

    Returns (summary_df, details_df, fragmentation_df); details_df is None
    unless ``include_details``. Fragmentation counts use the PRIMARY gate only.
    """
    if networks_of_interest is None:
        networks_of_interest = list(NETWORK_MAPPINGS.keys())

    summary_data, details_data, fragmentation_data = [], [], []
    print(f"\nAnalyzing {len(networks_of_interest)} networks across {len(all_data)} subjects...")

    n_fallback_total = 0
    for subject_id, subject_data in sorted(all_data.items()):
        row = {"subject_id": subject_id}
        frag_row = {"subject_id": subject_id}

        for network in networks_of_interest:
            candidates = find_candidate_ics_for_network(
                subject_data, network, min_dice, max_p_value)
            frag_row[f"n_significant_{network}"] = len(candidates)
            frag_row[f"significant_{network}_ICs"] = ";".join(
                f"IC{ic}(score={s['score']:.4f})" for ic, s in sorted(candidates.items()))

            best_ic, score_dict = find_best_ic_for_network(
                subject_data, network, min_dice, max_p_value,
                fallback_min_dice, fallback_max_p_value)

            row[network] = best_ic
            row[f"{network}_is_fallback"] = (
                score_dict["is_fallback"] if score_dict is not None else None)

            if score_dict is not None and score_dict["is_fallback"]:
                n_fallback_total += 1
                print(f"  {subject_id} / {network}: FALLBACK assignment "
                      f"(IC{best_ic}, dice={score_dict['best_dice']:.4f}, "
                      f"p={score_dict['best_p_value']:.4f}) -- no candidate cleared the primary gate")

            if best_ic is not None and include_details:
                details_data.append({
                    "subject_id": subject_id,
                    "network": network,
                    "ic_num": best_ic,
                    "score": score_dict["score"],
                    "dice_mean": score_dict["dice_mean"],
                    "dice_best": score_dict["best_dice"],
                    "p_value_mean": score_dict["p_value_mean"],
                    "p_value_best": score_dict["best_p_value"],
                    "n_atlases": score_dict["n_atlases"],
                    "is_fallback": score_dict["is_fallback"],
                })

        summary_data.append(row)
        fragmentation_data.append(frag_row)

        n_identified = sum(row.get(n) is not None for n in networks_of_interest)
        fragmented = [n for n in networks_of_interest
                      if frag_row.get(f"n_significant_{n}", 0) > 1]
        note = f" | FRAGMENTED: {', '.join(fragmented)}" if fragmented else ""
        print(f"  {subject_id}: {n_identified} networks identified{note}")

    summary_df = pd.DataFrame(summary_data)
    details_df = pd.DataFrame(details_data) if include_details else None
    fragmentation_df = pd.DataFrame(fragmentation_data)

    if n_fallback_total:
        print(f"\nNOTE: {n_fallback_total} subject-network assignment(s) used the FALLBACK gate. "
              f"See '<network>_is_fallback' columns / 'is_fallback' in the details table.")
    return summary_df, details_df, fragmentation_df


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nct-results-dir", default=None,
                        help="NCT results from step 03 (default: $CATD_NCT_RESULTS_DIR)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write the CSVs (default: $CATD_ASSIGNMENTS_DIR)")
    args = parser.parse_args()

    nct_results_dir = args.nct_results_dir or str(catd_config.nct_results_dir())
    output_dir = args.output_dir or str(catd_config.assignments_dir())

    networks_of_interest = ["DMN", "Salience", "Frontoparietal",
                            "DorsalAttention", "Visual", "Somatomotor"]

    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60 + "\nICA component -> canonical network assignment\n" + "=" * 60)

    all_data = load_all_subject_data(nct_results_dir)
    if not all_data:
        print("ERROR: No data found!")
        return

    summary_df, details_df, fragmentation_df = create_summary_table(
        all_data, networks_of_interest=networks_of_interest, include_details=True)

    summary_path = os.path.join(output_dir, "subject_network_assignments.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary table saved: {summary_path}")
    print(f"  Shape: {summary_df.shape[0]} subjects x {len(networks_of_interest)} networks")

    details_path = os.path.join(output_dir, "network_assignment_details.csv")
    details_df.to_csv(details_path, index=False)
    print(f"Detailed statistics saved: {details_path}")

    frag_path = os.path.join(output_dir, "network_fragmentation_counts.csv")
    fragmentation_df.to_csv(frag_path, index=False)
    print(f"Fragmentation counts saved: {frag_path}")

    print(f"\n{'=' * 60}\nSummary statistics\n{'=' * 60}")
    for network in networks_of_interest:
        n_id = summary_df[network].notna().sum()
        print(f"  {network:20s}: {n_id:3d}/{len(summary_df):3d} subjects "
              f"({100 * n_id / len(summary_df):.1f}%)")

    print(f"\n{'=' * 60}\nFallback assignments (primary gate cleared by nobody)\n{'=' * 60}")
    for network in networks_of_interest:
        fb = summary_df[f"{network}_is_fallback"] == True  # noqa: E712 (handles None)
        if fb.any():
            ids = ", ".join(summary_df.loc[fb, "subject_id"])
            print(f"  {network:20s}: {fb.sum()} fallback assignment(s) -- {ids}")
        else:
            print(f"  {network:20s}: 0 fallback assignments")

    print(f"\n{'=' * 60}\nFragmentation (candidate ICs per subject)\n{'=' * 60}")
    for network in networks_of_interest:
        col = f"n_significant_{network}"
        print(f"  {network:20s}: {(fragmentation_df[col] == 1).sum():3d} clean (1), "
              f"{(fragmentation_df[col] > 1).sum():3d} fragmented (>1), "
              f"{(fragmentation_df[col] == 0).sum():3d} none (0)")

    print(f"\n{'=' * 60}\nExample assignments (first 5 subjects)\n{'=' * 60}")
    print(summary_df.head(5).to_string(index=False))
    print(f"\n{'=' * 60}\nDone!\n{'=' * 60}")


if __name__ == "__main__":
    main()
