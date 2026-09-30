#!/usr/bin/env python3
"""
diagnose_min_dice_floor.py -- how strict should the per-network Dice floor be?

Step 04 (and 06) accept an IC as a *candidate* for a network only if its best Dice
against that network's atlas labels is at least a network-specific floor
(nct_common.CANDIDATE_MIN_DICE) and its best p-value is <= CANDIDATE_MAX_P. For each
participant and network, the candidate with the highest composite score is the
"winner"; every other candidate that clears the gate is a "secondary" candidate and is
what makes the fragmentation count larger than 1.

This diagnostic re-derives all candidates from the NCT results, separates winners
from secondary candidates, and shows for each network:

  * the Dice distribution of the secondary candidates and of the winners;
  * how many secondary candidates a stricter floor would remove; and
  * how many WINNERS the same floor would also remove.

It answers: are the extra ICs strong but redundant matches (a genuinely split
network) or weak matches that barely clear the floor (noise that a stricter floor
would correctly cut)?  A floor that removes many secondary candidates and no winners
is well justified; one that starts removing winners is cutting real signal. This is
how the network-specific floors in nct_common.py were chosen (DMN and Frontoparietal
0.3; Salience kept at 0.2 because its winners overlap the secondary range).

The gate and the scoring are imported from nct_common.py, so the candidate sets are
exactly those used by steps 04 and 06.

Usage
  python scripts/diagnostics/diagnose_min_dice_floor.py
  python scripts/diagnostics/diagnose_min_dice_floor.py --nct-dir /path/to/nct_results \\
      --floors 0.2 0.3 0.4 --out-prefix floor_diag

Inputs : $CATD_NCT_RESULTS_DIR (output of step 03): sub-*/sub-*_ICxx/network_correspondence.csv
Outputs: a text report on stdout; with --out-prefix, <prefix>_winners.csv and
         <prefix>_secondary.csv (one row per subject/network/IC).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import catd_config as cfg  # noqa: E402
from nct_common import (  # noqa: E402
    CANDIDATE_MAX_P,
    CANDIDATE_MIN_DICE,
    NETWORK_MAPPINGS,
    calculate_ic_score,
    load_subject_ic_tables,
    passes_gate,
)

DEFAULT_FLOORS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
SCORE_KEYS = ("score", "best_dice", "best_p_value", "n_atlases")


def collect_candidates(nct_dir):
    """Winner and secondary records for every subject x network, under the current gate."""
    subject_dirs = sorted(d for d in Path(nct_dir).glob("sub-*") if d.is_dir())
    winners, secondary = [], []
    for sub_dir in subject_dirs:
        tables = load_subject_ic_tables(sub_dir)
        if not tables:
            continue
        for net in NETWORK_MAPPINGS:
            candidates = {}
            for ic_num, table in tables.items():          # ascending IC order
                sd = calculate_ic_score(table, net)
                if passes_gate(sd, net, CANDIDATE_MIN_DICE, CANDIDATE_MAX_P):
                    candidates[ic_num] = sd
            if not candidates:
                continue
            # highest score wins; ties go to the lowest IC number (as in step 04)
            ranked = sorted(candidates.items(), key=lambda kv: -kv[1]["score"])
            for rank, (ic_num, sd) in enumerate(ranked):
                rec = {"subject_id": sub_dir.name, "network": net, "ic_num": ic_num,
                       **{k: sd[k] for k in SCORE_KEYS}}
                (winners if rank == 0 else secondary).append(rec)
    return pd.DataFrame(winners), pd.DataFrame(secondary), len(subject_dirs)


def report(winner_df, secondary_df, floors):
    print(f"Total winners: {len(winner_df)}")
    print(f"Total secondary (fragmenting) candidates: {len(secondary_df)}\n")
    if secondary_df.empty:
        print("No fragmentation at current thresholds -- nothing to diagnose.")
        return

    for net in NETWORK_MAPPINGS:
        sec = secondary_df[secondary_df["network"] == net]
        win = winner_df[winner_df["network"] == net]
        if sec.empty:
            print(f"=== {net}: no secondary candidates (no fragmentation) ===\n")
            continue

        print(f"=== {net}: {len(sec)} secondary candidates across "
              f"{sec['subject_id'].nunique()} subjects ===")
        print("  Secondary candidate dice_best distribution:")
        print(f"    min:    {sec['best_dice'].min():.4f}")
        print(f"    25th:   {sec['best_dice'].quantile(0.25):.4f}")
        print(f"    median: {sec['best_dice'].median():.4f}")
        print(f"    75th:   {sec['best_dice'].quantile(0.75):.4f}")
        print(f"    max:    {sec['best_dice'].max():.4f}")

        print("\n  Winner dice_best distribution (for comparison):")
        print(f"    min:    {win['best_dice'].min():.4f}")
        print(f"    median: {win['best_dice'].median():.4f}")
        print(f"    max:    {win['best_dice'].max():.4f}")

        print("\n  If we raise min_dice further, how many secondary candidates")
        print("  (fragmentation sources) get removed at each floor:")
        for floor in floors:
            remaining = int((sec["best_dice"] >= floor).sum())
            removed = len(sec) - remaining
            print(f"    min_dice={floor:<5}: {remaining:3d} secondary candidates "
                  f"remain ({removed} removed, {100 * removed / len(sec):.1f}%)")

        print("\n  And critically -- would raising the floor ALSO cut any winners?")
        for floor in floors:
            print(f"    min_dice={floor:<5}: {int((win['best_dice'] < floor).sum())} "
                  "winner(s) would be lost")
        print()

    print("=" * 60)
    print("Interpretation:")
    print("- If secondary candidates cluster just above the current floor")
    print("  (e.g. dice 0.2-0.3) while winners sit much higher (e.g. 0.35+),")
    print("  raising the floor toward the secondary ceiling is well-justified")
    print("  and should cost zero winners.")
    print("- If secondary dice values are spread widely or overlap heavily")
    print("  with winner dice values, some 'fragmentation' may be real")
    print("  (genuinely split network representation) rather than noise --")
    print("  worth inspecting those specific subjects directly.")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nct-dir", type=Path, default=None,
                    help="step-03 output directory (default: $CATD_NCT_RESULTS_DIR)")
    ap.add_argument("--floors", type=float, nargs="+", default=DEFAULT_FLOORS,
                    help="Dice floors to evaluate (default: 0.20 to 0.70 in steps of 0.05)")
    ap.add_argument("--out-prefix", type=Path, default=None,
                    help="also write <prefix>_winners.csv and <prefix>_secondary.csv")
    args = ap.parse_args()

    nct_dir = args.nct_dir or cfg.nct_results_dir()
    if not Path(nct_dir).is_dir():
        sys.exit(f"ERROR: NCT results directory not found: {nct_dir}")
    winner_df, secondary_df, n_subjects = collect_candidates(nct_dir)

    print(f"Found {n_subjects} subjects\n")
    print(f"Using current gate: min_dice={CANDIDATE_MIN_DICE}, max_p_value={CANDIDATE_MAX_P}\n")
    report(winner_df, secondary_df, args.floors)

    if args.out_prefix:
        winner_df.to_csv(f"{args.out_prefix}_winners.csv", index=False)
        secondary_df.to_csv(f"{args.out_prefix}_secondary.csv", index=False)
        print(f"\nRecords written to {args.out_prefix}_winners.csv / _secondary.csv")


if __name__ == "__main__":
    main()
