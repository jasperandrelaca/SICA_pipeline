"""
nct_common.py -- single source of truth for the network-assignment logic.

Both 04_assign_networks.py and 06_extract_dr_metrics.py score each subject-level
ICA component against canonical networks using the CBIG Network Correspondence
Toolbox (NCT) output. The atlas-label mappings, candidate gates and the
composite score live here so that the two scripts can never drift apart.

Composite score for one IC vs. one target network
-------------------------------------------------
For every NCT row whose (atlas, label) belongs to the target network:
    weight = (1 - p_value) * dice
    score  = mean(weight over matched rows) + 0.1 * (# distinct atlases matched)

Candidate gate (an IC is a "candidate" for a network iff both hold):
    best_dice    >= CANDIDATE_MIN_DICE[network]   (max Dice over matched rows)
    best_p_value <= CANDIDATE_MAX_P               (min p over matched rows)
"""

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Network label mappings across the NCT atlases used in this study.
# Keys are NCT atlas identifiers; values are the network labels in that atlas
# that we treat as the canonical network.
# --------------------------------------------------------------------------- #
NETWORK_MAPPINGS = {
    "DMN": {
        "TY7": ["Default"],
        "TY17": ["DefaultA", "DefaultB", "DefaultC"],
        "AS400Y17": ["DefaultA", "DefaultB", "DefaultC"],
        "EG17": ["Default"],
    },
    "Salience": {
        "TY7": ["Sal/VenAttn"],
        "TY17": ["Sal/VenAttnA", "Sal/VenAttnB"],
        "AS400Y17": ["Sal/VenAttnA", "Sal/VenAttnB"],
        "EG17": ["Salience"],
    },
    "Frontoparietal": {
        "TY7": ["Control"],
        "TY17": ["ControlA", "ControlB", "ControlC"],
        "AS400Y17": ["ControlA", "ControlB", "ControlC"],
        "EG17": ["FrontPar"],
    },
    "DorsalAttention": {
        "TY7": ["DorsAttn"],
        "TY17": ["DorsAttnA", "DorsAttnB"],
        "AS400Y17": ["DorsAttnA", "DorsAttnB"],
        "EG17": ["DorsAttn"],
    },
    "Visual": {
        "TY7": ["Visual"],
        "TY17": ["VisualA", "VisualB"],
        "AS400Y17": ["VisualA", "VisualB"],
        "EG17": ["LatVis", "MedVis"],
    },
    "Somatomotor": {
        "TY7": ["Somatomotor"],
        "TY17": ["SomatomotorA", "SomatomotorB"],
        "AS400Y17": ["SomatomotorA", "SomatomotorB"],
        "EG17": ["HandSM", "FaceSM", "FootSM", "Premotor"],
    },
}

# Networks of primary interest; these additionally get secondary (_b) and
# tertiary (_c) components in 06_extract_dr_metrics.py.
RANKED_NETWORKS = ["DMN", "Salience", "Frontoparietal"]

# --------------------------------------------------------------------------- #
# Candidate gate
# --------------------------------------------------------------------------- #
# Network-specific Dice floors. DMN / Frontoparietal use 0.3; Salience keeps the
# permissive 0.2 because its winning-component Dice distribution overlaps the
# secondary-candidate range, so a higher floor would discard genuine matches.
# To re-examine these values run scripts/diagnostics/diagnose_min_dice_floor.py.
CANDIDATE_MIN_DICE = {
    "DMN": 0.3,
    "Salience": 0.2,
    "Frontoparietal": 0.3,
    "DorsalAttention": 0.2,
    "Visual": 0.2,
    "Somatomotor": 0.2,
}
CANDIDATE_MAX_P = 0.05  # nominal significance, uniform across networks

# Fallback gate: used ONLY when a subject has zero candidates under the primary
# gate for a network (dice requirement relaxed; significance still required).
FALLBACK_MIN_DICE = 0.0
FALLBACK_MAX_P = 0.05


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def calculate_ic_score(ic_data: pd.DataFrame, target_network: str) -> dict:
    """
    Composite match score of one IC (one NCT ``network_correspondence.csv``
    table) against ``target_network``. See module docstring for the formula.

    Returns a dict with: score, dice_mean, p_value_mean, n_atlases, best_dice,
    best_p_value, details (list of matched rows).
    """
    matches = []
    for atlas_name, labels in NETWORK_MAPPINGS[target_network].items():
        atlas_rows = ic_data[ic_data["group"] == atlas_name]
        for label in labels:
            for _, row in atlas_rows[atlas_rows["name"] == label].iterrows():
                dice, p_value = row["dice"], row["p_value"]
                matches.append({
                    "atlas": atlas_name,
                    "network": label,
                    "dice": dice,
                    "p_value": p_value,
                    "weight": (1 - p_value) * dice,
                })

    if not matches:
        return {
            "score": 0.0, "dice_mean": 0.0, "p_value_mean": 1.0, "n_atlases": 0,
            "best_dice": 0.0, "best_p_value": 1.0, "details": [],
        }

    dice_values = [m["dice"] for m in matches]
    p_values = [m["p_value"] for m in matches]
    weights = [m["weight"] for m in matches]
    n_atlases = len({m["atlas"] for m in matches})

    return {
        "score": float(np.mean(weights) + 0.1 * n_atlases),
        "dice_mean": float(np.mean(dice_values)),
        "p_value_mean": float(np.mean(p_values)),
        "n_atlases": n_atlases,
        "best_dice": float(np.max(dice_values)),
        "best_p_value": float(np.min(p_values)),
        "details": matches,
    }


def passes_gate(score_dict: dict, network: str,
                min_dice=CANDIDATE_MIN_DICE, max_p=CANDIDATE_MAX_P) -> bool:
    """True if ``score_dict`` clears the (network-specific) Dice / p gate."""
    net_min_dice = min_dice[network] if isinstance(min_dice, dict) else min_dice
    net_max_p = max_p[network] if isinstance(max_p, dict) else max_p
    return (score_dict["best_dice"] >= net_min_dice
            and score_dict["best_p_value"] <= net_max_p)


# --------------------------------------------------------------------------- #
# NCT output loading
# --------------------------------------------------------------------------- #
def load_subject_ic_tables(subject_dir) -> dict:
    """
    Load every ``{subject}_ICxx/network_correspondence.csv`` under one subject's
    NCT results directory.  Returns {ic_num (int): DataFrame}. IC numbers are
    0-based and match the volume index in MELODIC's melodic_IC.nii.gz and in
    the dual-regression stage-2 maps.
    """
    subject_dir = Path(subject_dir)
    tables = {}
    if not subject_dir.exists():
        return tables

    ic_dirs = sorted(
        (d for d in subject_dir.iterdir() if d.is_dir() and "_IC" in d.name),
        key=lambda d: int(d.name.split("_IC")[1]),
    )
    for ic_dir in ic_dirs:
        ic_num = int(ic_dir.name.split("_IC")[1])
        csv_path = ic_dir / "network_correspondence.csv"
        if not csv_path.exists():
            continue
        try:
            tables[ic_num] = pd.read_csv(csv_path, index_col=0)
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: could not read {csv_path}: {exc}")
    return tables
