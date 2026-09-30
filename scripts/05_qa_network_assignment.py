#!/usr/bin/env python3
"""
05_qa_network_assignment.py
===========================
Quality-assessment plots for the ICA-component -> canonical-network assignments
produced by 04_assign_networks.py.

Figures written to --plots-dir
------------------------------
coverage_by_network.png     % of participants with an identified component per network
dice_distributions.png      histogram of the winning component's best Dice, per network
pvalue_distributions.png    histogram of the winning component's best p-value, per network
atlas_consistency.png       number of atlases that agree with the assignment, per network
score_vs_dice.png           composite score vs. best Dice for all winning components
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import catd_config

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)


def _save(fig_path):
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")


def plot_assignment_coverage(summary_df, output_path):
    """Percentage of subjects with an identified component, per network."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Network columns only: exclude the id column and the '<network>_is_fallback' flags.
    networks = [c for c in summary_df.columns
                if c != "subject_id" and not c.endswith("_is_fallback")]
    coverage = [summary_df[n].notna().mean() * 100 for n in networks]

    bars = ax.barh(networks, coverage, color="steelblue", alpha=0.7)
    for i, (bar, pct) in enumerate(zip(bars, coverage)):
        ax.text(pct + 1, i, f"{pct:.1f}%", va="center")

    ax.set_xlabel("% of Subjects with Identified Network", fontsize=12)
    ax.set_title("Network Assignment Coverage Across Subjects", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 110)
    _save(output_path)


def _grid_of_histograms(details_df, column, color, xlabel, title, output_path,
                        vline=None, vline_label=None, mean_line=False, pct_sig_in_title=False):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    networks = details_df["network"].unique()

    for idx, network in enumerate(networks[:len(axes)]):
        ax = axes[idx]
        data = details_df[details_df["network"] == network]
        ax.hist(data[column], bins=20, color=color, alpha=0.7, edgecolor="black")

        if mean_line:
            mean_val = data[column].mean()
            ax.axvline(mean_val, color="red", linestyle="--", linewidth=2,
                       label=f"Mean: {mean_val:.3f}")
        if vline is not None:
            ax.axvline(vline, color="red", linestyle="--", linewidth=2, label=vline_label)

        net_title = network
        if pct_sig_in_title:
            net_title = f"{network}\n({(data[column] < 0.05).mean() * 100:.1f}% p<0.05)"

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Number of Subjects", fontsize=10)
        ax.set_title(net_title, fontsize=12, fontweight="bold")
        ax.legend()

    for idx in range(len(networks), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    _save(output_path)


def plot_dice_distributions(details_df, output_path):
    _grid_of_histograms(details_df, "dice_best", "steelblue", "Best Dice Coefficient",
                        "Distribution of Dice Coefficients by Network", output_path,
                        mean_line=True)


def plot_pvalue_distributions(details_df, output_path):
    _grid_of_histograms(details_df, "p_value_best", "coral", "Best P-value",
                        "Distribution of P-values by Network", output_path,
                        vline=0.05, vline_label="p=0.05", pct_sig_in_title=True)


def plot_atlas_consistency(details_df, output_path):
    """Violin plot: number of atlases (max 4) agreeing with each assignment."""
    fig, ax = plt.subplots(figsize=(10, 6))
    networks = details_df["network"].unique()
    data = [details_df.loc[details_df["network"] == n, "n_atlases"].values for n in networks]

    parts = ax.violinplot(data, positions=range(len(networks)), showmeans=True, showmedians=True)
    for body in parts["bodies"]:
        body.set_facecolor("steelblue")
        body.set_alpha(0.7)

    ax.set_xticks(range(len(networks)))
    ax.set_xticklabels(networks, rotation=45, ha="right")
    ax.set_ylabel("Number of Atlases with Match", fontsize=12)
    ax.set_title("Multi-Atlas Consistency per Network", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 5)
    _save(output_path)


def plot_score_vs_dice(details_df, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    networks = details_df["network"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(networks)))

    for network, color in zip(networks, colors):
        data = details_df[details_df["network"] == network]
        ax.scatter(data["dice_best"], data["score"], label=network, alpha=0.6, s=50, color=color)

    ax.set_xlabel("Best Dice Coefficient", fontsize=12)
    ax.set_ylabel("Composite Score", fontsize=12)
    ax.set_title("Composite Score vs. Dice Coefficient", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    _save(output_path)


def create_quality_report(summary_path, details_path, output_dir):
    print("\n" + "=" * 60 + "\nCreating quality assessment plots\n" + "=" * 60)
    summary_df = pd.read_csv(summary_path)
    details_df = pd.read_csv(details_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_assignment_coverage(summary_df, output_dir / "coverage_by_network.png")
    plot_dice_distributions(details_df, output_dir / "dice_distributions.png")
    plot_pvalue_distributions(details_df, output_dir / "pvalue_distributions.png")
    plot_atlas_consistency(details_df, output_dir / "atlas_consistency.png")
    plot_score_vs_dice(details_df, output_dir / "score_vs_dice.png")

    print("\n" + "=" * 60 + f"\nAll plots saved to: {output_dir}\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assignments-dir", default=None,
                        help="Folder containing the step-04 CSVs (default: $CATD_ASSIGNMENTS_DIR)")
    parser.add_argument("--plots-dir", default=None,
                        help="Where to write the PNGs (default: <assignments-dir>/quality_plots)")
    args = parser.parse_args()

    base = Path(args.assignments_dir) if args.assignments_dir else catd_config.assignments_dir()
    plots_dir = Path(args.plots_dir) if args.plots_dir else base / "quality_plots"

    create_quality_report(base / "subject_network_assignments.csv",
                          base / "network_assignment_details.csv",
                          plots_dir)


if __name__ == "__main__":
    main()
