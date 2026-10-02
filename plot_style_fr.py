#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot: Failure Rate by Rhetorical Style × Model Family
Horizontal grouped bar chart.

Usage:
    python plot_style_fr_h.py --input results_table.csv --output style_fr.pdf
"""

import csv
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

STYLES       = ["NEUTRAL", "AUTHORITY", "PSEUDO-LOGIC", "OMISSION"]
STYLE_KEYS   = ["FR_NEUTRAL", "FR_AUTHORITY", "FR_PSEUDO-LOGIC", "FR_OMISSION"]
STYLE_LABELS = ["Neutral", "Authority", "Pseudo-Logic", "Omission"]

# GPT at top → reversed for horizontal (bottom = first plotted)
FAMILY_ORDER = ["GPT", "Claude", "Gemini", "DeepSeek", "Qwen"]

STYLE_COLORS = {
    "NEUTRAL":      "#6baed6",
    "AUTHORITY":    "#fd8d3c",
    "PSEUDO-LOGIC": "#d62728",
    "OMISSION":     "#74c476",
}

FAMILY_MARKER_COLORS = {
    "GPT":      "#1f77b4",
    "Claude":   "#ff7f0e",
    "Gemini":   "#2ca02c",
    "DeepSeek": "#9467bd",
    "Qwen":     "#8c564b",
}

REASONING_MODELS = {"DeepSeek-R1", "Qwen3-235B-Thinking"}


def load_data(path):
    rows = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["display_name"]] = r
    return rows


def family_means(rows):
    family_data = defaultdict(list)
    for name, r in rows.items():
        family_data[r["family"]].append(r)
    means = {}
    for fam, frows in family_data.items():
        means[fam] = {
            sk: np.mean([float(r[sk]) for r in frows])
            for sk in STYLE_KEYS
        }
    return means


def model_points(rows):
    family_data = defaultdict(list)
    for name, r in rows.items():
        family_data[r["family"]].append(r)
    return family_data


def plot(rows, output_path):
    matplotlib.rcParams.update({
        "font.family":       "Arial",
        "font.size":         10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "axes.grid.axis":    "x",
        "grid.alpha":        0.35,
        "grid.linestyle":    "--",
    })

    means   = family_means(rows)
    fam_pts = model_points(rows)

    n_families = len(FAMILY_ORDER)
    n_styles   = len(STYLES)
    bar_h      = 0.15
    group_gap  = 0.14
    group_h    = n_styles * bar_h + group_gap

    fig, ax = plt.subplots(figsize=(7, 5.5))

    y_ticks  = []
    y_labels = []

    for fi, fam in enumerate(FAMILY_ORDER):
        group_center = fi * group_h
        y_ticks.append(group_center)
        y_labels.append(fam)

        for si, (sk, style, color) in enumerate(
            zip(STYLE_KEYS, STYLES,
                [STYLE_COLORS[s] for s in STYLES])
        ):
            y = group_center + (si - (n_styles - 1) / 2) * bar_h
            mean_val = means.get(fam, {}).get(sk, 0)

            # Bar
            ax.barh(
                y, mean_val,
                height=bar_h * 0.85,
                color=color,
                alpha=0.85,
                zorder=2,
                label=STYLE_LABELS[si] if fi == 0 else "_nolegend_",
            )

            # Value label: only show if bar is wide enough to avoid overlap
            if mean_val >= 3.0:
                ax.text(
                    mean_val + 0.5, y,
                    f"{mean_val:.1f}",
                    ha="left", va="center",
                    fontsize=10, color="#333333",
                    zorder=4,
                )

            # Individual model dots — jitter y slightly to avoid overlap
            model_vals = [(float(r[sk]), r["display_name"])
                          for r in fam_pts.get(fam, [])]
            model_vals.sort(key=lambda x: x[0])
            n = len(model_vals)
            for idx, (val, name) in enumerate(model_vals):
                jitter = (idx - (n - 1) / 2) * 0.022
                marker = "*" if name in REASONING_MODELS else "o"
                ax.scatter(
                    val, y + jitter,
                    s=35 if marker == "o" else 65,
                    marker=marker,
                    color=FAMILY_MARKER_COLORS[fam],
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=5,
                    alpha=0.9,
                )

    # Axes
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=11, fontweight="bold")
    ax.set_xlabel("Failure Rate (%)", fontsize=11)
    ax.set_xlim(0, 54)
    ax.set_ylim(-group_h * 0.6, (n_families - 1) * group_h + group_h * 0.6)
    ax.invert_yaxis()  # GPT on top

    # Average Pseudo-Logic vertical line
    all_pl = [float(r["FR_PSEUDO-LOGIC"]) for r in rows.values()]
    avg_pl = np.mean(all_pl)
    ax.axvline(
        avg_pl,
        color=STYLE_COLORS["PSEUDO-LOGIC"],
        linestyle=":", linewidth=1.2, alpha=0.65,
    )
    ax.text(
        avg_pl + 0.4,
        ax.get_ylim()[1] - 0.05,
        f"avg. P-L\n{avg_pl:.1f}%",
        ha="left", va="top",
        fontsize=10, color=STYLE_COLORS["PSEUDO-LOGIC"],
        alpha=0.85,
    )

    # Style legend
    style_handles = [
        mpatches.Patch(color=STYLE_COLORS[s], alpha=0.85, label=l)
        for s, l in zip(STYLES, STYLE_LABELS)
    ]
    reasoning_handle = plt.scatter(
        [], [], marker="*", s=80, color="gray", label="Reasoning model (†)"
    )
    normal_handle = plt.scatter(
        [], [], marker="o", s=35, color="gray", label="Standard model"
    )

    leg1 = ax.legend(
        handles=style_handles,
        title="Rhetorical Style",
        title_fontsize=10,
        fontsize=10,
        loc="lower right",
        framealpha=0.9,
        edgecolor="#cccccc",
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=[normal_handle, reasoning_handle],
        fontsize=10,
        loc="upper right",
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, default="results_table.csv")
    parser.add_argument("--output", type=str, default="style_fr.pdf")
    args = parser.parse_args()
    rows = load_data(Path(args.input))
    plot(rows, Path(args.output))


if __name__ == "__main__":
    main()