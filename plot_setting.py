#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot: Failure Rate by World Setting × Model Family (grouped bar)
x-axis: 4 world settings, each group has 5 bars (one per family)

Usage:
    python plot_setting_bar.py --input setting_fr.csv --output setting_bar.pdf
"""

import csv
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

FAMILY_MAP = {
    "GPT-5.4":             "GPT",
    "GPT-5":               "GPT",
    "GPT-5-mini":          "GPT",
    "GPT-4.1":             "GPT",
    "Claude Opus 4.6":     "Claude",
    "Claude Sonnet 4.6":   "Claude",
    "Claude Sonnet 4.5":   "Claude",
    "Claude Haiku 4.5":    "Claude",
    "Gemini 2.5 Flash":    "Gemini",
    "Gemini 3 Flash":      "Gemini",
    "Gemini 3.5 Flash":    "Gemini",
    "DeepSeek-V4-Flash":   "DeepSeek",
    "DeepSeek-V4-Pro":     "DeepSeek",
    "DeepSeek-V3.2":       "DeepSeek",
    "DeepSeek-R1":         "DeepSeek",
    "Qwen3.7-Max":         "Qwen",
    "Qwen3.6-Flash":       "Qwen",
    "Qwen3-Max":           "Qwen",
    "Qwen3-235B-Thinking": "Qwen",
    "Qwen3-235B-Instruct": "Qwen",
}

FAMILY_ORDER = ["GPT", "Claude", "Gemini", "DeepSeek", "Qwen"]

FAMILY_COLORS = {
    "GPT":      "#4e79a7",
    "Claude":   "#f28e2b",
    "Gemini":   "#59a14f",
    "DeepSeek": "#b07aa1",
    "Qwen":     "#e15759",
}

SETTINGS   = ["1920s_Urban", "2020s_Urban", "Ancient_China", "Wilderness"]
SET_LABELS = ["1920s Urban", "2020s Urban", "Ancient China", "Wilderness"]


def load(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["display_name"]] = r
    return rows


def family_means(rows):
    """Returns {family: {setting: mean_fr}}"""
    fam_data = defaultdict(list)
    for name, r in rows.items():
        fam = FAMILY_MAP.get(name)
        if fam:
            fam_data[fam].append(r)

    means = {}
    for fam, frows in fam_data.items():
        means[fam] = {
            s: np.mean([float(r[s]) for r in frows])
            for s in SETTINGS
        }
    return means


def family_individual(rows):
    """Returns {family: {setting: [individual model FRs]}}"""
    fam_data = defaultdict(list)
    for name, r in rows.items():
        fam = FAMILY_MAP.get(name)
        if fam:
            fam_data[fam].append(r)

    pts = {}
    for fam, frows in fam_data.items():
        pts[fam] = {
            s: [float(r[s]) for r in frows]
            for s in SETTINGS
        }
    return pts


def plot(rows, output_path):
    matplotlib.rcParams.update({
        "font.family":       "Arial",
        "font.size":         12,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "axes.grid.axis":    "y",
        "grid.alpha":        0.35,
        "grid.linestyle":    "--",
    })

    means = family_means(rows)
    pts   = family_individual(rows)

    n_settings = len(SETTINGS)
    n_families = len(FAMILY_ORDER)
    bar_w      = 0.14
    group_gap  = 0.15
    group_w    = n_families * bar_w + group_gap

    fig, ax = plt.subplots(figsize=(11, 4))

    x_ticks  = []
    x_labels = []

    for si, (s, label) in enumerate(zip(SETTINGS, SET_LABELS)):
        group_center = si * group_w
        x_ticks.append(group_center)
        x_labels.append(label)

        for fi, fam in enumerate(FAMILY_ORDER):
            x = group_center + (fi - (n_families - 1) / 2) * bar_w
            mean_val = means.get(fam, {}).get(s, 0)
            color    = FAMILY_COLORS[fam]

            # Bar
            ax.bar(
                x, mean_val,
                width=bar_w * 0.88,
                color=color,
                alpha=0.82,
                zorder=2,
                label=fam if si == 0 else "_nolegend_",
            )

            # Value label on bar (skip 0.0)
            if mean_val > 0.5:
                ax.text(
                    x, mean_val + 0.4,
                    f"{mean_val:.1f}",
                    ha="center", va="bottom",
                    fontsize=7.5, color="#333333",
                    zorder=4,
                )

            # Individual model dots
            for val in pts.get(fam, {}).get(s, []):
                ax.scatter(
                    x, val,
                    s=22,
                    marker="o",
                    color=color,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=5,
                    alpha=0.9,
                )

    # Axes
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=12, fontweight="bold")
    ax.set_ylabel("Failure Rate (%)", fontsize=12)
    ax.set_ylim(0, 48)
    ax.set_xlim(-group_w * 0.55, (n_settings - 1) * group_w + group_w * 0.55)

    # Vertical separators between setting groups
    for si in range(1, n_settings):
        sep_x = (si - 0.5) * group_w
        ax.axvline(sep_x, color="#cccccc", linewidth=0.8, linestyle="--", zorder=1)

    # Legend
    handles = [
        mpatches.Patch(color=FAMILY_COLORS[fam], alpha=0.82, label=fam)
        for fam in FAMILY_ORDER
    ]
    ax.legend(
        handles=handles,
        title="Model Family",
        title_fontsize=10,
        fontsize=9.5,
        loc="upper left",
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    # Annotation for GPT-5 Ancient China outlier
    gpt5_ac = means["GPT"]["Ancient_China"]
    ac_center = SETTINGS.index("Ancient_China") * group_w
    gpt5_x = ac_center + (FAMILY_ORDER.index("GPT") - (n_families - 1) / 2) * bar_w
    # ax.annotate(
    #     f"GPT-5: 45.8%",
    #     xy=(gpt5_x, gpt5_ac),
    #     xytext=(gpt5_x + 0.25, gpt5_ac + 4),
    #     fontsize=8.5,
    #     color=FAMILY_COLORS["GPT"],
    #     arrowprops=dict(arrowstyle="->", color=FAMILY_COLORS["GPT"], lw=1.2),
    # )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, default="setting_fr.csv")
    parser.add_argument("--output", type=str, default="setting_bar.pdf")
    args = parser.parse_args()

    rows = load(Path(args.input))
    plot(rows, Path(args.output))


if __name__ == "__main__":
    main()