#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot: Generator × Target Model Heatmap of Failure Rate
Usage:
    python plot_generator_heatmap.py \
        --input generator_target_fr.csv \
        --results results_table.csv \
        --output generator_heatmap.pdf
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
    "gpt-5.4-2026-03-05":           "GPT",
    "gpt-5-2025-08-07":             "GPT",
    "gpt-5-mini-2025-08-07":        "GPT",
    "gpt-4.1-2025-04-14":           "GPT",
    "claude-opus-4-6":              "Claude",
    "claude-sonnet-4-6":            "Claude",
    "claude-sonnet-4-5":            "Claude",
    "claude-haiku-4-5-20251001":    "Claude",
    "gemini-2.5-flash":             "Gemini",
    "gemini-3-flash-preview":       "Gemini",
    "gemini-3.5-flash":             "Gemini",
    "deepseek-v4-flash":            "DeepSeek",
    "deepseek-v4-pro":              "DeepSeek",
    "deepseek-v3.2":                "DeepSeek",
    "deepseek-r1":                  "DeepSeek",
    "qwen3.7-max":                  "Qwen",
    "qwen3.6-flash":                "Qwen",
    "qwen3-max":                    "Qwen",
    "qwen3-235b-a22b-thinking-2507":"Qwen",
    "qwen3-235b-a22b-instruct-2507":"Qwen",
}

DISPLAY_MAP = {
    "gpt-5.4-2026-03-05":           "GPT-5.4",
    "gpt-5-2025-08-07":             "GPT-5",
    "gpt-5-mini-2025-08-07":        "GPT-5-mini",
    "gpt-4.1-2025-04-14":           "GPT-4.1",
    "claude-opus-4-6":              "Opus 4.6",
    "claude-sonnet-4-6":            "Sonnet 4.6",
    "claude-sonnet-4-5":            "Sonnet 4.5",
    "claude-haiku-4-5-20251001":    "Haiku 4.5",
    "gemini-2.5-flash":             "Gemini 2.5F",
    "gemini-3-flash-preview":       "Gemini 3F",
    "gemini-3.5-flash":             "Gemini 3.5F",
    "deepseek-v4-flash":            "V4-Flash",
    "deepseek-v4-pro":              "V4-Pro",
    "deepseek-v3.2":                "V3.2",
    "deepseek-r1":                  "R1†",
    "qwen3.7-max":                  "3.7-Max",
    "qwen3.6-flash":                "3.6-Flash",
    "qwen3-max":                    "3-Max",
    "qwen3-235b-a22b-thinking-2507":"235B-T†",
    "qwen3-235b-a22b-instruct-2507":"235B-I",
}

FAMILY_ORDER = ["GPT", "Claude", "Gemini", "DeepSeek", "Qwen"]
MODEL_ORDER = [
    "gpt-4.1-2025-04-14", "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07", "gpt-5.4-2026-03-05",
    "claude-haiku-4-5-20251001", "claude-sonnet-4-5",
    "claude-sonnet-4-6", "claude-opus-4-6",
    "gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3.5-flash",
    "deepseek-v3.2", "deepseek-v4-flash",
    "deepseek-v4-pro", "deepseek-r1",
    "qwen3-max", "qwen3.6-flash", "qwen3.7-max",
    "qwen3-235b-a22b-instruct-2507", "qwen3-235b-a22b-thinking-2507",
]
GENERATORS = ["gpt", "claude", "gemini"]
GEN_LABELS = ["GPT-5.4", "Claude Sonnet 4.6", "Gemini 3.5 Flash"]
GEN_FAMILY = {"gpt": "GPT", "claude": "Claude", "gemini": "Gemini 3.5 Flash"}


def load_data(gen_path, results_path):
    # generator × target FR
    gen_target = defaultdict(dict)
    with open(gen_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gen_target[r['judge']][r['generator']] = float(r['FR'])

    return gen_target


def plot(gen_target, output_path):
    matplotlib.rcParams.update({
        "font.family":       "Arial",
        "font.size":         10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  False,
        "axes.spines.bottom":False,
    })

    n_models = len(MODEL_ORDER)
    n_gen    = len(GENERATORS)

    # Build matrix: rows=models, cols=generators
    matrix = np.zeros((n_models, n_gen))
    for mi, model in enumerate(MODEL_ORDER):
        for gi, gen in enumerate(GENERATORS):
            matrix[mi, gi] = gen_target.get(model, {}).get(gen, np.nan)

    fig, ax = plt.subplots(figsize=(6, 9))

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=0, vmax=40,
    )

    # Cell annotations
    for mi in range(n_models):
        for gi in range(n_gen):
            val = matrix[mi, gi]
            if not np.isnan(val):
                textcolor = "white" if val > 25 else "black"
                ax.text(gi, mi, f"{val:.1f}",
                        ha="center", va="center",
                        fontsize=12, color=textcolor, fontweight="bold")

    # Axes
    ax.set_xticks(range(n_gen))
    ax.set_xticklabels(GEN_LABELS, fontsize=12, fontweight="bold")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xlabel("Adversarial Generator", fontsize=12, labelpad=8)

    ax.set_yticks(range(n_models))
    ax.set_yticklabels(
        [DISPLAY_MAP.get(m, m) for m in MODEL_ORDER],
        fontsize=12,
    )

    # Family separators and labels
    family_boundaries = []
    current_fam = None
    fam_start = 0
    fam_ranges = []
    for mi, model in enumerate(MODEL_ORDER):
        fam = FAMILY_MAP.get(model, "?")
        if fam != current_fam:
            if current_fam is not None:
                fam_ranges.append((current_fam, fam_start, mi - 1))
                ax.axhline(mi - 0.5, color="white", linewidth=2.5, zorder=3)
            current_fam = fam
            fam_start = mi
    fam_ranges.append((current_fam, fam_start, len(MODEL_ORDER) - 1))

    # Family labels on right
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([
        (r[1] + r[2]) / 2 for r in fam_ranges
    ])
    ax2.set_yticklabels(
        [r[0] for r in fam_ranges],
        fontsize=12, fontweight="bold",
    )
    ax2.spines[:].set_visible(False)
    ax2.tick_params(length=0)

    # Home-series diagonal highlight
    home_pairs = {
        "GPT": "gpt", "Claude": "claude", "Gemini": "gemini"
    }
    for mi, model in enumerate(MODEL_ORDER):
        fam = FAMILY_MAP.get(model, "?")
        home_gen = home_pairs.get(fam)
        if home_gen and home_gen in GENERATORS:
            gi = GENERATORS.index(home_gen)
            rect = mpatches.FancyBboxPatch(
                (gi - 0.48, mi - 0.48), 0.96, 0.96,
                boxstyle="round,pad=0.02",
                linewidth=2, edgecolor="#2c2c2c",
                facecolor="none", zorder=4,
            )
            ax.add_patch(rect)

    # Colorbar
    cbar_ax = fig.add_axes([0.15, 0.04, 0.73, 0.018])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Failure Rate (%)", fontsize=10, labelpad=4)
    cbar.ax.tick_params(labelsize=9)

    # ax.set_title(
    #     "FR (%) by Adversarial Generator × Target Model\n"
    #     "□ = home-series pair",
    #     fontsize=11, pad=30,
    # )

    plt.subplots_adjust(bottom=0.10, top=0.93, left=0.15, right=0.88)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   type=str, default="generator_target_fr.csv")
    parser.add_argument("--results", type=str, default="results_table.csv")
    parser.add_argument("--output",  type=str, default="generator_heatmap.pdf")
    args = parser.parse_args()

    gen_target = load_data(Path(args.input), Path(args.results))
    plot(gen_target, Path(args.output))


if __name__ == "__main__":
    main()