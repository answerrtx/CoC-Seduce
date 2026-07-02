#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute response time statistics per judge model.
File naming: {generator}_{category}_{judge}_{skill}_{setting}_{group}_{id}_{variant}.json

Usage:
    python compute_response_time.py --results_dir results --output response_time.csv
"""

import json
import csv
import argparse
import math
from pathlib import Path
from collections import defaultdict

GENERATORS = ["gpt", "claude", "gemini"]

DISPLAY_MAP = {
    "gpt-5.4-2026-03-05":           "GPT-5.4",
    "gpt-5-2025-08-07":             "GPT-5",
    "gpt-5-mini-2025-08-07":        "GPT-5-mini",
    "gpt-4.1-2025-04-14":           "GPT-4.1",
    "claude-opus-4-6":              "Claude Opus 4.6",
    "claude-sonnet-4-6":            "Claude Sonnet 4.6",
    "claude-sonnet-4-5":            "Claude Sonnet 4.5",
    "claude-haiku-4-5-20251001":    "Claude Haiku 4.5",
    "gemini-2.5-flash":             "Gemini 2.5 Flash",
    "gemini-3-flash-preview":       "Gemini 3 Flash",
    "gemini-3.5-flash":             "Gemini 3.5 Flash",
    "deepseek-v4-flash":            "DeepSeek-V4-Flash",
    "deepseek-v4-pro":              "DeepSeek-V4-Pro",
    "deepseek-v3.2":                "DeepSeek-V3.2",
    "deepseek-r1":                  "DeepSeek-R1",
    "qwen3.7-max":                  "Qwen3.7-Max",
    "qwen3.6-flash":                "Qwen3.6-Flash",
    "qwen3-max":                    "Qwen3-Max",
    "qwen3-235b-a22b-thinking-2507":"Qwen3-235B-Thinking",
    "qwen3-235b-a22b-instruct-2507":"Qwen3-235B-Instruct",
}

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

MODEL_ORDER = [
    "gpt-5.4-2026-03-05", "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07", "gpt-4.1-2025-04-14",
    "claude-opus-4-6", "claude-sonnet-4-6",
    "claude-sonnet-4-5", "claude-haiku-4-5-20251001",
    "gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3.5-flash",
    "deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v3.2", "deepseek-r1",
    "qwen3.7-max", "qwen3.6-flash", "qwen3-max",
    "qwen3-235b-a22b-thinking-2507", "qwen3-235b-a22b-instruct-2507",
]


def extract_judge(filename: str) -> str | None:
    """
    Extract judge model from filename.
    Format: {generator}_{category}_{judge}_{...}.json
    Generator is one of: gpt, claude, gemini
    Category is one of: CMB, INV, PHY, PRO
    """
    name = filename.replace(".json", "")
    parts = name.split("_")
    if len(parts) < 3:
        return None
    # First part is generator, second is category, rest starts with judge
    # Judge model string itself contains underscores, so we match known judges
    remainder = "_".join(parts[2:])
    for judge in MODEL_ORDER:
        if remainder.startswith(judge):
            return judge
    return None


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def median(vals):
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    return (s[n // 2] + s[(n - 1) // 2]) / 2.0


def stddev(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))


def compute(results_dir: Path, output_path: Path):
    # {judge: [elapsed_s, ...]}
    times = defaultdict(list)
    unmatched = 0

    for fpath in results_dir.glob("*.json"):
        judge = extract_judge(fpath.name)
        if judge is None:
            unmatched += 1
            continue
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            elapsed = data.get("elapsed_s")
            if elapsed is not None:
                times[judge].append(float(elapsed))
        except Exception as e:
            print(f"  [WARN] {fpath.name}: {e}")

    if unmatched:
        print(f"  [WARN] {unmatched} files could not be matched to a judge model")

    # ── Print summary ─────────────────────────────────────────────────
    col = 10
    header = (f"{'Model':<28} {'Family':<10} {'N':>{col}} "
              f"{'Total(s)':>{col}} {'Mean(s)':>{col}} "
              f"{'Median(s)':>{col}} {'Std(s)':>{col}} {'Max(s)':>{col}}")
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)

    current_family = None
    rows_out = []

    for judge in MODEL_ORDER:
        vals = times.get(judge, [])
        family = FAMILY_MAP.get(judge, "?")
        display = DISPLAY_MAP.get(judge, judge)

        if family != current_family:
            if current_family is not None:
                print()
            current_family = family

        if not vals:
            print(f"  {display:<26} {family:<10} {'N/A':>{col}}")
            continue

        total  = sum(vals)
        mn     = mean(vals)
        med    = median(vals)
        std    = stddev(vals)
        mx     = max(vals)
        n      = len(vals)

        print(f"  {display:<26} {family:<10} "
              f"{n:>{col},} {total:>{col},.1f} {mn:>{col}.3f} "
              f"{med:>{col}.3f} {std:>{col}.3f} {mx:>{col}.3f}")

        rows_out.append({
            "judge":        judge,
            "display_name": display,
            "family":       family,
            "n":            n,
            "total_s":      round(total, 2),
            "mean_s":       round(mn, 3),
            "median_s":     round(med, 3),
            "std_s":        round(std, 3),
            "max_s":        round(mx, 3),
            "total_min":    round(total / 60, 2),
            "total_hr":     round(total / 3600, 3),
        })

    print(sep)

    # Grand total
    all_vals = [v for vs in times.values() for v in vs]
    if all_vals:
        grand_total = sum(all_vals)
        print(f"\n  Grand total: {grand_total:,.1f}s  "
              f"= {grand_total/60:,.1f} min  "
              f"= {grand_total/3600:.2f} hr")
        print(f"  Across {len(all_vals):,} samples from {len(times)} models")

    # ── Save CSV ──────────────────────────────────────────────────────
    fields = ["judge","display_name","family","n",
              "total_s","mean_s","median_s","std_s","max_s",
              "total_min","total_hr"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\n  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output",      type=str, default="response_time.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERROR] Directory not found: {results_dir}")
        return

    print(f"\nScanning: {results_dir}\n")
    compute(results_dir, Path(args.output))


if __name__ == "__main__":
    main()