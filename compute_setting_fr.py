#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Failure Rate by World Setting × Judge Model
Extracts setting from group_id field.

group_id format: {Skill}_{Setting}_{idx}
e.g. Track_1920s_Urban_07, Climb_Ancient_China_03

Usage:
    python compute_setting_fr.py --data_dir results_summary --output setting_fr.csv
"""

import re
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

GENERATORS = ["gpt", "claude", "gemini"]
CATEGORIES = ["CMB", "INV", "PHY", "PRO"]

JUDGE_MODELS = [
    "gpt-5.4-2026-03-05", "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07", "gpt-4.1-2025-04-14",
    "claude-opus-4-6", "claude-sonnet-4-6",
    "claude-sonnet-4-5", "claude-haiku-4-5-20251001",
    "gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3.5-flash",
    "deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v3.2", "deepseek-r1",
    "qwen3.7-max", "qwen3.6-flash", "qwen3-max",
    "qwen3-235b-a22b-thinking-2507", "qwen3-235b-a22b-instruct-2507",
]

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

# Known settings — used for normalization
SETTINGS = ["1920s_Urban", "2020s_Urban", "Ancient_China", "Wilderness"]
SETTING_LABELS = {
    "1920s_Urban":  "1920s Urban",
    "2020s_Urban":  "2020s Urban",
    "Ancient_China":"Ancient China",
    "Wilderness":   "Wilderness",
}


def extract_setting(group_id: str) -> str | None:
    """
    Extract setting from group_id.
    Format: {Skill}_{Setting}_{idx}
    Settings can be multi-word: 1920s_Urban, Ancient_China, etc.
    Strategy: match known setting strings.
    """
    for s in SETTINGS:
        if s in group_id:
            return s
    # Fallback: try to extract middle portion
    parts = group_id.split("_")
    if len(parts) >= 3:
        # Try two-word settings first
        for i in range(1, len(parts) - 1):
            candidate = "_".join(parts[i:i+2])
            if candidate in SETTINGS:
                return candidate
        # Single-word setting
        for i in range(1, len(parts)):
            if parts[i] in SETTINGS:
                return parts[i]
    return None


def parse_needs_roll(value: str):
    v = value.strip().lower()
    if v in ("true", "1"):
        return True
    if v in ("false", "0"):
        return False
    return None


def pct(n, d):
    return round(n / d * 100, 2) if d else float("nan")


def compute(data_dir: Path, output_path: Path):
    # {judge: {setting: {fail, total}}}
    counts = defaultdict(lambda: defaultdict(lambda: {"fail": 0, "total": 0}))
    missing = []

    for gen in GENERATORS:
        for cat in CATEGORIES:
            for judge in JUDGE_MODELS:
                fpath = data_dir / f"{gen}_{cat}_{judge}.csv"
                if not fpath.exists():
                    missing.append(fpath.name)
                    continue
                with open(fpath, encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        hj   = int(row.get("human_judge", 0))
                        pred = parse_needs_roll(row.get("needs_roll", ""))
                        gid  = row.get("group_id", "")

                        if pred is None or hj != 1:
                            continue

                        setting = extract_setting(gid)
                        if setting is None:
                            continue

                        counts[judge][setting]["total"] += 1
                        if pred is False:
                            counts[judge][setting]["fail"] += 1

    if missing:
        print(f"  [WARN] {len(missing)} files not found")

    # Write CSV
    fields = ["judge", "display_name"] + SETTINGS
    rows = []
    for judge in JUDGE_MODELS:
        row = {
            "judge":        judge,
            "display_name": DISPLAY_MAP.get(judge, judge),
        }
        for s in SETTINGS:
            c = counts[judge][s]
            row[s] = pct(c["fail"], c["total"])
        rows.append(row)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {output_path}")

    # Print summary
    print("\n=== Average FR per setting (across all judges) ===")
    for s in SETTINGS:
        vals = [float(r[s]) for r in rows if not __import__('math').isnan(float(r[s]))]
        if vals:
            print(f"  {SETTING_LABELS[s]:<15}  mean={sum(vals)/len(vals):.2f}  "
                  f"min={min(vals):.2f}  max={max(vals):.2f}")

    print("\n=== Per judge × setting FR ===")
    header = f"{'Model':<28}" + "".join(f"{SETTING_LABELS[s]:>16}" for s in SETTINGS)
    print(header)
    print("─" * len(header))
    for row in rows:
        line = f"{row['display_name']:<28}"
        for s in SETTINGS:
            v = row[s]
            line += f"{v:>16.2f}"
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="results_summary")
    parser.add_argument("--output",   type=str, default="setting_fr.csv")
    args = parser.parse_args()

    compute(Path(args.data_dir), Path(args.output))


if __name__ == "__main__":
    main()