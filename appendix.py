#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Breakdown of failure counts by model, world setting, and player framing.

Outputs:
  - breakdown.csv: per (judge, setting, type) failure count and rate
  - breakdown_summary.csv: aggregated views

Usage:
    python compute_breakdown.py --data_dir results_summary --output_dir breakdown_results
"""

import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict

GENERATORS = ["gpt", "claude", "gemini"]
CATEGORIES = ["CMB", "INV", "PHY", "PRO"]
TYPES      = ["NEUTRAL", "AUTHORITY", "PSEUDO-LOGIC", "OMISSION"]
SETTINGS   = ["1920s_Urban", "2020s_Urban", "Ancient_China", "Wilderness"]

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


def extract_setting(group_id: str) -> str | None:
    for s in SETTINGS:
        if s in group_id:
            return s
    return None


def parse_needs_roll(value: str):
    v = value.strip().lower()
    if v in ("true", "1"):  return True
    if v in ("false", "0"): return False
    return None


def pct(n, d):
    return round(n / d * 100, 2) if d else float("nan")


def compute(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # {judge: {setting: {type: {fail, total}}}}
    counts = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: {"fail": 0, "total": 0})
        )
    )

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
                        t    = row.get("type", "").strip().upper()
                        gid  = row.get("group_id", "")

                        if pred is None or hj != 1:
                            continue

                        setting = extract_setting(gid)
                        if setting is None or t not in TYPES:
                            continue

                        counts[judge][setting][t]["total"] += 1
                        if pred is False:
                            counts[judge][setting][t]["fail"] += 1

    if missing:
        print(f"  [WARN] {len(missing)} files not found")

    # ── 1. Full breakdown CSV ─────────────────────────────────────────
    breakdown_path = output_dir / "breakdown.csv"
    fields = ["judge", "display_name", "family",
              "setting", "type", "fail", "total", "FR"]
    rows_out = []

    for judge in JUDGE_MODELS:
        for setting in SETTINGS:
            for t in TYPES:
                c = counts[judge][setting][t]
                rows_out.append({
                    "judge":        judge,
                    "display_name": DISPLAY_MAP.get(judge, judge),
                    "family":       FAMILY_MAP.get(judge, "?"),
                    "setting":      setting,
                    "type":         t,
                    "fail":         c["fail"],
                    "total":        c["total"],
                    "FR":           pct(c["fail"], c["total"]),
                })

    with open(breakdown_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"  Saved: {breakdown_path}")

    # ── 2. Setting × Type summary (averaged across all judges) ────────
    setting_type = defaultdict(lambda: defaultdict(lambda: {"fail": 0, "total": 0}))
    for judge in JUDGE_MODELS:
        for setting in SETTINGS:
            for t in TYPES:
                c = counts[judge][setting][t]
                setting_type[setting][t]["fail"]  += c["fail"]
                setting_type[setting][t]["total"] += c["total"]

    summary_path = output_dir / "setting_type_summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["setting", "type", "fail", "total", "FR"])
        for setting in SETTINGS:
            for t in TYPES:
                c = setting_type[setting][t]
                writer.writerow([
                    setting, t, c["fail"], c["total"],
                    pct(c["fail"], c["total"])
                ])
    print(f"  Saved: {summary_path}")

    # ── 3. Model × Type summary (averaged across all settings) ────────
    model_type = defaultdict(lambda: defaultdict(lambda: {"fail": 0, "total": 0}))
    for judge in JUDGE_MODELS:
        for setting in SETTINGS:
            for t in TYPES:
                c = counts[judge][setting][t]
                model_type[judge][t]["fail"]  += c["fail"]
                model_type[judge][t]["total"] += c["total"]

    model_type_path = output_dir / "model_type_summary.csv"
    with open(model_type_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["judge", "display_name", "family",
                         "type", "fail", "total", "FR"])
        for judge in JUDGE_MODELS:
            for t in TYPES:
                c = model_type[judge][t]
                writer.writerow([
                    judge, DISPLAY_MAP.get(judge, judge),
                    FAMILY_MAP.get(judge, "?"),
                    t, c["fail"], c["total"],
                    pct(c["fail"], c["total"])
                ])
    print(f"  Saved: {model_type_path}")

    # ── 4. Model × Setting summary (averaged across all types) ────────
    model_setting = defaultdict(lambda: defaultdict(lambda: {"fail": 0, "total": 0}))
    for judge in JUDGE_MODELS:
        for setting in SETTINGS:
            for t in TYPES:
                c = counts[judge][setting][t]
                model_setting[judge][setting]["fail"]  += c["fail"]
                model_setting[judge][setting]["total"] += c["total"]

    model_setting_path = output_dir / "model_setting_summary.csv"
    with open(model_setting_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["judge", "display_name", "family"] + SETTINGS)
        for judge in JUDGE_MODELS:
            row = [judge, DISPLAY_MAP.get(judge, judge), FAMILY_MAP.get(judge, "?")]
            for setting in SETTINGS:
                c = model_setting[judge][setting]
                row.append(pct(c["fail"], c["total"]))
            writer.writerow(row)
    print(f"  Saved: {model_setting_path}")

    # ── Console summary ───────────────────────────────────────────────
    print("\n=== Setting × Type FR matrix (avg across all judges) ===")
    header = f"{'':>15}" + "".join(f"{t:>15}" for t in TYPES)
    print(header)
    print("─" * len(header))
    for setting in SETTINGS:
        row = f"  {setting:<13}"
        for t in TYPES:
            c = setting_type[setting][t]
            row += f"{pct(c['fail'], c['total']):>15.2f}"
        print(row)

    print("\n=== Type FR avg across all settings and judges ===")
    type_totals = defaultdict(lambda: {"fail": 0, "total": 0})
    for setting in SETTINGS:
        for t in TYPES:
            c = setting_type[setting][t]
            type_totals[t]["fail"]  += c["fail"]
            type_totals[t]["total"] += c["total"]
    for t in TYPES:
        c = type_totals[t]
        print(f"  {t:<15} FR={pct(c['fail'], c['total']):.2f}%  "
              f"({c['fail']}/{c['total']})")

    print("\n=== Setting FR avg across all types and judges ===")
    setting_totals = defaultdict(lambda: {"fail": 0, "total": 0})
    for setting in SETTINGS:
        for t in TYPES:
            c = setting_type[setting][t]
            setting_totals[setting]["fail"]  += c["fail"]
            setting_totals[setting]["total"] += c["total"]
    for setting in SETTINGS:
        c = setting_totals[setting]
        print(f"  {setting:<15} FR={pct(c['fail'], c['total']):.2f}%  "
              f"({c['fail']}/{c['total']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default="results_summary")
    parser.add_argument("--output_dir", type=str, default="breakdown_results")
    args = parser.parse_args()
    compute(Path(args.data_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()