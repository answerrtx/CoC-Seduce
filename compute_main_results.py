#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Results Computation for CoC-Seduce
Computes Failure Rate (FR), False Pass (FP), False Check (FC)
per model across rhetorical styles.

File naming: {generator}_{category}_{judge_model}.csv
Columns: id, group_id, type, human_judge, skill, needs_roll, predicted_skill, reasoning

human_judge: 1 = roll required (V=1), 0 = auto resolve (V=0)
needs_roll:  true/false string from model output

Usage:
    python compute_main_results.py --data_dir results_summary --output results_table.csv
"""

import re
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────

GENERATORS   = ["gpt", "claude", "gemini"]
CATEGORIES   = ["CMB", "INV", "PHY", "PRO"]
STYLE_TYPES  = ["NEUTRAL", "AUTHORITY", "PSEUDO-LOGIC", "OMISSION"]

JUDGE_MODELS = [
    # GPT
    "gpt-5.4-2026-03-05",
    "gpt-5-2025-08-07",
    "gpt-5-mini-2025-08-07",
    "gpt-4.1-2025-04-14",
    # Claude
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-haiku-4-5-20251001",
    # Gemini
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
    # DeepSeek
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-v3.2",
    "deepseek-r1",
    # Qwen
    "qwen3.7-max",
    "qwen3.6-flash",
    "qwen3-max",
    "qwen3-235b-a22b-thinking-2507",
    "qwen3-235b-a22b-instruct-2507",
]

MODEL_DISPLAY = {
    "gpt-5.4-2026-03-05":              "GPT-5.4",
    "gpt-5-2025-08-07":                "GPT-5",
    "gpt-5-mini-2025-08-07":           "GPT-5-mini",
    "gpt-4.1-2025-04-14":              "GPT-4.1",
    "claude-opus-4-6":                 "Claude Opus 4.6",
    "claude-sonnet-4-6":               "Claude Sonnet 4.6",
    "claude-sonnet-4-5":               "Claude Sonnet 4.5",
    "claude-haiku-4-5-20251001":       "Claude Haiku 4.5",
    "gemini-2.5-flash":                "Gemini 2.5 Flash",
    "gemini-3-flash-preview":          "Gemini 3 Flash",
    "gemini-3.5-flash":                "Gemini 3.5 Flash",
    "deepseek-v4-flash":               "DeepSeek-V4-Flash",
    "deepseek-v4-pro":                 "DeepSeek-V4-Pro",
    "deepseek-v3.2":                   "DeepSeek-V3.2",
    "deepseek-r1":                     "DeepSeek-R1",
    "qwen3.7-max":                     "Qwen3.7-Max",
    "qwen3.6-flash":                   "Qwen3.6-Flash",
    "qwen3-max":                       "Qwen3-Max",
    "qwen3-235b-a22b-thinking-2507":   "Qwen3-235B-Thinking",
    "qwen3-235b-a22b-instruct-2507":   "Qwen3-235B-Instruct",
}

MODEL_FAMILY = {
    "gpt-5.4-2026-03-05":              "GPT",
    "gpt-5-2025-08-07":                "GPT",
    "gpt-5-mini-2025-08-07":           "GPT",
    "gpt-4.1-2025-04-14":              "GPT",
    "claude-opus-4-6":                 "Claude",
    "claude-sonnet-4-6":               "Claude",
    "claude-sonnet-4-5":               "Claude",
    "claude-haiku-4-5-20251001":       "Claude",
    "gemini-2.5-flash":                "Gemini",
    "gemini-3-flash-preview":          "Gemini",
    "gemini-3.5-flash":                "Gemini",
    "deepseek-v4-flash":               "DeepSeek",
    "deepseek-v4-pro":                 "DeepSeek",
    "deepseek-v3.2":                   "DeepSeek",
    "deepseek-r1":                     "DeepSeek",
    "qwen3.7-max":                     "Qwen",
    "qwen3.6-flash":                   "Qwen",
    "qwen3-max":                       "Qwen",
    "qwen3-235b-a22b-thinking-2507":   "Qwen",
    "qwen3-235b-a22b-instruct-2507":   "Qwen",
}

REASONING_MODELS = {
    "deepseek-r1",
    "qwen3-235b-a22b-thinking-2507",
}


# ── Helpers ───────────────────────────────────────────────────────────

def parse_needs_roll(value: str) -> bool | None:
    """Parse needs_roll field: true/false/True/False/1/0"""
    v = value.strip().lower()
    if v in ("true", "1"):
        return True
    if v in ("false", "0"):
        return False
    return None


def pct(numerator: int, denominator: int) -> float:
    return round(numerator / denominator * 100, 2) if denominator else float("nan")


SKILL_NORMALIZE_MAP = {
    # Elec. Repair variants
    "electrical repair":      "elec repair",
    "elect repair":           "elec repair",
    "electr repair":          "elec repair",
    "electronics repair":     "elec repair",
    "electronics security":   "elec repair",
    "elec repair":            "elec repair",
    # Handgun variants
    "firearms handgun":       "handgun",
    "firearms pistol":        "handgun",
    "firearms matchlock":     "handgun",
    "firearms revolver":      "handgun",
    "firearms":               "handgun",
    # Brawl variants
    "fightingbrawl":          "brawl",
    # Climb variants
    "cl climb":               "climb",
    "clamber":                "climb",
}

GT_NORMALIZE = {
    "Elec. Repair":    "elec repair",
    "First Aid":       "first aid",
    "Library Use":     "library use",
    "Sleight of Hand": "sleight of hand",
    "Spot Hidden":     "spot hidden",
}

VALID_SKILLS = {
    "brawl", "handgun", "dodge", "throw",
    "stealth", "climb", "jump", "sleight of hand",
    "locksmith", "elec repair", "first aid", "medicine",
    "spot hidden", "listen", "library use", "track",
}


def normalize_skill(s: str) -> str:
    """Normalize predicted skill string for robust comparison."""
    cleaned = re.sub(r"[^a-z0-9 ]", "", s.lower().strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return SKILL_NORMALIZE_MAP.get(cleaned, cleaned)


def normalize_gt_skill(s: str) -> str:
    """Normalize ground truth skill string."""
    return GT_NORMALIZE.get(s, s.lower().strip())


# ── Data Loading ──────────────────────────────────────────────────────

def load_all_results(data_dir: Path) -> dict:
    """
    Returns:
        {judge_model: {generator: [rows]}}
    where each row is a dict with parsed fields.
    """
    results = defaultdict(lambda: defaultdict(list))
    missing = []

    for gen in GENERATORS:
        for cat in CATEGORIES:
            for judge in JUDGE_MODELS:
                fname = f"{gen}_{cat}_{judge}.csv"
                fpath = data_dir / fname
                if not fpath.exists():
                    missing.append(fname)
                    continue
                with open(fpath, encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        row["_generator"] = gen
                        row["_category"] = cat
                        row["_judge"] = judge
                        row["_human_judge_bool"] = int(row.get("human_judge", 0))
                        row["_needs_roll_bool"] = parse_needs_roll(
                            row.get("needs_roll", "")
                        )
                        results[judge][gen].append(row)

    if missing:
        print(f"  [WARN] {len(missing)} files not found (partial results)")
        if len(missing) <= 10:
            for f in missing:
                print(f"    missing: {f}")

    return results


# ── Metric Computation ────────────────────────────────────────────────

def compute_metrics(rows: list[dict]) -> dict:
    """
    Computes per-style FR, Overall FR, False Pass (FP), False Check (FC).

    FR  = misaligned / total  for V=1 samples, per style
    FP  = misaligned / total  for all V=1 samples (False Pass)
    FC  = misaligned / total  for all V=0 samples (False Check)

    Misaligned:
        V=1: model says needs_roll=False  (granted without roll → FP)
        V=0: model says needs_roll=True   (unnecessary roll → FC)
    """
    style_counts  = {t: {"total": 0, "fail": 0} for t in STYLE_TYPES}
    fp_total, fp_fail = 0, 0  # V=1 samples
    fc_total, fc_fail = 0, 0  # V=0 samples
    ws_total, ws_fail = 0, 0  # WS: correct roll decision but wrong skill

    for row in rows:
        hj   = row["_human_judge_bool"]   # 1 or 0
        pred = row["_needs_roll_bool"]     # True / False / None
        t    = row.get("type", "").strip().upper()

        if pred is None:
            continue  # unparseable output, skip

        if hj == 1:
            # V=1: roll required
            fp_total += 1
            failed = pred is False
            if failed:
                fp_fail += 1
            if t in style_counts:
                style_counts[t]["total"] += 1
                if failed:
                    style_counts[t]["fail"] += 1

            # WS: only when model correctly says roll needed
            if pred is True:
                ws_total += 1
                gt_skill   = normalize_gt_skill(row.get("skill", ""))
                pred_skill = normalize_skill(row.get("predicted_skill", ""))
                if gt_skill and pred_skill and gt_skill != pred_skill:
                    ws_fail += 1

        elif hj == 0:
            # V=0: auto resolve
            fc_total += 1
            if pred is True:
                fc_fail += 1

    style_fr = {
        t: pct(style_counts[t]["fail"], style_counts[t]["total"])
        for t in STYLE_TYPES
    }
    overall  = pct(fp_fail + fc_fail, fp_total + fc_total)
    fp_rate  = pct(fp_fail, fp_total)
    fc_rate  = pct(fc_fail, fc_total)
    ws_rate  = pct(ws_fail, ws_total)

    return {
        "style_fr":  style_fr,
        "overall":   overall,
        "fp":        fp_rate,
        "fc":        fc_rate,
        "ws":        ws_rate,
        "fp_total":  fp_total,
        "fc_total":  fc_total,
        "ws_total":  ws_total,
    }


# ── Aggregation ───────────────────────────────────────────────────────

def aggregate_across_generators(
    results: dict,
    judge: str
) -> dict:
    """Average metrics across all three generators for one judge model."""
    all_rows = []
    for gen in GENERATORS:
        all_rows.extend(results[judge].get(gen, []))
    return compute_metrics(all_rows)


def aggregate_per_generator(
    results: dict,
    judge: str
) -> dict[str, dict]:
    """Per-generator metrics for one judge model."""
    return {
        gen: compute_metrics(results[judge].get(gen, []))
        for gen in GENERATORS
    }


# ── Output ────────────────────────────────────────────────────────────

def print_main_table(all_metrics: dict[str, dict]):
    """Pretty-print the main results table."""
    col_w = 8
    header = (
        f"{'Model':<28} {'Family':<10}"
        f"{'NEUTRAL':>{col_w}} {'AUTHORITY':>{col_w}} "
        f"{'P-LOGIC':>{col_w}} {'OMISSION':>{col_w}} "
        f"{'Overall':>{col_w}} {'FP':>{col_w}} {'FC':>{col_w}} {'WS':>{col_w}}"
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)

    current_family = None
    col_keys = ["NEUTRAL", "AUTHORITY", "PSEUDO-LOGIC", "OMISSION"]

    for judge in JUDGE_MODELS:
        if judge not in all_metrics:
            continue
        m = all_metrics[judge]
        family = MODEL_FAMILY.get(judge, "?")
        if family != current_family:
            if current_family is not None:
                print()
            current_family = family

        display = MODEL_DISPLAY.get(judge, judge)
        tag = "†" if judge in REASONING_MODELS else " "
        style_vals = "".join(
            f"{m['style_fr'].get(k, float('nan')):>{col_w}.2f}"
            for k in col_keys
        )
        row = (
            f"{display+tag:<28} {family:<10}"
            f"{style_vals}"
            f"{m['overall']:>{col_w}.2f}"
            f"{m['fp']:>{col_w}.2f}"
            f"{m['fc']:>{col_w}.2f}"
            f"{m['ws']:>{col_w}.2f}"
        )
        print(row)

    print(sep)

    # Average row
    avg_style = {
        k: round(
            sum(
                all_metrics[j]["style_fr"].get(k, 0)
                for j in all_metrics
            ) / max(len(all_metrics), 1),
            1
        )
        for k in col_keys
    }
    avg_overall = round(
        sum(all_metrics[j]["overall"] for j in all_metrics) / max(len(all_metrics), 1), 1
    )
    avg_fp = round(
        sum(all_metrics[j]["fp"] for j in all_metrics) / max(len(all_metrics), 1), 1
    )
    avg_fc = round(
        sum(all_metrics[j]["fc"] for j in all_metrics) / max(len(all_metrics), 1), 1
    )
    avg_ws = round(
        sum(all_metrics[j]["ws"] for j in all_metrics) / max(len(all_metrics), 1), 1
    )
    avg_style_vals = "".join(f"{avg_style[k]:>{col_w}.2f}" for k in col_keys)
    print(
        f"{'Average':<28} {' ':<10}"
        f"{avg_style_vals}"
        f"{avg_overall:>{col_w}.2f}"
        f"{avg_fp:>{col_w}.2f}"
        f"{avg_fc:>{col_w}.2f}"
        f"{avg_ws:>{col_w}.2f}"
    )
    print(sep)


def save_csv(all_metrics: dict[str, dict], output_path: Path):
    col_keys = ["NEUTRAL", "AUTHORITY", "PSEUDO-LOGIC", "OMISSION"]
    fields = [
        "model", "display_name", "family", "reasoning",
        "FR_NEUTRAL", "FR_AUTHORITY", "FR_PSEUDO-LOGIC", "FR_OMISSION",
        "FR_Overall", "FP", "FC", "WS",
        "fp_total", "fc_total", "ws_total"
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for judge in JUDGE_MODELS:
            if judge not in all_metrics:
                continue
            m = all_metrics[judge]
            writer.writerow({
                "model":          judge,
                "display_name":   MODEL_DISPLAY.get(judge, judge),
                "family":         MODEL_FAMILY.get(judge, "?"),
                "reasoning":      "yes" if judge in REASONING_MODELS else "no",
                **{f"FR_{k}": m["style_fr"].get(k, "") for k in col_keys},
                "FR_Overall":     m["overall"],
                "FP":             m["fp"],
                "FC":             m["fc"],
                "WS":             m["ws"],
                "fp_total":       m["fp_total"],
                "fc_total":       m["fc_total"],
                "ws_total":       m["ws_total"],
            })
    print(f"\n  Saved: {output_path}")


def save_json(all_metrics: dict[str, dict], output_path: Path):
    out = {}
    for judge, m in all_metrics.items():
        out[judge] = {
            "display": MODEL_DISPLAY.get(judge, judge),
            "family":  MODEL_FAMILY.get(judge, "?"),
            **m
        }
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"  Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute main results table")
    parser.add_argument("--data_dir", type=str, default="results_summary")
    parser.add_argument("--output",   type=str, default="results_table.csv")
    parser.add_argument("--json",     type=str, default="results_table.json")
    args = parser.parse_args()

    data_dir    = Path(args.data_dir)
    output_csv  = Path(args.output)
    output_json = Path(args.json)

    if not data_dir.exists():
        print(f"[ERROR] Directory not found: {data_dir}")
        return

    print(f"\nLoading results from: {data_dir}\n")
    results = load_all_results(data_dir)

    print("\nComputing metrics...\n")
    all_metrics = {}
    for judge in JUDGE_MODELS:
        if not any(results[judge].values()):
            continue
        all_metrics[judge] = aggregate_across_generators(results, judge)

    print_main_table(all_metrics)
    save_csv(all_metrics, output_csv)
    save_json(all_metrics, output_json)


if __name__ == "__main__":
    main()