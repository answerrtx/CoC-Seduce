#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator Style Analysis
Analyzes attack length and vocabulary style across GPT, Claude, Gemini generators.
Usage: python analyze_generators.py --data_dir /path/to/scenes/query
"""

import os
import re
import csv
import argparse
import math
from pathlib import Path
from collections import Counter

GENERATORS = ["gpt", "claude", "gemini"]
CATEGORIES = ["CMB", "INV", "PHY", "PRO"]
TYPES = ["NEUTRAL", "AUTHORITY", "PSEUDO-LOGIC", "OMISSION"]

# Simple stopwords
STOPWORDS = {
    "i", "the", "a", "an", "to", "and", "is", "in", "of", "my", "it",
    "that", "this", "for", "on", "with", "at", "by", "from", "as", "are",
    "was", "be", "have", "has", "will", "can", "do", "not", "but", "or",
    "so", "if", "up", "out", "me", "he", "she", "we", "they", "you",
    "its", "their", "our", "his", "her", "am", "been", "being", "had",
    "would", "could", "should", "just", "than", "then", "when", "there",
    "what", "which", "who", "how", "all", "any", "no", "s", "t", "re"
}


# ── I/O ──────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_all(data_dir: Path) -> dict[str, list[dict]]:
    """Returns {generator: [rows]}"""
    data = {}
    for gen in GENERATORS:
        rows = []
        for cat in CATEGORIES:
            p = data_dir / f"{gen}_{cat}.csv"
            if p.exists():
                rows.extend(load_csv(p))
            else:
                print(f"  [WARN] missing: {p.name}")
        data[gen] = rows
        print(f"  Loaded {len(rows):>5} rows  ← {gen}")
    return data


# ── Tokenizer ────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def token_count(text: str) -> int:
    """Count words + em/en dashes as separate tokens."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    dashes = re.findall(r"—|--|–", text)
    return len(words) + len(dashes)


def dash_rate(text: str) -> float:
    """Dashes per 100 tokens."""
    total = token_count(text)
    dashes = len(re.findall(r"—|--|–", text))
    return round(dashes / total * 100, 2) if total else 0.0


# ── Length Analysis ──────────────────────────────────────────────────

def mean(values):
    return sum(values) / len(values) if values else 0.0


def median(values):
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return (s[n // 2] + s[(n - 1) // 2]) / 2


def length_analysis(data: dict[str, list[dict]]) -> dict:
    """
    Returns per-generator, per-type length stats for 'player' field.
    Also returns scenario_truth length for reference.
    """
    results = {}
    for gen, rows in data.items():
        by_type = {t: {"lengths": [], "dash_rates": []} for t in TYPES}
        truth_lengths = []
        for row in rows:
            t = row.get("type", "").strip().upper()
            player = row.get("player", "")
            truth = row.get("scenario_truth", "")
            if t in by_type:
                by_type[t]["lengths"].append(token_count(player))
                by_type[t]["dash_rates"].append(dash_rate(player))
            truth_lengths.append(token_count(truth))

        results[gen] = {
            "by_type": {
                t: {
                    "n": len(v["lengths"]),
                    "mean": round(mean(v["lengths"]), 1),
                    "median": round(median(v["lengths"]), 1),
                    "min": min(v["lengths"]) if v["lengths"] else 0,
                    "max": max(v["lengths"]) if v["lengths"] else 0,
                    "dash_rate": round(mean(v["dash_rates"]), 2),
                }
                for t, v in by_type.items()
            },
            "truth": {
                "mean": round(mean(truth_lengths), 1),
                "median": round(median(truth_lengths), 1),
            }
        }
    return results


# ── TF-IDF Vocabulary Analysis ───────────────────────────────────────

def build_tf(rows: list[dict], field: str = "player") -> Counter:
    counts = Counter()
    for row in rows:
        tokens = [t for t in tokenize(row.get(field, "")) if t not in STOPWORDS]
        counts.update(tokens)
    return counts


def compute_tfidf(
    gen_tfs: dict[str, Counter],
    top_n: int = 20
) -> dict[str, list[tuple[str, float]]]:
    """
    Simple corpus-level TF-IDF where each generator is a 'document'.
    IDF = log(N / df), df = number of generators the term appears in.
    """
    all_terms = set()
    for tf in gen_tfs.values():
        all_terms.update(tf.keys())

    N = len(gen_tfs)
    idf = {}
    for term in all_terms:
        df = sum(1 for tf in gen_tfs.values() if tf[term] > 0)
        idf[term] = math.log(N / df) if df else 0.0

    results = {}
    for gen, tf in gen_tfs.items():
        total = sum(tf.values()) or 1
        scored = []
        for term, count in tf.items():
            tf_score = count / total
            score = tf_score * idf[term]
            if score > 0:
                scored.append((term, round(score, 6)))
        scored.sort(key=lambda x: -x[1])
        results[gen] = scored[:top_n]

    return results


def vocab_analysis(
    data: dict[str, list[dict]],
    top_n: int = 20
) -> dict:
    """Vocabulary analysis split by adversarial vs neutral."""
    adv_types = {"AUTHORITY", "PSEUDO-LOGIC", "OMISSION"}

    gen_tfs_all = {}
    gen_tfs_adv = {}
    gen_tfs_neutral = {}

    for gen, rows in data.items():
        all_rows = rows
        adv_rows = [r for r in rows if r.get("type", "").strip().upper() in adv_types]
        neu_rows = [r for r in rows if r.get("type", "").strip().upper() == "NEUTRAL"]

        gen_tfs_all[gen] = build_tf(all_rows)
        gen_tfs_adv[gen] = build_tf(adv_rows)
        gen_tfs_neutral[gen] = build_tf(neu_rows)

    return {
        "all": compute_tfidf(gen_tfs_all, top_n),
        "adversarial": compute_tfidf(gen_tfs_adv, top_n),
        "neutral": compute_tfidf(gen_tfs_neutral, top_n),
    }


# ── Printers ─────────────────────────────────────────────────────────

def print_separator(char="─", width=72):
    print(char * width)


def print_length_report(results: dict):
    print_separator("═")
    print("  LENGTH ANALYSIS  (token count of player statements)")
    print_separator("═")

    # Header
    col_w = 12
    type_cols = TYPES
    header = f"{'Generator':<10}" + "".join(f"{t:>{col_w}}" for t in type_cols)
    print(header)
    print_separator()

    for gen in GENERATORS:
        stats = results[gen]["by_type"]
        row = f"{gen:<10}"
        for t in type_cols:
            s = stats.get(t, {})
            row += f"{s.get('mean', 0):>{col_w}.1f}"
        print(row)

    print()
    print("  Mean token count by type (all generators)")
    print_separator()
    for t in TYPES:
        vals = [results[g]["by_type"][t]["mean"] for g in GENERATORS]
        overall = round(mean(vals), 1)
        bar = "█" * int(overall / 2)
        print(f"  {t:<15} {overall:>6.1f}  {bar}")

    print()
    print("  Scenario truth length (reference)")
    print_separator()
    for gen in GENERATORS:
        m = results[gen]["truth"]["mean"]
        print(f"  {gen:<10} mean: {m:.1f} tokens")
    print()


def print_vocab_report(results: dict, section: str = "adversarial"):
    print_separator("═")
    print(f"  VOCABULARY ANALYSIS  [{section.upper()}]  (TF-IDF top terms per generator)")
    print_separator("═")

    section_data = results[section]
    max_rows = max(len(v) for v in section_data.values())

    col_w = 22
    header = "".join(f"{g:<{col_w}}" for g in GENERATORS)
    print(header)
    print_separator()

    for i in range(min(max_rows, 20)):
        row = ""
        for gen in GENERATORS:
            terms = section_data.get(gen, [])
            if i < len(terms):
                term, score = terms[i]
                row += f"{term:<16} {score:.4f}  "
            else:
                row += " " * col_w
        print(row)
    print()


def print_per_type_length(results: dict):
    print_separator("═")
    print("  DETAILED LENGTH STATS  (mean / median / max)")
    print_separator("═")
    for gen in GENERATORS:
        print(f"\n  {gen.upper()}")
        print_separator("─", 50)
        for t in TYPES:
            s = results[gen]["by_type"][t]
            print(f"    {t:<15}  mean={s['mean']:>5.1f}  "
                  f"median={s['median']:>5.1f}  max={s['max']:>4d}  "
                  f"dash_rate={s['dash_rate']:>5.2f}%  n={s['n']}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generator style analysis")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="scenes/query",
        help="Path to directory containing generator CSV files"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of top TF-IDF terms to show per generator"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] Directory not found: {data_dir}")
        return

    print(f"\nLoading data from: {data_dir}\n")
    data = load_all(data_dir)

    print("\nRunning length analysis...")
    length_results = length_analysis(data)

    print("Running vocabulary analysis...")
    vocab_results = vocab_analysis(data, top_n=args.top_n)

    print("\n")
    print_length_report(length_results)
    print_per_type_length(length_results)
    print()
    print_vocab_report(vocab_results, section="adversarial")
    print_vocab_report(vocab_results, section="neutral")
    print_vocab_report(vocab_results, section="all")


if __name__ == "__main__":
    main()