#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from pathlib import Path

ROOT      = Path(__file__).parent.parent
QUERY_DIR = ROOT / "scenes" / "query"

# model_name → csv directory
SOURCES = {
    "gpt":    ROOT / "scenes" / "csv_gpt",
    "claude": ROOT / "scenes" / "csv_claude",
    "gemini": ROOT / "scenes" / "csv_gemini",
    "ds":     ROOT / "scenes" / "csv_ds",
    "qwen":   ROOT / "scenes" / "csv_qwen",
}

CATEGORIES = {"CMB", "INV", "PHY", "PRO"}


def human_judge(filename: str) -> int:
    """1 = regular ROLL file, 0 = NO_ROLL file (has _no suffix)."""
    stem = Path(filename).stem  # strip .csv
    return 0 if stem.endswith("_no") else 1


def collect(src_dir: Path) -> dict[str, list[dict]]:
    """Return {category: [rows]} for all CSV files in src_dir."""
    buckets: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    if not src_dir.exists():
        print(f"  [WARN] directory not found: {src_dir}")
        return buckets

    for csv_path in sorted(src_dir.glob("*.csv")):
        category = csv_path.name.split("_")[0]
        if category not in CATEGORIES:
            continue
        judge = human_judge(csv_path.name)
        with open(csv_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row["human_judge"] = judge
                buckets[category].append(row)

    return buckets


def write_output(model: str, category: str, rows: list[dict]):
    if not rows:
        print(f"  [SKIP] {model}_{category} — no data")
        return

    out_path = QUERY_DIR / f"{model}_{category}.csv"
    fieldnames = list(rows[0].keys())
    if "human_judge" not in fieldnames:
        fieldnames.append("human_judge")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {out_path.name:30s}  {len(rows)} rows")


def main():
    QUERY_DIR.mkdir(parents=True, exist_ok=True)

    for model, src_dir in SOURCES.items():
        print(f"\n[{model}] ← {src_dir.name}")
        buckets = collect(src_dir)
        for category in sorted(CATEGORIES):
            write_output(model, category, buckets[category])

    print("\nAll done.")


if __name__ == "__main__":
    main()
