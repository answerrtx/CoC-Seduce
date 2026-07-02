
import csv
from pathlib import Path
from collections import defaultdict

data_dir = Path("results_summary")
GENERATORS = ["gpt", "claude", "gemini"]
JUDGE_MODELS = [
    "gpt-5.4-2026-03-05", "gpt-5-2025-08-07", "gpt-5-mini-2025-08-07", "gpt-4.1-2025-04-14",
    "claude-opus-4-6", "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-haiku-4-5-20251001",
    "gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3.5-flash",
    "deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v3.2", "deepseek-r1",
    "qwen3.7-max", "qwen3.6-flash", "qwen3-max",
    "qwen3-235b-a22b-thinking-2507", "qwen3-235b-a22b-instruct-2507",
]
CATEGORIES = ["CMB", "INV", "PHY", "PRO"]

results = []
for gen in GENERATORS:
    for judge in JUDGE_MODELS:
        fp_fail, fp_total = 0, 0
        for cat in CATEGORIES:
            f = data_dir / f"{gen}_{cat}_{judge}.csv"
            if not f.exists():
                continue
            with open(f, encoding="utf-8", newline="") as fp:
                for row in csv.DictReader(fp):
                    hj = int(row.get("human_judge", 0))
                    nr = row.get("needs_roll", "").strip().lower()
                    pred = True if nr in ("true","1") else False if nr in ("false","0") else None
                    if pred is None or hj != 1:
                        continue
                    fp_total += 1
                    if pred is False:
                        fp_fail += 1
        fr = round(fp_fail / fp_total * 100, 2) if fp_total else float("nan")
        results.append({"generator": gen, "judge": judge, "FR": fr, "total": fp_total})

with open("generator_target_fr.csv", "w", newline="") as f:
    import csv as csv2
    w = csv2.DictWriter(f, fieldnames=["generator","judge","FR","total"])
    w.writeheader()
    w.writerows(results)
