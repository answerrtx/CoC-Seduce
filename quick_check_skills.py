# quick_check_skills.py
import csv
from pathlib import Path
from collections import Counter

data_dir = Path("results_summary")
gt_skills = Counter()
pred_skills = Counter()
mismatches = []

for f in data_dir.glob("*.csv"):
    with open(f, encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if row.get("human_judge", "").strip() == "1":
                nr = row.get("needs_roll", "").strip().lower()
                if nr == "true":
                    gt  = row.get("skill", "").strip()
                    pred = row.get("predicted_skill", "").strip()
                    gt_skills[gt] += 1
                    pred_skills[pred] += 1
                    if gt.lower() != pred.lower():
                        mismatches.append((gt, pred))

print("=== Ground Truth Skills ===")
for k, v in sorted(gt_skills.items()):
    print(f"  {v:>6}  '{k}'")

print("\n=== Predicted Skills ===")
for k, v in sorted(pred_skills.items()):
    print(f"  {v:>6}  '{k}'")

print(f"\n=== Sample Mismatches (first 20) ===")
for gt, pred in mismatches[:20]:
    print(f"  GT: '{gt}'  →  PRED: '{pred}'")