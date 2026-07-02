#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import re
import time
from pathlib import Path

from openai import OpenAI

# =====================================================================
# TOP-LEVEL PARAMETERS
# =====================================================================
ROOT            = Path(__file__).parent.parent
LLM_CONFIG_PATH = ROOT / "config" / "llm_config.json"
SCENE_CONFIG_PATH = ROOT / "config" / "scene_config.json"
JSON_DIR        = ROOT / "scenes" / "json"
CSV_DIR         = ROOT / "scenes" / "csv"

# Model override: set to None to use the value from llm_config.json providers.gpt.dialogue
MODEL_OVERRIDE: str | None = None

# NO_JUDGE_MODE: if True, use scene_no_judge.txt and only process PHY + INV categories
NO_JUDGE_MODE: bool = False

PROMPT_PATH = ROOT / "prompt" / ("scene_no_judge.txt" if NO_JUDGE_MODE else "scene.txt")
NO_JUDGE_CATEGORIES = {"PHY", "INV"}

# RERUN_ONLY: if non-empty, only process files whose base name (without suffix/_no/extension)
# matches an entry. Example: ["PHY_Stealth_1920s_Urban", "PHY_Stealth_2020s_Urban"]
RERUN_ONLY: list[str] = [
    "PHY_Stealth_1920s_Urban",
    "PHY_Stealth_2020s_Urban",
]

MAX_RETRIES = 3

CSV_FIELDS = ["id", "group_id", "setting", "skill", "scenario_truth", "player", "type"]


# =====================================================================
# LOAD CONFIGS
# =====================================================================
def load_configs():
    llm_cfg   = json.loads(LLM_CONFIG_PATH.read_text(encoding="utf-8"))
    scene_cfg = json.loads(SCENE_CONFIG_PATH.read_text(encoding="utf-8"))

    gpt_cfg = llm_cfg["providers"]["gpt"]
    api_key = gpt_cfg["api_key"]
    model   = MODEL_OVERRIDE or gpt_cfg["dialogue"]

    categories    = scene_cfg["categories"]       # [{name, id, skills}, ...]
    target_settings = scene_cfg["target_settings"]  # ["1920s Urban", ...]

    return api_key, model, categories, target_settings


# =====================================================================
# PROMPT BUILDER
# =====================================================================
def build_prompt(template: str, category_name: str, skill: str, setting: str) -> str:
    prompt = template.replace("[INSERT CATEGORY]", category_name)
    prompt = prompt.replace("[INSERT SKILL]", skill)
    prompt = prompt.replace("[INSERT SETTING]", setting)
    return prompt


# =====================================================================
# API CALL
# =====================================================================
def call_gpt(client: OpenAI, model: str, prompt: str) -> str:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
            )
            return response.output_text
        except Exception as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            print(f"    [WARN] attempt {attempt} failed: {e}. Retrying in {wait}s ...")
            time.sleep(wait)
    raise RuntimeError(f"API failed after {MAX_RETRIES} retries: {last_err}")


# =====================================================================
# JSON EXTRACTION
# =====================================================================
def strip_comments(s: str) -> str:
    return re.sub(r'//[^\n"]*', '', s)


def extract_json(text: str):
    def try_parse(s):
        try:
            return json.loads(s.strip())
        except Exception:
            pass
        try:
            return json.loads(strip_comments(s).strip())
        except Exception:
            return None

    obj = try_parse(text)
    if obj is not None:
        return obj

    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        obj = try_parse(m.group(1))
        if obj is not None:
            return obj

    # Find outermost [ ... ] or { ... }
    for opener, closer in [('[', ']'), ('{', '}')]:
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    obj = try_parse(text[start:i + 1])
                    if obj is not None:
                        return obj
    return None


# =====================================================================
# SAVE JSON
# =====================================================================
def save_json(data, cat_id: str, skill: str, setting: str) -> Path:
    safe_skill   = skill.replace(" ", "_").replace(".", "")
    safe_setting = setting.replace(" ", "_")
    suffix = "_no" if NO_JUDGE_MODE else ""
    filename = f"{cat_id}_{safe_skill}_{safe_setting}{suffix}.json"
    path = JSON_DIR / filename
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# =====================================================================
# SAVE CSV
# =====================================================================
def save_csv(rows: list, cat_id: str, skill: str, setting: str) -> Path:
    safe_skill   = skill.replace(" ", "_").replace(".", "")
    safe_setting = setting.replace(" ", "_")
    suffix = "_no" if NO_JUDGE_MODE else ""
    filename = f"{cat_id}_{safe_skill}_{safe_setting}{suffix}.csv"
    path = CSV_DIR / filename
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
    return path


# =====================================================================
# MAIN
# =====================================================================
def main():
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    api_key, model, categories, target_settings = load_configs()
    template = PROMPT_PATH.read_text(encoding="utf-8")
    client = OpenAI(api_key=api_key)

    total = sum(len(cat["skills"]) for cat in categories) * len(target_settings)
    done  = 0

    print(f"Model   : {model}")
    print(f"Tasks   : {len(categories)} categories × skills × {len(target_settings)} settings = {total} requests\n")

    for cat in categories:
        cat_name = cat["name"]
        cat_id   = cat["id"]
        skills   = cat["skills"]

        if NO_JUDGE_MODE and cat_id not in NO_JUDGE_CATEGORIES:
            print(f"[SKIP] {cat_name} ({cat_id}) — no_judge mode only processes PHY + INV")
            continue

        for skill in skills:
            for setting in target_settings:
                done += 1
                safe_skill   = skill.replace(" ", "_").replace(".", "")
                safe_setting = setting.replace(" ", "_")
                base_name    = f"{cat_id}_{safe_skill}_{safe_setting}"
                if RERUN_ONLY and base_name not in RERUN_ONLY:
                    print(f"[SKIP] {base_name}")
                    continue

                tag = f"[{done}/{total}] {cat_name} / {skill} / {setting}"
                print(f"{tag} → calling API ...")

                prompt = build_prompt(template, cat_name, skill, setting)

                t0 = time.time()
                try:
                    raw = call_gpt(client, model, prompt)
                except RuntimeError as e:
                    print(f"  [ERROR] {e}")
                    continue
                elapsed = time.time() - t0

                data = extract_json(raw)
                if data is None:
                    print(f"  [ERROR] Could not parse JSON from response. Raw saved for inspection.")
                    # Save raw text for debugging
                    safe_skill   = skill.replace(" ", "_").replace(".", "")
                    safe_setting = setting.replace(" ", "_")
                    (JSON_DIR / f"{cat_id}_{safe_skill}_{safe_setting}.raw.txt").write_text(
                        raw, encoding="utf-8"
                    )
                    continue

                rows = data if isinstance(data, list) else [data]

                json_path = save_json(data, cat_id, skill, setting)
                csv_path  = save_csv(rows, cat_id, skill, setting)

                print(f"  OK  {len(rows)} rows  |  {elapsed:.1f}s  |  {json_path.name}  +  {csv_path.name}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
