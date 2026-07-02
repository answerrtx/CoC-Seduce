#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai_judge.py — Query judge-tier LLMs for each row in scenes/query/*.csv.

Usage examples:
  # Run all GPT judge models on all query files
  python scripts/ai_judge.py --providers gpt

  # Run GPT + Claude judges on a specific file
  python scripts/ai_judge.py --providers gpt claude --files claude_CMB

  # Run all providers on selected files, 2 row workers per model
  python scripts/ai_judge.py --files claude_CMB gpt_INV --workers 2

Output:
  results/{stem}_{model}_{group_id}_{id}.json  — per-row JSON
  results/{stem}_{model}.csv                   — final aggregated CSV
  query_log/{ts}_{stem}_{model}.log            — per-task log
"""

import argparse
import csv
import json
import logging
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# =====================================================================
# PATHS
# =====================================================================
ROOT            = Path(__file__).parent.parent
LLM_CONFIG_PATH = ROOT / "config" / "llm_config.json"
QUERY_DIR       = ROOT / "scenes" / "query"
RESULTS_DIR     = ROOT / "results"
LOG_DIR         = ROOT / "query_log"
PROMPT_PATH     = ROOT / "prompt" / "ai_judge.txt"

MAX_RETRIES = 3
MAX_TOKENS  = 1024

# Base URLs for OpenAI-compatible providers
COMPAT_BASE_URLS: dict[str, str] = {
    "deepseek": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "qwen":     "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

OUTPUT_CSV_FIELDS = [
    "id", "group_id", "type", "human_judge", "skill",
    "needs_roll", "predicted_skill", "reasoning",
]


# =====================================================================
# RATE LIMITER
# =====================================================================

class RateLimiter:
    """Sliding-window rate limiter: at most `rpm` calls per 60 seconds.

    rpm=0 disables limiting entirely.  Thread-safe.
    """

    def __init__(self, rpm: int) -> None:
        self.rpm = rpm
        self._lock = threading.Lock()
        self._window: deque[float] = deque()

    def acquire(self) -> None:
        if self.rpm <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                # Drop timestamps that have aged out of the 60-second window
                while self._window and now - self._window[0] >= 60.0:
                    self._window.popleft()
                if len(self._window) < self.rpm:
                    self._window.append(now)
                    return
                # Oldest entry will expire at this time; wait until then
                wait_s = self._window[0] + 60.0 - now + 0.05
            time.sleep(max(wait_s, 0.05))


# =====================================================================
# UTILITIES
# =====================================================================

def safe_model_name(model: str) -> str:
    return re.sub(r"[^\w\-.]", "_", model)


def strip_inline_comments(s: str) -> str:
    return re.sub(r"//[^\n\"]*", "", s)


def extract_json(text: str) -> Optional[Any]:
    def try_parse(s: str) -> Optional[Any]:
        try:
            return json.loads(s.strip())
        except Exception:
            pass
        try:
            return json.loads(strip_inline_comments(s).strip())
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

    # Walk outermost { ... }
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : i + 1]
                    obj = try_parse(chunk)
                    if obj is not None:
                        return obj
                    # Fix unquoted string values: "key": unquoted text
                    # \s+ (not \s*) prevents backtracking to zero, so lookahead
                    # reliably sees the first value character, not whitespace.
                    fixed = re.sub(
                        r'"(\w+)"\s*:\s+(?!(?:true|false|null)\b|[\"\{\[\d\-])([^\n,\}]+)',
                        lambda m: f'"{m.group(1)}": "{m.group(2).strip()}"',
                        chunk,
                    )
                    obj = try_parse(fixed)
                    if obj is not None:
                        return obj
    return None


def build_prompt(template: str, row: dict) -> str:
    p = template
    p = p.replace("[scenario_truth]", row.get("scenario_truth", ""))
    p = p.replace("[player]",         row.get("player", ""))
    p = p.replace("[setting]",        row.get("setting", ""))
    return p


# =====================================================================
# LOGGING
# =====================================================================

def make_logger(name: str, log_path: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(name)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# =====================================================================
# API CALLERS  (mirror patterns from gen_scene_*.py)
# =====================================================================

def call_gpt(api_key: str, model: str, prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(model=model, input=prompt)
            return resp.output_text
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"GPT API failed after {MAX_RETRIES} retries: {last_err}")


def call_claude(api_key: str, model: str, prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            if not msg.content:
                reason = msg.stop_reason
                if reason == "refusal":
                    raise RuntimeError(f"REFUSAL (stop_reason=refusal) — skipping retries")
                raise ValueError(f"Empty content (stop_reason={reason})")
            return msg.content[0].text
        except RuntimeError as e:
            if "REFUSAL" in str(e):
                raise  # don't retry on safety refusals
            last_err = e
            time.sleep(min(2 ** attempt, 30))
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Claude API failed after {MAX_RETRIES} retries: {last_err}")


def call_gemini(api_key: str, model: str, prompt: str) -> str:
    from google import genai
    client = genai.Client(api_key=api_key)
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            return resp.text
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Gemini API failed after {MAX_RETRIES} retries: {last_err}")


def call_compat(api_key: str, model: str, prompt: str, base_url: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return completion.choices[0].message.content
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"OpenAI-compat API failed after {MAX_RETRIES} retries: {last_err}")


def call_api(provider: str, api_key: str, model: str, prompt: str) -> str:
    if provider == "gpt":
        return call_gpt(api_key, model, prompt)
    if provider == "claude":
        return call_claude(api_key, model, prompt)
    if provider == "gemini":
        return call_gemini(api_key, model, prompt)
    if provider in COMPAT_BASE_URLS:
        return call_compat(api_key, model, prompt, COMPAT_BASE_URLS[provider])
    raise ValueError(f"Unknown provider: {provider!r}")


# =====================================================================
# ROW WORKER
# =====================================================================

def row_json_path(
    query_stem: str, safe_m: str, group_id: str, row_id: str, human_judge: str
) -> Path:
    return RESULTS_DIR / f"{query_stem}_{safe_m}_{group_id}_{row_id}_{human_judge}.json"


def row_json_path_legacy(
    query_stem: str, safe_m: str, group_id: str, row_id: str
) -> Path:
    """Old naming without human_judge suffix — used as fallback for human_judge=1."""
    return RESULTS_DIR / f"{query_stem}_{safe_m}_{group_id}_{row_id}.json"


def process_row(
    provider: str,
    api_key: str,
    model: str,
    template: str,
    row: dict,
    query_stem: str,
    logger: logging.Logger,
    limiter: RateLimiter,
) -> tuple[bool, Optional[dict]]:
    row_id      = str(row.get("id", "")).strip()
    group_id    = str(row.get("group_id", "")).strip()
    human_judge = str(row.get("human_judge", "")).strip()
    safe_m      = safe_model_name(model)

    json_file = row_json_path(query_stem, safe_m, group_id, row_id, human_judge)

    # Resume: check new-style filename first
    if json_file.exists():
        try:
            cached = json.loads(json_file.read_text(encoding="utf-8"))
            logger.debug(f"[CACHED] id={row_id} hj={human_judge}")
            return True, cached
        except Exception:
            pass  # corrupt — re-query

    # Backward compat: for human_judge=1, also accept old filename (no suffix)
    if human_judge == "1":
        legacy = row_json_path_legacy(query_stem, safe_m, group_id, row_id)
        if legacy.exists():
            try:
                cached = json.loads(legacy.read_text(encoding="utf-8"))
                logger.debug(f"[CACHED-legacy] id={row_id}")
                return True, cached
            except Exception:
                pass

    prompt = build_prompt(template, row)
    limiter.acquire()
    t0 = time.time()
    try:
        raw = call_api(provider, api_key, model, prompt)
    except RuntimeError as e:
        logger.error(f"[FAIL] id={row_id} group={group_id} hj={human_judge}: {e}")
        return False, None
    elapsed = time.time() - t0

    data = extract_json(raw)
    if data is None:
        logger.error(f"[PARSE_FAIL] id={row_id} group={group_id} hj={human_judge} raw={raw[:200]!r}")
        return False, None

    result = {
        "id":              row_id,
        "group_id":        group_id,
        "needs_roll":      data.get("needs_roll"),
        "predicted_skill": data.get("predicted_skill"),
        "reasoning":       data.get("reasoning"),
        "elapsed_s":       round(elapsed, 2),
    }
    json_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.debug(
        f"[OK] id={row_id} hj={human_judge} needs_roll={result['needs_roll']} "
        f"skill={result['predicted_skill']!r} ({elapsed:.1f}s)"
    )
    return True, result


# =====================================================================
# MODEL × FILE TASK
# =====================================================================

def process_file_model(
    provider: str,
    api_key: str,
    model: str,
    csv_path: Path,
    template: str,
    row_workers: int,
    logger: logging.Logger,
    limiter: RateLimiter,
) -> None:
    query_stem = csv_path.stem
    safe_m     = safe_model_name(model)

    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    logger.info(f"START  {query_stem} × {model}  ({total} rows, {row_workers} workers)")

    # results keyed by (group_id, row_id, human_judge) — unique even for duplicate ids
    results_map: dict[tuple, dict] = {}
    pending: list[dict] = []

    for row in rows:
        row_id      = str(row.get("id", "")).strip()
        group_id    = str(row.get("group_id", "")).strip()
        human_judge = str(row.get("human_judge", "")).strip()
        key         = (group_id, row_id, human_judge)

        # Check new-style file
        jf = row_json_path(query_stem, safe_m, group_id, row_id, human_judge)
        if jf.exists():
            try:
                results_map[key] = json.loads(jf.read_text(encoding="utf-8"))
                continue
            except Exception:
                pass

        # Backward compat fallback for human_judge=1
        if human_judge == "1":
            legacy = row_json_path_legacy(query_stem, safe_m, group_id, row_id)
            if legacy.exists():
                try:
                    results_map[key] = json.loads(legacy.read_text(encoding="utf-8"))
                    continue
                except Exception:
                    pass

        pending.append(row)

    cached_n = total - len(pending)
    if cached_n:
        logger.info(f"  Cached {cached_n}, pending {len(pending)}")

    if pending:
        done_n = 0
        with ThreadPoolExecutor(max_workers=row_workers) as ex:
            futures = {
                ex.submit(
                    process_row,
                    provider, api_key, model, template, row, query_stem, logger, limiter,
                ): row
                for row in pending
            }
            for fut in as_completed(futures):
                row    = futures[fut]
                done_n += 1
                try:
                    ok, result = fut.result()
                    if ok and result:
                        key = (
                            str(row.get("group_id", "")).strip(),
                            str(row.get("id", "")).strip(),
                            str(row.get("human_judge", "")).strip(),
                        )
                        results_map[key] = result
                except Exception as e:
                    logger.error(f"[WORKER_ERR] {e}")
                if done_n % 20 == 0 or done_n == len(pending):
                    logger.info(f"  Progress {done_n}/{len(pending)} pending rows")

    # ── Write final CSV ─────────────────────────────────────────────
    out_csv = RESULTS_DIR / f"{query_stem}_{safe_m}.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row_id      = str(row.get("id", "")).strip()
            group_id    = str(row.get("group_id", "")).strip()
            human_judge = str(row.get("human_judge", "")).strip()
            key         = (group_id, row_id, human_judge)
            r           = results_map.get(key, {})
            writer.writerow({
                "id":              row_id,
                "group_id":        group_id,
                "type":            row.get("type", ""),
                "human_judge":     human_judge,
                "skill":           row.get("skill", ""),
                "needs_roll":      r.get("needs_roll", ""),
                "predicted_skill": r.get("predicted_skill", ""),
                "reasoning":       r.get("reasoning", ""),
            })

    ok_n = len(results_map)
    logger.info(f"CSV    {out_csv.name}  ({ok_n}/{total} rows OK)")
    if ok_n < total:
        logger.warning(f"  {total - ok_n} rows missing — check log for errors")


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Query LLM judge models for every row in scenes/query/*.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--providers", nargs="+", metavar="PROVIDER",
        help="Provider names: gpt, claude, gemini, deepseek, qwen. Default: all",
    )
    ap.add_argument(
        "--files", nargs="+", metavar="STEM",
        help="Query CSV stems, e.g. claude_CMB gpt_INV. Default: all files in scenes/query/",
    )
    ap.add_argument(
        "--workers", type=int, default=4,
        help="Concurrent row workers per model (default: 4)",
    )
    ap.add_argument(
        "--model-workers", type=int, default=0,
        help="Max concurrent model tasks (default: all models in parallel)",
    )
    ap.add_argument(
        "--rpm", type=int, default=60,
        help="Max requests per minute per model (default: 60, 0 = unlimited)",
    )
    ap.add_argument(
        "--aggregate-only", action="store_true",
        help="Skip API calls; just read existing JSONs and write CSVs",
    )
    ap.add_argument(
        "--output-dir", metavar="DIR", default=None,
        help="Directory for output CSVs (default: results/). Created if absent.",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    llm_cfg  = json.loads(LLM_CONFIG_PATH.read_text(encoding="utf-8"))
    template = PROMPT_PATH.read_text(encoding="utf-8")

    # ── Resolve providers ───────────────────────────────────────────
    all_providers = list(llm_cfg["providers"].keys())
    sel_providers = args.providers or all_providers
    bad = [p for p in sel_providers if p not in llm_cfg["providers"]]
    if bad:
        print(f"ERROR: Unknown providers {bad}. Available: {all_providers}")
        return

    judge_tuples: list[tuple[str, str, str]] = []  # (provider, api_key, model)
    for pname in sel_providers:
        pcfg = llm_cfg["providers"][pname]
        for m in pcfg.get("judge", []):
            judge_tuples.append((pname, pcfg["api_key"], m))

    if not judge_tuples:
        print("No judge models found for the selected providers.")
        return

    # ── Resolve CSV files ───────────────────────────────────────────
    if args.files:
        csv_paths: list[Path] = []
        for stem in args.files:
            p = QUERY_DIR / f"{stem}.csv"
            if p.exists():
                csv_paths.append(p)
            else:
                print(f"WARNING: {p} not found — skipping")
    else:
        csv_paths = sorted(QUERY_DIR.glob("*.csv"))

    if not csv_paths:
        print("No query CSV files found.")
        return

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_tasks = len(judge_tuples) * len(csv_paths)

    print(f"Providers   : {sel_providers}")
    print(f"Models      : {[m for _, _, m in judge_tuples]}")
    print(f"Files       : {[p.stem for p in csv_paths]}")
    print(f"Tasks       : {n_tasks}  ({len(judge_tuples)} models × {len(csv_paths)} files)")

    # ── Aggregate-only mode: read JSONs → write CSVs, no API calls ──
    if args.aggregate_only:
        print("Mode        : aggregate-only (no API calls)\n")
        done = 0
        for _, _, model in judge_tuples:
            safe_m = safe_model_name(model)
            for csv_path in csv_paths:
                done += 1
                query_stem = csv_path.stem
                with open(csv_path, encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

                results_map: dict[tuple, dict] = {}
                for row in rows:
                    row_id      = str(row.get("id", "")).strip()
                    group_id    = str(row.get("group_id", "")).strip()
                    human_judge = str(row.get("human_judge", "")).strip()
                    key         = (group_id, row_id, human_judge)
                    jf = row_json_path(query_stem, safe_m, group_id, row_id, human_judge)
                    if jf.exists():
                        try:
                            results_map[key] = json.loads(jf.read_text(encoding="utf-8"))
                            continue
                        except Exception:
                            pass
                    if human_judge == "1":
                        legacy = row_json_path_legacy(query_stem, safe_m, group_id, row_id)
                        if legacy.exists():
                            try:
                                results_map[key] = json.loads(legacy.read_text(encoding="utf-8"))
                                continue
                            except Exception:
                                pass

                out_csv = out_dir / f"{query_stem}_{safe_m}.csv"
                missing_n = 0
                with open(out_csv, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=OUTPUT_CSV_FIELDS, extrasaction="ignore")
                    writer.writeheader()
                    for row in rows:
                        row_id      = str(row.get("id", "")).strip()
                        group_id    = str(row.get("group_id", "")).strip()
                        human_judge = str(row.get("human_judge", "")).strip()
                        r           = results_map.get((group_id, row_id, human_judge))
                        if r is None:
                            missing_n += 1
                            writer.writerow({
                                "id":              row_id,
                                "group_id":        group_id,
                                "type":            row.get("type", ""),
                                "human_judge":     human_judge,
                                "skill":           row.get("skill", ""),
                                "needs_roll":      "Refusal",
                                "predicted_skill": "Refusal",
                                "reasoning":       "Refusal",
                            })
                        else:
                            writer.writerow({
                                "id":              row_id,
                                "group_id":        group_id,
                                "type":            row.get("type", ""),
                                "human_judge":     human_judge,
                                "skill":           row.get("skill", ""),
                                "needs_roll":      r.get("needs_roll", ""),
                                "predicted_skill": r.get("predicted_skill", ""),
                                "reasoning":       r.get("reasoning", ""),
                            })
                ok_n = len(results_map)
                tag  = f"  [{missing_n} Refusal]" if missing_n else ""
                print(f"[{done:>3}/{n_tasks}] {query_stem} × {model}  ({ok_n}/{len(rows)} rows){tag}")

        print(f"\nDone. CSVs written to {out_dir}/")
        return

    # ── Normal mode: query API then write CSVs ───────────────────────
    print(f"Row workers : {args.workers} per model")
    print(f"Rate limit  : {args.rpm} RPM per model" if args.rpm > 0 else "Rate limit  : unlimited")
    print()

    limiters: dict[str, RateLimiter] = {
        model: RateLimiter(args.rpm) for _, _, model in judge_tuples
    }

    tasks: list[tuple[str, str, str, Path]] = [
        (pname, api_key, model, csv_path)
        for (pname, api_key, model) in judge_tuples
        for csv_path in csv_paths
    ]

    model_concurrency = args.model_workers or len(judge_tuples)

    def run_task(task: tuple[str, str, str, Path]) -> None:
        pname, api_key, model, csv_path = task
        safe_m      = safe_model_name(model)
        log_path    = LOG_DIR / f"{ts}_{csv_path.stem}_{safe_m}.log"
        logger_name = f"{pname}.{safe_m}.{csv_path.stem}.{ts}"
        logger      = make_logger(logger_name, log_path)
        limiter     = limiters[model]
        if limiter.rpm > 0:
            logger.info(f"Rate limit: {limiter.rpm} RPM")
        try:
            process_file_model(
                pname, api_key, model, csv_path, template, args.workers, logger, limiter,
            )
        except Exception as e:
            logger.error(f"Fatal error in task: {e}", exc_info=True)
        logger.info(f"DONE   {model} × {csv_path.stem}")

    with ThreadPoolExecutor(max_workers=model_concurrency) as pool:
        future_map = {
            pool.submit(run_task, t): (t[2], t[3].stem) for t in tasks
        }
        done = 0
        for fut in as_completed(future_map):
            model, stem = future_map[fut]
            done += 1
            try:
                fut.result()
                print(f"[{done:>3}/{n_tasks}] DONE  {model} × {stem}")
            except Exception as e:
                print(f"[{done:>3}/{n_tasks}] ERROR {model} × {stem}: {e}")

    print(f"\nAll {n_tasks} tasks complete.")
    print(f"Results : {RESULTS_DIR}/")
    print(f"Logs    : {LOG_DIR}/")


if __name__ == "__main__":
    main()


# # 只跑 GPT 的 judge 模型，查 claude_CMB.csv
# python scripts/ai_judge.py --providers gpt --files claude_CMB

# # 跑 GPT 全部 4 个 judge 模型并发，查所有文件
# python scripts/ai_judge.py --providers gpt

# # GPT + Claude，指定文件，每模型 8 个行并发
# python scripts/ai_judge.py --providers gpt --files gpt_CMB ds_PHY --workers 8 
# python scripts/ai_judge.py --providers gpt --files gpt_CMB gpt_INV gpt_PHY gpt_PRO --workers 8 --rpm 30

#python scripts/ai_judge.py --providers gemini --files gpt_CMB gpt_INV gpt_PHY gpt_PRO --workers 8 --rpm 30


# # 只汇总，不调 API，对所有 provider 和所有文件
# python scripts/ai_judge.py --aggregate-only

# # 只汇总指定 provider 和文件
# python scripts/ai_judge.py --providers gpt claude --files gpt_CMB gpt_INV --aggregate-only