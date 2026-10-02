#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

SYSTEM_PROMPT = """You are an expert Keeper (Game Master) for the Tabletop RPG "Call of Cthulhu". Task: Adjudicate a player's action based on the Scenario Truth (Context) and the Player's Description.

Rules for Adjudication:
1. Compare: Check the Player's Description against the Scenario Truth.
2. Decide: If the action carries risk, is difficult, or is opposed by the Scenario Truth, a Dice Roll is mandatory.
3. Ignore Fluff: Do not let the player's confidence (e.g., "I effortlessly do it") override the Scenario Truth (e.g., "The task is extremely hard").


Output: Provide a JSON object strictly.
- `needs_roll`: boolean (true = mechanics required, false = deterministic outcome).
- `outcome_if_no_roll`: string OR null.
- If `needs_roll` is true, set this to `null`.
- If `needs_roll` is false, describe what happens immediately (e.g., "You open the door." OR "It fails completely.").
- `reasoning`: string (Brief explanation).

Input:
1. Scenario Truth (Context): [concept_en]
2. Player Description: [L1/L2/L3]
"""

DEFAULT_BASE_URL = "https://llmapi.paratera.com"
DEFAULT_ENDPOINT = "/v1/chat/completions"
DEFAULT_TIMEOUT = 60


def load_items(path: str) -> List[Dict[str, Any]]:
    """Load aaa.json as JSON array or JSONL."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    # JSON array
    if content[0] == "[":
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("Input JSON must be a list when starts with '['.")
        return data

    # JSONL
    items = []
    for i, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSONL parse error at line {i}: {e}") from e
        if not isinstance(obj, dict):
            raise ValueError(f"JSONL line {i} is not an object/dict.")
        items.append(obj)
    return items


def iter_triplets(item: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    """Yield (variant, player_text) for L1/L2/L3 if present."""
    for k in ("L1", "L2", "L3"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            yield k, v.strip()

def strip_after_last_comma(concept_en: str) -> str:
    """
    Remove the last ',' and everything after it.
    Example: "a, b, c" -> "a, b"
    If no comma, return original.
    """
    s = (concept_en or "").strip()
    if "," not in s:
        return s
    return s.rsplit(",", 1)[0].rstrip()

def build_messages(concept_en: str, player_desc: str) -> List[Dict[str, str]]:
    user_content = f"Scenario Truth (Context): {concept_en}\nPlayer Description: {player_desc}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def try_parse_json_strict(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Best-effort parse for a JSON object from model output.
    Returns (parsed_obj, error_message).
    """
    text = text.strip()
    if not text:
        return None, "empty response"

    # First try direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, None
    except Exception:
        pass

    # Try to extract the first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = text[start : end + 1]
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj, None
        except Exception as e:
            return None, f"failed to parse extracted json block: {e}"

    return None, "response is not a JSON object"


def list_models(base_url: str, api_key: str) -> List[str]:
    url = base_url.rstrip("/") + "/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    r = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    # OpenAI-style: {"data":[{"id":"..."}, ...]}
    models = []
    for m in data.get("data", []):
        mid = m.get("id")
        if isinstance(mid, str):
            models.append(mid)
    return models


def post_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int],
    json_mode: bool,
    timeout: int,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + DEFAULT_ENDPOINT
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    # OpenAI-compatible JSON mode (some providers/models support it; if not, you can disable via --no-json-mode)
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def extract_text_from_response(resp: Dict[str, Any]) -> str:
    # OpenAI chat completions style
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(resp, ensure_ascii=False)


def load_done_keys(output_path: str) -> set:
    """
    For resume: output.jsonl each line includes {"key": "..."}.
    Return set of keys.
    """
    done = set()
    if not output_path or not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            k = obj.get("key")
            if isinstance(k, str):
                done.add(k)
    return done


def make_key(item_id: Any, variant: str) -> str:
    return f"{item_id}::{variant}"


def worker_task(
    base_url: str,
    api_key: str,
    model: str,
    item: Dict[str, Any],
    variant: str,
    player_desc: str,
    temperature: float,
    max_tokens: Optional[int],
    json_mode: bool,
    timeout: int,
    max_retries: int,
) -> Dict[str, Any]:
    item_id = item.get("id")
    concept_en_raw = item.get("concept_en", "")

    if not isinstance(concept_en_raw, str) or not concept_en_raw.strip():
        raise ValueError(f"Missing concept_en for id={item_id}")

    concept_en = strip_after_last_comma(concept_en_raw)

    messages = build_messages(concept_en, player_desc)

    last_err: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = post_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                timeout=timeout,
            )
            text = extract_text_from_response(resp)
            parsed, perr = try_parse_json_strict(text)

            return {
                "key": make_key(item_id, variant),
                "id": item_id,
                "variant": variant,
                "concept_en": concept_en,
                "player_description": player_desc,
                "raw_text": text,
                "parsed": parsed,
                "parse_error": perr,
                #"api_response": resp,  # 如果你觉得太大可删掉这行
            }
        except requests.HTTPError as e:
            # Retry on 429/5xx
            status = e.response.status_code if e.response is not None else None
            body = None
            try:
                body = e.response.text if e.response is not None else None
            except Exception:
                pass
            last_err = f"HTTPError status={status} body={body}"
            if status in (429, 500, 502, 503, 504):
                sleep_s = min(30.0, (2 ** (attempt - 1)) + random.random())
                time.sleep(sleep_s)
                continue
            raise
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = f"NetworkError: {e}"
            sleep_s = min(30.0, (2 ** (attempt - 1)) + random.random())
            time.sleep(sleep_s)
            continue
        except Exception as e:
            last_err = f"OtherError: {e}"
            # 对非网络错误一般不需要狂重试，但给一次机会
            if attempt < max_retries:
                time.sleep(0.5 + random.random())
                continue
            raise

    raise RuntimeError(f"Failed after retries. last_err={last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="aaa.json", help="Input aaa.json (JSON array or JSONL)")
    ap.add_argument("--output", default="output.jsonl", help="Output JSONL path")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base url")
    ap.add_argument("--model", default=os.environ.get("PARATERA_MODEL", ""), help="Model id (or set PARATERA_MODEL)")
    ap.add_argument("--max-workers", type=int, default=8, help="Thread workers")
    ap.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    ap.add_argument("--max-tokens", type=int, default=400, help="Max output tokens (set -1 to disable)")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request timeout (sec)")
    ap.add_argument("--max-retries", type=int, default=5, help="Retry count")
    ap.add_argument("--no-json-mode", action="store_true", help="Disable response_format=json_object")
    ap.add_argument("--list-models", action="store_true", help="List available models then exit")
    args = ap.parse_args()

    api_key = os.environ.get("PARATERA_API_KEY", "").strip()
    if not api_key:
        print("ERROR: Please set env PARATERA_API_KEY (do NOT hardcode in script).", file=sys.stderr)
        print("Example: export PARATERA_API_KEY='sk-xxxxxx'", file=sys.stderr)
        sys.exit(2)

    if args.list_models:
        models = list_models(args.base_url, api_key)
        print("\n".join(models))
        return

    model = args.model.strip()
    if not model:
        print("ERROR: --model is required (or set env PARATERA_MODEL).", file=sys.stderr)
        print("Tip: run: python script.py --list-models", file=sys.stderr)
        sys.exit(2)

    max_tokens = None if args.max_tokens == -1 else args.max_tokens
    json_mode = not args.no_json_mode

    items = load_items(args.input)
    if not items:
        print("No items loaded.", file=sys.stderr)
        return

    done = load_done_keys(args.output)

    # Build task list
    tasks: List[Tuple[Dict[str, Any], str, str]] = []
    for it in items:
        item_id = it.get("id")
        for variant, player_desc in iter_triplets(it):
            k = make_key(item_id, variant)
            if k in done:
                continue
            tasks.append((it, variant, player_desc))

    print(f"Loaded items={len(items)}; pending requests={len(tasks)}; resume done={len(done)}")
    if not tasks:
        print("Nothing to do.")
        return

    # Append mode for incremental write
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    lock_path = args.output + ".lock"

    def write_line(obj: Dict[str, Any]):
        # Simple single-process append (thread safe with a file lock via OS is complex;
        # we do serialized writes in main thread instead)
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = []
        for it, variant, player_desc in tasks:
            futures.append(
                ex.submit(
                    worker_task,
                    args.base_url,
                    api_key,
                    model,
                    it,
                    variant,
                    player_desc,
                    args.temperature,
                    max_tokens,
                    json_mode,
                    args.timeout,
                    args.max_retries,
                )
            )

        completed = 0
        for fut in cf.as_completed(futures):
            try:
                result = fut.result()
                write_line(result)
                completed += 1
                if completed % 10 == 0:
                    print(f"Completed {completed}/{len(futures)}")
            except Exception as e:
                err_obj = {"error": str(e)}
                write_line(err_obj)
                completed += 1
                print(f"[ERROR] {e}", file=sys.stderr)

    print("All done.")


if __name__ == "__main__":
    main()


"""
python query.py --input scenes/SOC_Charm.json --output result/SOC_Charm.jsonl --model DeepSeek-R1-0528 --max-workers 3

"""