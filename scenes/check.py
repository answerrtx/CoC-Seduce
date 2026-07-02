#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd


FIELDS = ["id", "category", "skill", "setting", "concept_zh", "concept_en", "L1", "L2", "L3"]


def norm_cell(v: Any) -> str:
    """Normalize cell to comparable string."""
    if v is None:
        return ""
    # pandas NaN
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass

    # numeric id etc.
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        # keep ints clean, floats like 1.0 -> 1
        if float(v).is_integer():
            return str(int(v))
        return str(v)

    s = str(v)
    # normalize newlines/spaces
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.strip()
    return s


def load_json_rows(json_path: Path) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON 顶层必须是 list（每个元素是一条记录 dict）。")
    rows = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"JSON 第 {i} 条不是 dict。")
        rows.append(item)
    return rows


def load_excel_rows(excel_path: Path, sheet: Optional[str]) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet, engine="openpyxl")
    # drop fully empty rows
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def build_col_map(df: pd.DataFrame) -> Dict[str, str]:
    """
    Map required field -> actual column name, case-insensitive & strip.
    """
    cols = list(df.columns)
    norm_to_col = {}
    for c in cols:
        key = str(c).strip().lower()
        norm_to_col[key] = c

    mapping = {}
    for f in FIELDS:
        key = f.lower()
        if key in norm_to_col:
            mapping[f] = norm_to_col[key]
    return mapping


def rows_to_keyed(
    json_rows: List[Dict[str, Any]],
    excel_df: pd.DataFrame,
    excel_col_map: Dict[str, str],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]], List[str]]:
    """
    Return keyed dicts: key -> {field->norm_str}, and keys in comparison order.
    Prefer key=id when possible; otherwise key=row_index.
    """
    json_has_id = all("id" in r for r in json_rows)
    excel_has_id = "id" in excel_col_map

    json_dict = {}
    excel_dict = {}
    keys_order = []

    if json_has_id and excel_has_id:
        # Key by id
        for r in json_rows:
            k = norm_cell(r.get("id"))
            json_dict[k] = {f: norm_cell(r.get(f, "")) for f in FIELDS}
        for _, row in excel_df.iterrows():
            k = norm_cell(row[excel_col_map["id"]])
            excel_dict[k] = {f: norm_cell(row[excel_col_map[f]]) if f in excel_col_map else "" for f in FIELDS}

        # union keys (keep stable order: json ids then extras)
        keys_order = list(dict.fromkeys(list(json_dict.keys()) + list(excel_dict.keys())))
        return json_dict, excel_dict, keys_order

    # Fallback: align by index
    n = max(len(json_rows), len(excel_df))
    for i in range(n):
        k = str(i)
        if i < len(json_rows):
            r = json_rows[i]
            json_dict[k] = {f: norm_cell(r.get(f, "")) for f in FIELDS}
        else:
            json_dict[k] = {f: "" for f in FIELDS}

        if i < len(excel_df):
            row = excel_df.iloc[i]
            excel_dict[k] = {f: norm_cell(row[excel_col_map[f]]) if f in excel_col_map else "" for f in FIELDS}
        else:
            excel_dict[k] = {f: "" for f in FIELDS}

        keys_order.append(k)

    return json_dict, excel_dict, keys_order


def compare(
    json_rows: List[Dict[str, Any]],
    excel_df: pd.DataFrame,
    sheet: Optional[str],
) -> Tuple[bool, pd.DataFrame]:
    col_map = build_col_map(excel_df)

    missing_cols = [f for f in FIELDS if f not in col_map]
    if missing_cols:
        print(f"[WARN] Excel 缺少列：{missing_cols}（这些列会按空字符串参与对比）", file=sys.stderr)

    json_dict, excel_dict, keys_order = rows_to_keyed(json_rows, excel_df, col_map)

    diffs = []
    for k in keys_order:
        jr = json_dict.get(k, {f: "" for f in FIELDS})
        er = excel_dict.get(k, {f: "" for f in FIELDS})

        for f in FIELDS:
            jv = jr.get(f, "")
            ev = er.get(f, "")
            if jv != ev:
                diffs.append(
                    {
                        "key": k,
                        "field": f,
                        "json_value": jv,
                        "excel_value": ev,
                    }
                )

    diff_df = pd.DataFrame(diffs, columns=["key", "field", "json_value", "excel_value"])
    ok = diff_df.empty
    return ok, diff_df


def main():
    ap = argparse.ArgumentParser(description="Compare JSON records with Excel rows by fields.")
    ap.add_argument("--json", required=True, help="Path to JSON file (list of dicts).")
    ap.add_argument("--excel", required=True, help="Path to Excel file (.xlsx/.xls).")
    ap.add_argument("--sheet", default=None, help="Excel sheet name (default: first sheet).")
    ap.add_argument("--out", default=None, help="Optional output diff CSV path.")
    args = ap.parse_args()

    json_path = Path(args.json)
    excel_path = Path(args.excel)

    if not json_path.exists():
        print(f"[ERR] JSON 文件不存在：{json_path}", file=sys.stderr)
        sys.exit(1)
    if not excel_path.exists():
        print(f"[ERR] Excel 文件不存在：{excel_path}", file=sys.stderr)
        sys.exit(1)

    try:
        json_rows = load_json_rows(json_path)
        excel_df = load_excel_rows(excel_path, args.sheet)
    except Exception as e:
        print(f"[ERR] 读取文件失败：{e}", file=sys.stderr)
        sys.exit(1)

    ok, diff_df = compare(json_rows, excel_df, args.sheet)

    # summary
    print(f"JSON records: {len(json_rows)}")
    print(f"Excel rows:   {len(excel_df)}")
    if ok:
        print("✅ 全部字段一致。")
        sys.exit(0)

    print(f"❌ 发现不一致项：{len(diff_df)}")
    # show first N diffs
    show_n = min(50, len(diff_df))
    print("\n--- Diff Preview (first {} rows) ---".format(show_n))
    for _, r in diff_df.head(show_n).iterrows():
        print(
            f"[key={r['key']}] field={r['field']}\n"
            f"  JSON : {r['json_value']}\n"
            f"  Excel: {r['excel_value']}\n"
        )

    if args.out:
        out_path = Path(args.out)
        try:
            diff_df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"已写出差异报告：{out_path}")
        except Exception as e:
            print(f"[WARN] 写出报告失败：{e}", file=sys.stderr)

    # mismatch exit code
    sys.exit(2)


if __name__ == "__main__":
    main()


"""
python check.py --json PRO_Medicine.json \
    --excel PRO_Medicine_translated.xlsx \
    --sheet Sheet1 --out diff_report.csv

"""