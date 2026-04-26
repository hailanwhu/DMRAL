import pandas as pd
import glob
import tqdm
from tqdm import tqdm
from collections import defaultdict
import re
import json
import argparse
from uuid import uuid4
import utils
from utils import prompt, model_config, tool
from typing import List, Dict, Any, Tuple, Optional




recover_column_headers = recover_column_headers_prompt | llm


##########################################
#               FUNCTIONS
##########################################
def run_structured_prompt(title_name, headers, sampled_rows):
    response = recover_column_headers.invoke({"title_name": title_name, "headers":headers, "sampled_rows": sampled_rows})
    return response.content.strip()


def remove_trailing_number(s):
    # Match ending digits and remove them if found
    return re.sub(r'\d+$', '', s)

def serialize_schema(table_name, schema):
    return f"Table: {table_name}\nColumns: {schema}"


def _is_bad_header(col: str) -> bool:
    """
    Heuristics for "missing / useless" header names.
    """
    if col is None:
        return True
    c = str(col).strip()
    if c == "":
        return True
    # pandas sometimes produces "Unnamed: 0"
    if c.lower().startswith("unnamed"):
        return True
    # generic placeholders often produced by earlier pipelines
    if re.match(r"^column\s*\d+$", c.lower()):
        return True
    if re.match(r"^col\s*\d+$", c.lower()):
        return True
    return False


def _needs_recover(headers: List[str]) -> bool:
    """
    Decide whether this table likely has missing/placeholder headers.
    """
    if not headers:
        return True
    bad = sum(1 for h in headers if _is_bad_header(h))
    # If many are bad, recover
    if bad / max(1, len(headers)) >= 0.4:
        return True
    # If almost all are purely numeric strings, recover
    numeric_like = 0
    for h in headers:
        s = str(h).strip()
        if re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
            numeric_like += 1
    if numeric_like / max(1, len(headers)) >= 0.6:
        return True
    return False


def _sample_rows_as_text(df: pd.DataFrame, n: int = 5) -> str:
    """
    Provide the model some representative rows (string).
    Keep it compact.
    """
    if df is None or df.empty:
        return ""
    smp = df.head(n).copy()
    # avoid super-wide text
    if smp.shape[1] > 30:
        smp = smp.iloc[:, :30]
    return smp.to_csv(index=False)


def _parse_llm_headers(text: str, expected_n: int) -> Optional[List[str]]:
    """
    Parse model output into a list of headers.
    Accept:
      - JSON list: ["a","b",...]
      - JSON dict with "columns": [...]
      - newline-separated or comma-separated text
    """
    if not text:
        return None

    # Strip code fences if any
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s).strip()

    # Try JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            cols = [str(x).strip() for x in obj]
        elif isinstance(obj, dict):
            if "columns" in obj and isinstance(obj["columns"], list):
                cols = [str(x).strip() for x in obj["columns"]]
            elif "headers" in obj and isinstance(obj["headers"], list):
                cols = [str(x).strip() for x in obj["headers"]]
            else:
                cols = None
        else:
            cols = None

        if cols:
            cols = [c if c else f"Column_{i+1}" for i, c in enumerate(cols)]
            if len(cols) == expected_n:
                return cols
            # If off by small mismatch, try to fix gently (truncate/pad)
            if len(cols) > expected_n:
                return cols[:expected_n]
            if len(cols) < expected_n:
                cols = cols + [f"Column_{i+1}" for i in range(len(cols), expected_n)]
                return cols
    except Exception:
        pass

    # Fallback: split by newline or comma
    # Prefer newline splitting first
    if "\n" in s:
        parts = [p.strip(" -\t") for p in s.splitlines() if p.strip()]
    else:
        parts = [p.strip() for p in s.split(",") if p.strip()]

    if not parts:
        return None

    # If it looks like "1) xxx" -> remove leading numbering
    cleaned = []
    for p in parts:
        p2 = re.sub(r"^\s*\d+\s*[\).\:-]\s*", "", p).strip()
        cleaned.append(p2 if p2 else p)

    cleaned = [c if c else f"Column_{i+1}" for i, c in enumerate(cleaned)]

    if len(cleaned) == expected_n:
        return cleaned
    if len(cleaned) > expected_n:
        return cleaned[:expected_n]
    if len(cleaned) < expected_n:
        cleaned = cleaned + [f"Column_{i+1}" for i in range(len(cleaned), expected_n)]
        return cleaned

    return None


def _safe_unique_headers(cols: List[str]) -> List[str]:
    """
    Ensure headers are unique (pandas requires unique-ish for many pipelines).
    """
    seen = {}
    out = []
    for c in cols:
        base = (c or "").strip()
        if base == "":
            base = "Column"
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out


def process_dataset(dataset_name: str):
    # Input tables
    in_dir = os.path.abspath(os.path.join("../../dataset/datalake", dataset_name))
    fls = sorted(glob.glob(os.path.join(in_dir, "*.csv")))

    if not fls:
        raise FileNotFoundError(f"No CSV found under: {in_dir}")

    # Output dirs
    out_root = os.path.abspath(os.path.join("../../dataset/datalake", dataset_name, "recovered"))
    out_csv_dir = os.path.join(out_root, "csv")
    out_meta_dir = os.path.join(out_root, "meta")
    os.makedirs(out_csv_dir, exist_ok=True)
    os.makedirs(out_meta_dir, exist_ok=True)

    stats = {
        "dataset": dataset_name,
        "n_tables": len(fls),
        "n_recovered": 0,
        "n_skipped_ok": 0,
        "n_failed": 0,
        "failures": [],
    }

    for csv_path in tqdm(fls, desc=f"Recovering headers for {dataset_name}"):
        fname = os.path.basename(csv_path)
        table_id = os.path.splitext(fname)[0]
        title_name = remove_trailing_number(table_id)

        try:
            # Read with header=0 (most csvs have header row)
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
            headers = list(df.columns)

            if not _needs_recover(headers):
                stats["n_skipped_ok"] += 1
                # Still write metadata to keep consistent outputs
                meta = {
                    "table_id": table_id,
                    "source_csv": csv_path,
                    "recovered": False,
                    "original_headers": headers,
                    "final_headers": headers,
                    "method": "heuristic_skip",
                }
                with open(os.path.join(out_meta_dir, f"{table_id}.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                continue

            sampled_rows = _sample_rows_as_text(df, n=5)
            llm_out = run_structured_prompt(title_name, headers, sampled_rows)

            new_headers = _parse_llm_headers(llm_out, expected_n=len(headers))
            if not new_headers:
                # fallback: fill bad headers only, keep good ones
                new_headers = []
                for i, h in enumerate(headers):
                    if _is_bad_header(h):
                        new_headers.append(f"Column_{i+1}")
                    else:
                        new_headers.append(str(h).strip())
                new_headers = _safe_unique_headers(new_headers)

            # Apply + persist
            new_headers = _safe_unique_headers([str(x).strip() for x in new_headers])

            df2 = df.copy()
            df2.columns = new_headers

            out_csv_path = os.path.join(out_csv_dir, fname)
            df2.to_csv(out_csv_path, index=False)

            meta = {
                "table_id": table_id,
                "source_csv": csv_path,
                "recovered": True,
                "title_name": title_name,
                "original_headers": headers,
                "final_headers": new_headers,
                "sampled_rows_csv": sampled_rows,
                "llm_raw_output": llm_out,
                "out_csv": out_csv_path,
                "method": "llm_recover_column_headers",
                "run_id": str(uuid4()),
            }
            with open(os.path.join(out_meta_dir, f"{table_id}.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            stats["n_recovered"] += 1

        except Exception as e:
            stats["n_failed"] += 1
            stats["failures"].append({"table_id": table_id, "path": csv_path, "error": repr(e)})

    # Write dataset-level summary
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)



def main():
    parser = argparse.ArgumentParser(description="Process table files and recover metadata for a given dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., 'spider')")
    args = parser.parse_args()
    process_dataset(args.dataset)


if __name__ == "__main__":
    main()