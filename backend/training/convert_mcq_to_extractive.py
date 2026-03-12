import argparse
import csv
import json
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _iter_files(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        path = raw if raw.is_absolute() else raw.resolve()
        if not path.exists():
            alt = (ROOT / raw).resolve()
            path = alt if alt.exists() else path
        if path.is_file():
            out.append(path)
            continue
        if not path.exists():
            continue
        for ext in ("*.csv", "*.jsonl", "*.json"):
            out.extend(path.rglob(ext))
    return sorted(set(out))


def _read_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
            return [row for row in csv.DictReader(f) if isinstance(row, dict)]

    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
        return rows

    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return []
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ("data", "examples", "items", "records"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, dict)]
        return []
    return []


def _extract_options_from_dict(row: dict[str, Any]) -> dict[str, str]:
    options: dict[str, str] = {}
    options_value = row.get("options") or row.get("choices")
    if isinstance(options_value, dict):
        for k, v in options_value.items():
            label = _safe_text(k).upper()[:1]
            text = _normalize_ws(_safe_text(v))
            if label in LABELS and text:
                options[label] = text
        if options:
            return options

    if isinstance(options_value, list):
        for idx, item in enumerate(options_value):
            label = LABELS[idx] if idx < len(LABELS) else ""
            if not label:
                break
            if isinstance(item, dict):
                text = _normalize_ws(
                    _safe_text(item.get("text") or item.get("option") or item.get("content"))
                )
            else:
                text = _normalize_ws(_safe_text(item))
            if text:
                options[label] = text
        if options:
            return options

    for label in LABELS[:8]:
        key_variants = (label, label.lower(), f"option_{label.lower()}", f"choice_{label.lower()}")
        text = ""
        for key in key_variants:
            value = row.get(key)
            if _safe_text(value):
                text = _normalize_ws(_safe_text(value))
                break
        if text:
            options[label] = text
    return options


def _extract_question(row: dict[str, Any]) -> str:
    for key in ("question", "Question", "query", "Query", "prompt", "Prompt", "stem", "Stem"):
        value = _safe_text(row.get(key))
        if value:
            return _normalize_ws(value)
    return ""


def _resolve_answer_label(answer_raw: str, options: dict[str, str]) -> str:
    answer = answer_raw.strip()
    if not answer:
        return ""
    upper = answer.upper()
    if upper in options:
        return upper
    if len(upper) == 1 and upper in LABELS:
        return upper if upper in options else ""
    if answer.isdigit():
        idx = int(answer)
        if 0 <= idx < len(options):
            return list(sorted(options.keys()))[idx]
        if 1 <= idx <= len(options):
            return list(sorted(options.keys()))[idx - 1]
    norm = _normalize_ws(answer).lower()
    for label, text in options.items():
        if _normalize_ws(text).lower() == norm:
            return label
    return ""


def _extract_answer_text(row: dict[str, Any], options: dict[str, str]) -> str:
    raw_candidates = [
        _safe_text(row.get("answer")),
        _safe_text(row.get("Answer")),
        _safe_text(row.get("label")),
        _safe_text(row.get("Label")),
        _safe_text(row.get("gold")),
        _safe_text(row.get("Gold")),
        _safe_text(row.get("target")),
        _safe_text(row.get("Target")),
    ]
    for raw in raw_candidates:
        label = _resolve_answer_label(raw, options)
        if label and label in options:
            return options[label]
        if raw and raw in options.values():
            return _normalize_ws(raw)
    return ""


def _to_extractive(
    row: dict[str, Any],
    source_name: str,
    file_name: str,
) -> dict[str, Any] | None:
    question = _extract_question(row)
    options = _extract_options_from_dict(row)
    answer = _extract_answer_text(row, options)
    if not question or len(options) < 2 or not answer:
        return None

    option_parts = [f"{label}. {text}" for label, text in sorted(options.items()) if text]
    context = _normalize_ws(f"Question: {question} Options: {' | '.join(option_parts)}")
    answer_start = context.find(answer)
    if answer_start < 0:
        return None
    if len(question) < 3 or len(context) < 20 or len(answer) < 1:
        return None
    return {
        "question": question,
        "context": context,
        "answers": {"text": [answer], "answer_start": [answer_start]},
        "source": source_name,
        "meta": {"file": file_name},
    }


def _dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out: list[dict[str, Any]] = []
    for row in records:
        q = _normalize_ws(_safe_text(row.get("question")))
        c = _normalize_ws(_safe_text(row.get("context")))
        answers = row.get("answers", {})
        answer_text = ""
        if isinstance(answers, dict):
            texts = answers.get("text", [])
            answer_text = _normalize_ws(_safe_text(texts[0] if texts else ""))
        key = (q.lower(), c.lower(), answer_text.lower())
        if not q or not c or not answer_text or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _split_train_validation(
    records: list[dict[str, Any]],
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rnd = random.Random(seed)
    rows = records[:]
    rnd.shuffle(rows)
    val_count = int(len(rows) * validation_ratio)
    if len(rows) >= 10:
        val_count = max(1, val_count)
    val_count = min(val_count, len(rows))
    validation = rows[:val_count]
    train = rows[val_count:]
    if not validation and train:
        validation.append(train.pop())
    if not train and validation:
        train.append(validation.pop())
    return train, validation


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MCQ datasets into extractive QA JSONL.")
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="Input file or directory; can be repeated.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory with train/validation jsonl.")
    parser.add_argument("--source-name", default="mcq_multi_subject", help="Source field value.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_paths = [Path(p) for p in args.inputs]
    files = _iter_files(input_paths)
    out_dir_raw = Path(args.out_dir)
    if out_dir_raw.is_absolute():
        out_dir = out_dir_raw
    else:
        out_dir = out_dir_raw.resolve()

    raw_count = 0
    kept_count = 0
    per_file: dict[str, int] = defaultdict(int)
    converted: list[dict[str, Any]] = []

    for file_path in files:
        rows = _read_records(file_path)
        raw_count += len(rows)
        for row in rows:
            rec = _to_extractive(row=row, source_name=args.source_name, file_name=file_path.name)
            if rec is None:
                continue
            converted.append(rec)
            kept_count += 1
            per_file[file_path.name] += 1

    cleaned = _dedupe(converted)
    train_rows, validation_rows = _split_train_validation(
        records=cleaned,
        validation_ratio=max(0.05, min(args.validation_ratio, 0.3)),
        seed=int(args.seed),
    )

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "validation.jsonl"
    manifest_path = out_dir / "manifest.json"

    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, validation_rows)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": "edu_mix_qa_local",
        "source_name": args.source_name,
        "input_files": [str(p) for p in files],
        "totals": {
            "raw_records": raw_count,
            "converted_before_dedupe": kept_count,
            "converted_after_dedupe": len(cleaned),
            "train": len(train_rows),
            "validation": len(validation_rows),
            "all": len(train_rows) + len(validation_rows),
        },
        "per_file_kept": per_file,
        "cleaning_rules": {
            "single_choice_only": True,
            "answer_in_context_required": True,
            "dedupe_key": "lower(question,context,answer)",
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "out_dir": str(out_dir), "totals": manifest["totals"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
