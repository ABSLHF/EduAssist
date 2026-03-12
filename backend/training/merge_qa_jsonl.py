import argparse
import json
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _fix_record(record: dict[str, Any], source_hint: str) -> dict[str, Any] | None:
    question = _normalize_ws(_safe_text(record.get("question")))
    context = _normalize_ws(_safe_text(record.get("context")))
    answers = record.get("answers")
    answer_text = ""
    answer_start = -1
    if isinstance(answers, dict):
        texts = answers.get("text", [])
        starts = answers.get("answer_start", [])
        answer_text = _normalize_ws(_safe_text(texts[0] if texts else ""))
        try:
            answer_start = int(starts[0]) if starts else -1
        except Exception:
            answer_start = -1
    if answer_start < 0 and answer_text:
        answer_start = context.find(answer_text)

    source = _safe_text(record.get("source")) or source_hint
    if len(question) < 3 or len(context) < 30 or len(answer_text) < 1 or answer_start < 0:
        return None
    return {
        "question": question,
        "context": context,
        "answers": {"text": [answer_text], "answer_start": [answer_start]},
        "source": source,
        "meta": record.get("meta", {}),
    }


def _split_train_validation(
    records: list[dict[str, Any]],
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rnd = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        source = _safe_text(rec.get("source")) or "unknown"
        grouped[source].append(rec)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for group in grouped.values():
        rnd.shuffle(group)
        val_count = int(len(group) * validation_ratio)
        if len(group) >= 8:
            val_count = max(1, val_count)
        val_count = min(val_count, len(group))
        validation.extend(group[:val_count])
        train.extend(group[val_count:])

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
    parser = argparse.ArgumentParser(description="Merge multiple extractive QA datasets into one JSONL dataset.")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset entry in format name=path (path can be a directory or a train.jsonl file).",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory path.")
    parser.add_argument("--resplit", action="store_true", help="Ignore original splits and resplit by source.")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-per-dataset", type=int, default=0)
    parser.add_argument("--max-validation-per-dataset", type=int, default=0)
    return parser.parse_args()


def _parse_entry(entry: str) -> tuple[str, Path]:
    if "=" not in entry:
        raise ValueError(f"Invalid --dataset entry: {entry}")
    name, path_text = entry.split("=", 1)
    name = name.strip()
    path = Path(path_text.strip())
    if path.is_absolute():
        return name, path
    path_resolved = path.resolve()
    if path_resolved.exists():
        return name, path_resolved
    path_alt = (ROOT / path).resolve()
    path = path_alt if path_alt.exists() else path_resolved
    return name, path


def main() -> int:
    args = parse_args()
    out_dir_raw = Path(args.out_dir)
    if out_dir_raw.is_absolute():
        out_dir = out_dir_raw
    else:
        out_dir = out_dir_raw.resolve()

    merged_train: list[dict[str, Any]] = []
    merged_val: list[dict[str, Any]] = []
    source_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "validation": 0, "total": 0})
    dataset_stats: dict[str, dict[str, int]] = {}
    seen = set()

    for raw_entry in args.dataset:
        dataset_name, path = _parse_entry(raw_entry)
        if path.is_dir():
            train_path = path / "train.jsonl"
            val_path = path / "validation.jsonl"
            train_rows = _read_jsonl(train_path)
            val_rows = _read_jsonl(val_path)
        else:
            train_rows = _read_jsonl(path)
            val_rows = []

        if args.max_train_per_dataset > 0:
            train_rows = train_rows[: args.max_train_per_dataset]
        if args.max_validation_per_dataset > 0:
            val_rows = val_rows[: args.max_validation_per_dataset]

        fixed_train = [_fix_record(r, source_hint=dataset_name) for r in train_rows]
        fixed_val = [_fix_record(r, source_hint=dataset_name) for r in val_rows]
        fixed_train = [row for row in fixed_train if row is not None]
        fixed_val = [row for row in fixed_val if row is not None]

        dataset_stats[dataset_name] = {
            "train_loaded": len(fixed_train),
            "validation_loaded": len(fixed_val),
            "total_loaded": len(fixed_train) + len(fixed_val),
        }

        for row in fixed_train:
            q = _normalize_ws(_safe_text(row.get("question")))
            c = _normalize_ws(_safe_text(row.get("context")))
            answer_text = _safe_text(row["answers"]["text"][0])
            key = (q.lower(), c.lower(), answer_text.lower())
            if key in seen:
                continue
            seen.add(key)
            merged_train.append(row)

        for row in fixed_val:
            q = _normalize_ws(_safe_text(row.get("question")))
            c = _normalize_ws(_safe_text(row.get("context")))
            answer_text = _safe_text(row["answers"]["text"][0])
            key = (q.lower(), c.lower(), answer_text.lower())
            if key in seen:
                continue
            seen.add(key)
            merged_val.append(row)

    if args.resplit:
        combined = merged_train + merged_val
        merged_train, merged_val = _split_train_validation(
            records=combined,
            validation_ratio=max(0.05, min(args.validation_ratio, 0.3)),
            seed=int(args.seed),
        )

    for row in merged_train:
        source = _safe_text(row.get("source")) or "unknown"
        source_counts[source]["train"] += 1
        source_counts[source]["total"] += 1
    for row in merged_val:
        source = _safe_text(row.get("source")) or "unknown"
        source_counts[source]["validation"] += 1
        source_counts[source]["total"] += 1

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "validation.jsonl"
    manifest_path = out_dir / "manifest.json"
    _write_jsonl(train_path, merged_train)
    _write_jsonl(val_path, merged_val)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": "edu_mix_qa_local",
        "train_path": str(train_path),
        "validation_path": str(val_path),
        "totals": {
            "train": len(merged_train),
            "validation": len(merged_val),
            "all": len(merged_train) + len(merged_val),
        },
        "datasets": dataset_stats,
        "sources": source_counts,
        "resplit": bool(args.resplit),
        "validation_ratio": float(args.validation_ratio),
        "cleaning_rules": {
            "answer_in_context_required": True,
            "dedupe_key": "lower(question,context,answer)",
            "min_context_chars": 30,
            "min_question_chars": 3,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "out_dir": str(out_dir), "totals": manifest["totals"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
