import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any


def _pick_text(item: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_relevance_label(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "unknown"
    if text in {"1", "true", "yes", "relevant", "entailment", "correct"}:
        return "relevant"
    if text in {"0", "false", "no", "off_topic", "irrelevant", "contradiction", "incorrect"}:
        return "off_topic"
    if text in {"partially_correct", "partial", "neutral"}:
        return "partial"
    return "unknown"


def _normalize_record(item: dict[str, Any], source: str) -> dict[str, Any] | None:
    question = _pick_text(
        item,
        [
            "question",
            "prompt",
            "assignment_question",
            "task",
            "query",
        ],
    )
    reference_answer = _pick_text(
        item,
        [
            "reference_answer",
            "reference",
            "gold_answer",
            "expected_answer",
            "model_answer",
            "teacher_answer",
        ],
    )
    student_answer = _pick_text(
        item,
        [
            "student_answer",
            "answer",
            "response",
            "content",
            "text",
        ],
    )
    teacher_feedback = _pick_text(
        item,
        [
            "teacher_feedback",
            "feedback",
            "comment",
            "explanation",
            "rationale",
        ],
    )
    raw_label = (
        item.get("label")
        or item.get("gold_label")
        or item.get("grade")
        or item.get("score_label")
        or ""
    )
    relevance = _normalize_relevance_label(raw_label)

    if not question or not student_answer:
        return None

    return {
        "question": question,
        "reference_answer": reference_answer,
        "student_answer": student_answer,
        "rubric_labels": {
            "raw_label": str(raw_label),
            "relevance": relevance,
        },
        "teacher_feedback": teacher_feedback or None,
        "source": source,
    }


def _read_jsonl(path: Path, source: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    output: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except Exception:
            continue
        if not isinstance(item, dict):
            continue
        normalized = _normalize_record(item, source=source)
        if normalized:
            output.append(normalized)
    return output


def _load_hf_records(dataset_name: str, source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(dataset_name)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []

    for split_name in ("train", "validation", "dev", "test"):
        if split_name not in ds:
            continue
        target = train_rows if split_name == "train" else val_rows
        for item in ds[split_name]:
            if not isinstance(item, dict):
                continue
            normalized = _normalize_record(item, source=source)
            if normalized:
                target.append(normalized)

    return train_rows, val_rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(item, ensure_ascii=False) for item in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _source_stats(rows: list[dict[str, Any]]) -> dict[str, int]:
    stats: dict[str, int] = {}
    for item in rows:
        source = str(item.get("source", "")).strip() or "unknown"
        stats[source] = stats.get(source, 0) + 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build assignment feedback training mix in unified schema.")
    parser.add_argument("--out-dir", default="training/data/assignment_feedback_mix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--include-scientsbank", action="store_true")
    parser.add_argument("--include-beetle", action="store_true")
    parser.add_argument("--local-train", default="")
    parser.add_argument("--local-validation", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    source_counts: dict[str, dict[str, int]] = {}

    if args.include_scientsbank:
        try:
            train, val = _load_hf_records("nkazi/SciEntsBank", source="scientsbank")
            train_rows.extend(train)
            val_rows.extend(val)
            source_counts["scientsbank"] = {"train": len(train), "validation": len(val)}
        except Exception as exc:
            source_counts["scientsbank_error"] = {"train": 0, "validation": 0}
            print(f"[WARN] skip nkazi/SciEntsBank due to error: {exc}")

    if args.include_beetle:
        try:
            train, val = _load_hf_records("nkazi/Beetle", source="beetle")
            train_rows.extend(train)
            val_rows.extend(val)
            source_counts["beetle"] = {"train": len(train), "validation": len(val)}
        except Exception as exc:
            source_counts["beetle_error"] = {"train": 0, "validation": 0}
            print(f"[WARN] skip nkazi/Beetle due to error: {exc}")

    if args.local_train:
        rows = _read_jsonl(Path(args.local_train).expanduser(), source="local_teacher")
        train_rows.extend(rows)
        source_counts["local_train"] = {"train": len(rows), "validation": 0}

    if args.local_validation:
        rows = _read_jsonl(Path(args.local_validation).expanduser(), source="local_teacher")
        val_rows.extend(rows)
        source_counts["local_validation"] = {"train": 0, "validation": len(rows)}

    random.shuffle(train_rows)
    random.shuffle(val_rows)

    if not train_rows:
        raise RuntimeError("No training rows built. Provide local jsonl or enable public datasets.")

    if not val_rows:
        split_idx = max(1, int(len(train_rows) * (1.0 - args.val_ratio)))
        val_rows = train_rows[split_idx:]
        train_rows = train_rows[:split_idx]

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "validation.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": "assignment_feedback_mix",
        "schema": {
            "question": "str",
            "reference_answer": "str",
            "student_answer": "str",
            "rubric_labels": {"raw_label": "str", "relevance": "relevant|off_topic|partial|unknown"},
            "teacher_feedback": "str|null",
            "source": "str",
        },
        "train_path": str(train_path.resolve()),
        "validation_path": str(val_path.resolve()),
        "totals": {
            "train": len(train_rows),
            "validation": len(val_rows),
            "all": len(train_rows) + len(val_rows),
        },
        "sources": source_counts,
        "source_mix_train": _source_stats(train_rows),
        "source_mix_validation": _source_stats(val_rows),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "out_dir": str(out_dir.resolve()), "train": len(train_rows), "validation": len(val_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
