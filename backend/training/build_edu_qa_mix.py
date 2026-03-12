import argparse
import json
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "training" / "data" / "edu_mix_qa"
DEFAULT_MATERIAL_DIR = ROOT / "training" / "data" / "course_pack" / "materials_clean"

SENTENCE_SPLIT_RE = re.compile(r"[。！？!?]\s*|\n+")
TERM_PREFIX_RE = re.compile(r"^\s*([A-Za-z0-9\u4e00-\u9fff]{2,16})(?:是|指|用于|属于|表示)(.+)$")

LICENSE_NOTES = {
    "cmrc2018": "CMRC2018 (research usage, check upstream license before redistribution).",
    "dureader": "DuReader (research usage, check upstream license before redistribution).",
    "drcd": "DRCD (research/non-commercial, check upstream license before redistribution).",
    "course_local_auto": "Auto-generated from local course materials in this project.",
}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_valid_sample(question: str, context: str, answer: str, answer_start: int) -> bool:
    if len(question) < 3 or len(context) < 30 or len(answer) < 2:
        return False
    if answer_start < 0:
        return False
    return True


def _to_record(
    question: str,
    context: str,
    answer: str,
    source: str,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    q = _normalize_ws(question)
    c = _normalize_ws(context)
    a = _normalize_ws(answer)
    answer_start = c.find(a) if a else -1
    if not _is_valid_sample(q, c, a, answer_start):
        return None
    return {
        "question": q,
        "context": c,
        "answers": {
            "text": [a],
            "answer_start": [answer_start],
        },
        "source": source,
        "meta": meta or {},
    }


def _extract_answer_fields(example: dict[str, Any]) -> tuple[str, int]:
    answers = example.get("answers")
    if isinstance(answers, dict):
        texts = answers.get("text", [])
        starts = answers.get("answer_start", [])
        if texts:
            text = _safe_text(texts[0])
            start = int(starts[0]) if starts else -1
            return text, start
    if isinstance(answers, list) and answers:
        first = answers[0]
        if isinstance(first, dict):
            text = _safe_text(first.get("text") or first.get("answer"))
            start = first.get("answer_start", -1)
            try:
                return text, int(start)
            except Exception:
                return text, -1
        text = _safe_text(first)
        return text, -1

    answer_text = _safe_text(example.get("answer"))
    answer_start = example.get("answer_start", -1)
    try:
        answer_start = int(answer_start)
    except Exception:
        answer_start = -1
    return answer_text, answer_start


def _extract_context(example: dict[str, Any]) -> str:
    direct = _safe_text(example.get("context"))
    if direct:
        return direct

    documents = example.get("documents")
    if isinstance(documents, list):
        chunks: list[str] = []
        for item in documents[:2]:
            if isinstance(item, dict):
                paragraphs = item.get("paragraphs")
                if isinstance(paragraphs, list):
                    chunks.extend([_safe_text(p) for p in paragraphs[:3] if _safe_text(p)])
                else:
                    chunks.append(_safe_text(item.get("document")))
            else:
                chunks.append(_safe_text(item))
        merged = " ".join([part for part in chunks if part])
        if merged:
            return merged
    return ""


def _extract_question(example: dict[str, Any]) -> str:
    return _safe_text(example.get("question") or example.get("query"))


def _load_hf_source(
    source_name: str,
    dataset_candidates: list[str],
    max_samples: int,
) -> tuple[list[dict[str, Any]], str]:
    if load_dataset is None:
        return [], "datasets package unavailable"

    last_error = "unknown error"
    for candidate in dataset_candidates:
        try:
            ds = load_dataset(candidate)
            break
        except Exception as exc:
            last_error = str(exc)
            ds = None
    else:
        return [], last_error

    examples: list[dict[str, Any]] = []
    for split_name in ("train", "validation"):
        if split_name not in ds:
            continue
        split_ds = ds[split_name]
        take_n = min(len(split_ds), max_samples)
        rows = split_ds.select(range(take_n))
        for row in rows:
            if not isinstance(row, dict):
                continue
            question = _extract_question(row)
            context = _extract_context(row)
            answer_text, answer_start = _extract_answer_fields(row)
            if answer_start < 0 and answer_text:
                answer_start = context.find(answer_text)
            if not _is_valid_sample(question, context, answer_text, answer_start):
                continue
            rec = {
                "question": _normalize_ws(question),
                "context": _normalize_ws(context),
                "answers": {
                    "text": [_normalize_ws(answer_text)],
                    "answer_start": [answer_start],
                },
                "source": source_name,
                "meta": {"dataset": candidate, "split": split_name},
            }
            examples.append(rec)

    if len(examples) > max_samples:
        examples = examples[:max_samples]
    return examples, ""


def _build_course_local_samples(material_dir: Path, max_samples: int) -> list[dict[str, Any]]:
    if not material_dir.exists():
        return []

    records: list[dict[str, Any]] = []
    for txt_file in sorted(material_dir.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if len(s.strip()) >= 12]
        for idx, sentence in enumerate(sentences):
            match = TERM_PREFIX_RE.match(sentence)
            if not match:
                continue
            term = _normalize_ws(match.group(1))
            if len(term) < 2 or len(term) > 16:
                continue

            answer = sentence if sentence.endswith("。") else sentence + "。"
            next_sentence = sentences[idx + 1] if idx + 1 < len(sentences) else ""
            context = answer + (next_sentence if next_sentence else "")
            question = f"{term}是什么？"
            rec = _to_record(
                question=question,
                context=context,
                answer=answer,
                source="course_local_auto",
                meta={"file": txt_file.name},
            )
            if rec:
                records.append(rec)
            if len(records) >= max_samples:
                return records
    return records


def _dedupe_and_clean(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen = set()
    for item in records:
        question = _safe_text(item.get("question"))
        context = _safe_text(item.get("context"))
        answers = item.get("answers", {})
        answer_list = answers.get("text", []) if isinstance(answers, dict) else []
        answer = _safe_text(answer_list[0] if answer_list else "")
        answer_start_list = answers.get("answer_start", []) if isinstance(answers, dict) else []
        answer_start = int(answer_start_list[0]) if answer_start_list else context.find(answer)

        key = (question.lower(), context.lower(), answer.lower())
        if key in seen:
            continue
        if not _is_valid_sample(question, context, answer, answer_start):
            continue

        seen.add(key)
        out.append(
            {
                "question": question,
                "context": context,
                "answers": {"text": [answer], "answer_start": [answer_start]},
                "source": _safe_text(item.get("source")) or "unknown",
                "meta": item.get("meta", {}),
            }
        )
    return out


def _split_train_validation(
    records: list[dict[str, Any]],
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rnd = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[_safe_text(rec.get("source")) or "unknown"].append(rec)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for group in grouped.values():
        rnd.shuffle(group)
        val_count = max(1, int(len(group) * validation_ratio)) if len(group) >= 8 else 1 if len(group) >= 3 else 0
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
    parser = argparse.ArgumentParser(description="Build mixed educational extractive QA dataset.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--max-cmrc", type=int, default=7000)
    parser.add_argument("--max-dureader", type=int, default=2500)
    parser.add_argument("--max-drcd", type=int, default=1500)
    parser.add_argument("--max-course", type=int, default=2000)
    parser.add_argument("--include-drcd", action="store_true")
    parser.add_argument("--course-material-dir", default=str(DEFAULT_MATERIAL_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    material_dir = Path(args.course_material_dir)

    all_records: list[dict[str, Any]] = []
    source_errors: dict[str, str] = {}

    max_cmrc = int(args.max_cmrc)
    if max_cmrc > 0:
        cmrc_records, cmrc_err = _load_hf_source(
            source_name="cmrc2018",
            dataset_candidates=["cmrc2018"],
            max_samples=max_cmrc,
        )
        all_records.extend(cmrc_records)
        if cmrc_err:
            source_errors["cmrc2018"] = cmrc_err
    else:
        source_errors["cmrc2018"] = "skipped by config (--max-cmrc <= 0)"

    max_dureader = int(args.max_dureader)
    if max_dureader > 0:
        dureader_records, dr_err = _load_hf_source(
            source_name="dureader",
            dataset_candidates=["dureader_robust", "PaddlePaddle/dureader_robust"],
            max_samples=max_dureader,
        )
        all_records.extend(dureader_records)
        if dr_err:
            source_errors["dureader"] = dr_err
    else:
        source_errors["dureader"] = "skipped by config (--max-dureader <= 0)"

    max_drcd = int(args.max_drcd)
    if args.include_drcd and max_drcd > 0:
        drcd_records, drcd_err = _load_hf_source(
            source_name="drcd",
            dataset_candidates=["DRCD", "drcd"],
            max_samples=max_drcd,
        )
        all_records.extend(drcd_records)
        if drcd_err:
            source_errors["drcd"] = drcd_err
    elif args.include_drcd:
        source_errors["drcd"] = "skipped by config (--max-drcd <= 0)"

    local_records = _build_course_local_samples(material_dir=material_dir, max_samples=max(1, int(args.max_course)))
    all_records.extend(local_records)

    cleaned = _dedupe_and_clean(all_records)
    train_rows, validation_rows = _split_train_validation(
        cleaned,
        validation_ratio=max(0.05, min(float(args.validation_ratio), 0.3)),
        seed=int(args.seed),
    )

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "validation.jsonl"
    manifest_path = out_dir / "manifest.json"

    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, validation_rows)

    source_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "validation": 0, "total": 0})
    for row in train_rows:
        source = _safe_text(row.get("source")) or "unknown"
        source_counts[source]["train"] += 1
        source_counts[source]["total"] += 1
    for row in validation_rows:
        source = _safe_text(row.get("source")) or "unknown"
        source_counts[source]["validation"] += 1
        source_counts[source]["total"] += 1

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": "edu_mix_qa_local",
        "train_path": str(train_path),
        "validation_path": str(val_path),
        "totals": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "all": len(train_rows) + len(validation_rows),
        },
        "sources": source_counts,
        "license_notes": {k: LICENSE_NOTES.get(k, "Check upstream license.") for k in source_counts.keys()},
        "build_errors": source_errors,
        "cleaning_rules": {
            "answer_in_context_required": True,
            "min_context_chars": 30,
            "min_answer_chars": 2,
            "dedupe_key": "lower(question,context,answer)",
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"ok": True, "out_dir": str(out_dir), "totals": manifest["totals"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
