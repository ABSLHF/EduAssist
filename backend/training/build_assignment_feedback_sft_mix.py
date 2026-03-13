import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def _pick_text(item: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_relevance(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if text in {"off_topic", "irrelevant", "contradiction", "0", "false", "no"}:
        return "off_topic"
    if text in {"relevant", "correct", "entailment", "1", "true", "yes", "good"}:
        return "good"
    if text in {"partial", "neutral", "unknown", "ok"}:
        return "partial"
    return "partial"


def _infer_tier(question: str, answer: str, relevance: str) -> str:
    pure = re.sub(r"\s+", "", answer or "")
    if len(pure) < 4:
        return "invalid"
    if relevance == "off_topic":
        return "off_topic"
    if relevance == "good" and len(answer.strip()) >= 40:
        return "good"
    return "partial"


def _forbidden_hints(question: str) -> list[str]:
    q = (question or "").lower()
    hints = {"复杂度", "时间复杂度", "空间复杂度", "应用场景", "优缺点", "比较", "对比"}
    if any(k in q for k in ("复杂度", "时间复杂度", "空间复杂度")):
        hints -= {"复杂度", "时间复杂度", "空间复杂度"}
    if any(k in q for k in ("应用", "场景", "案例", "实例")):
        hints -= {"应用场景"}
    if any(k in q for k in ("比较", "对比", "优缺点", "区别", "异同")):
        hints -= {"比较", "对比", "优缺点"}
    return sorted(hints)


def _clean_feedback_text(text: str) -> str:
    raw = (text or "").strip()
    raw = raw.replace("**", "").replace("`", "")
    raw = re.sub(r"#+\s*", "", raw)
    raw = re.sub(r"(?:评分|得分|score)\s*[:：]\s*\d{1,3}", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\n{2,}", "\n", raw).strip()
    return raw


def _template_feedback(tier: str, must_cover: str) -> str:
    if tier == "invalid":
        return (
            "问题：答案信息不足，无法判断对题干的理解。\n"
            f"改进建议：请先给出清晰定义，并围绕“{must_cover or '题干核心概念'}”补充关键要点。"
        )
    if tier == "off_topic":
        return (
            "问题：回答与题干核心概念不一致。\n"
            f"改进建议：请围绕“{must_cover or '题干核心概念'}”重写答案，先给定义再补充要点。"
        )
    if tier == "partial":
        return (
            "问题：答案覆盖了部分要点，但完整性不足。\n"
            f"改进建议：请补齐“{must_cover or '题干关键要点'}”，并按“定义-要点-示例”组织答案。"
        )
    return (
        "优点：回答与题干核心概念基本对齐，结构较清晰。\n"
        "改进建议：可补充1-2个关键细节并精炼表述。"
    )


def _to_sft_record(item: dict[str, Any], source: str) -> dict[str, Any] | None:
    question = _pick_text(item, ["question", "prompt", "assignment_question", "task", "query"])
    reference_answer = _pick_text(item, ["reference_answer", "reference", "gold_answer", "expected_answer", "teacher_answer"])
    student_answer = _pick_text(item, ["student_answer", "answer", "response", "content", "text"])
    teacher_feedback = _pick_text(item, ["teacher_feedback", "feedback", "comment", "explanation", "rationale"])
    if not question or not student_answer:
        return None

    rubric = item.get("rubric_labels") if isinstance(item.get("rubric_labels"), dict) else {}
    relevance = _normalize_relevance((rubric or {}).get("relevance") or item.get("label"))
    tier = _infer_tier(question, student_answer, relevance)
    forbidden = _forbidden_hints(question)
    must_cover = question[:30]

    instruction = (
        "你是高校课程助教，请生成中文作业评语。"
        f"档位={tier}。"
        "不要出现评分。不要使用Markdown。"
        f"题干未要求时不要引入：{('、'.join(forbidden) if forbidden else '无')}。"
    )
    model_input = (
        f"题目：{question}\n"
        f"参考答案：{reference_answer or '无'}\n"
        f"学生答案：{student_answer}\n"
    )

    output = _clean_feedback_text(teacher_feedback) if teacher_feedback else ""
    if not output:
        output = _template_feedback(tier=tier, must_cover=must_cover)

    return {
        "instruction": instruction,
        "input": model_input,
        "output": output,
        "tier": tier,
        "source": source,
    }


def _load_jsonl(path: Path, source: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
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
        record = _to_sft_record(item, source=source)
        if record:
            rows.append(record)
    return rows


def _load_hf(dataset_name: str, source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(dataset_name)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "dev", "test"):
        if split not in ds:
            continue
        target = train_rows if split == "train" else val_rows
        for item in ds[split]:
            if not isinstance(item, dict):
                continue
            record = _to_sft_record(item, source=source)
            if record:
                target.append(record)
    return train_rows, val_rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    stats: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "")).strip() or "unknown"
        stats[value] = stats.get(value, 0) + 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT dataset for assignment feedback generation.")
    parser.add_argument("--out-dir", default="training/data/assignment_feedback_sft_mix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--include-scientsbank", action="store_true")
    parser.add_argument("--include-beetle", action="store_true")
    parser.add_argument("--local-train", default="")
    parser.add_argument("--local-validation", default="")
    parser.add_argument("--from-feedback-mix", default="training/data/assignment_feedback_mix")
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    sources: dict[str, dict[str, int]] = {}

    if args.include_scientsbank:
        try:
            tr, va = _load_hf("nkazi/SciEntsBank", "scientsbank")
            train_rows.extend(tr)
            val_rows.extend(va)
            sources["scientsbank"] = {"train": len(tr), "validation": len(va)}
        except Exception as exc:
            sources["scientsbank_error"] = {"train": 0, "validation": 0}
            print(f"[WARN] skip nkazi/SciEntsBank due to error: {exc}")

    if args.include_beetle:
        try:
            tr, va = _load_hf("nkazi/Beetle", "beetle")
            train_rows.extend(tr)
            val_rows.extend(va)
            sources["beetle"] = {"train": len(tr), "validation": len(va)}
        except Exception as exc:
            sources["beetle_error"] = {"train": 0, "validation": 0}
            print(f"[WARN] skip nkazi/Beetle due to error: {exc}")

    if args.from_feedback_mix:
        base = Path(args.from_feedback_mix).expanduser()
        tr = _load_jsonl(base / "train.jsonl", "feedback_mix")
        va = _load_jsonl(base / "validation.jsonl", "feedback_mix")
        train_rows.extend(tr)
        val_rows.extend(va)
        sources["feedback_mix"] = {"train": len(tr), "validation": len(va)}

    if args.local_train:
        rows = _load_jsonl(Path(args.local_train).expanduser(), "local_teacher")
        train_rows.extend(rows)
        sources["local_train"] = {"train": len(rows), "validation": 0}

    if args.local_validation:
        rows = _load_jsonl(Path(args.local_validation).expanduser(), "local_teacher")
        val_rows.extend(rows)
        sources["local_validation"] = {"train": 0, "validation": len(rows)}

    if not train_rows:
        raise RuntimeError("No training rows built for SFT. Please provide local jsonl or enable public datasets.")

    random.shuffle(train_rows)
    random.shuffle(val_rows)
    if not val_rows:
        split = max(1, int(len(train_rows) * (1.0 - args.val_ratio)))
        val_rows = train_rows[split:]
        train_rows = train_rows[:split]

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "validation.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": "assignment_feedback_sft_mix",
        "schema": {"instruction": "str", "input": "str", "output": "str", "tier": "str", "source": "str"},
        "train_path": str(train_path.resolve()),
        "validation_path": str(val_path.resolve()),
        "totals": {"train": len(train_rows), "validation": len(val_rows), "all": len(train_rows) + len(val_rows)},
        "sources": sources,
        "tier_distribution_train": _count_by(train_rows, "tier"),
        "tier_distribution_validation": _count_by(val_rows, "tier"),
        "source_mix_train": _count_by(train_rows, "source"),
        "source_mix_validation": _count_by(val_rows, "source"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"ok": True, "out_dir": str(out_dir.resolve()), "train": len(train_rows), "validation": len(val_rows)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

