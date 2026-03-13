from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.llm.client import call_llm
from app.services.assignment_feedback import generate_text_assignment_feedback


@dataclass
class EvalAssignment:
    title: str
    description: str | None
    keywords: str | None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _pick_text(item: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_label(item: dict[str, Any]) -> str:
    rubric = item.get("rubric_labels") if isinstance(item.get("rubric_labels"), dict) else {}
    raw = str((rubric or {}).get("relevance") or item.get("label") or "").strip().lower()
    if raw in {"off_topic", "irrelevant", "contradiction", "0", "false"}:
        return "off_topic"
    if raw in {"relevant", "good", "correct", "1", "true"}:
        return "relevant"
    return "partial"


def _feedback_says_offtopic(text: str) -> bool:
    t = (text or "").strip()
    return any(k in t for k in ("偏题", "不一致", "未围绕题干", "核心概念不一致"))


def _overscope_hit(question: str, feedback: str) -> bool:
    q = (question or "").lower()
    blocked = ["时间复杂度", "空间复杂度", "应用场景", "优缺点", "比较", "对比"]
    for token in blocked:
        if token in feedback and token not in q:
            return True
    return False


async def _direct_llm_feedback(question: str, answer: str) -> str:
    prompt = (
        "你是高校课程助教，请直接生成中文作业评语。"
        "不要出现分数，不要用Markdown。\n"
        f"题目：{question}\n"
        f"学生答案：{answer}\n"
    )
    return await call_llm(prompt)


async def main_async(args) -> dict[str, Any]:
    rows = _read_jsonl(Path(args.dataset).expanduser())
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("No samples found.")

    results: list[dict[str, Any]] = []
    t_direct: list[float] = []
    t_pipeline: list[float] = []
    err_direct = 0
    err_pipeline = 0
    overscope_direct = 0
    overscope_pipeline = 0

    for item in rows:
        question = _pick_text(item, ["question", "prompt", "assignment_question", "task", "query"])
        answer = _pick_text(item, ["student_answer", "answer", "response", "content", "text"])
        if not question or not answer:
            continue
        expected = _normalize_label(item)

        t0 = time.perf_counter()
        direct_feedback = (await _direct_llm_feedback(question, answer)).strip()
        t1 = time.perf_counter()
        assignment = EvalAssignment(title="作业", description=question, keywords=None)
        pipeline_feedback = (await generate_text_assignment_feedback(assignment, answer, call_llm, db=None)).strip()
        t2 = time.perf_counter()

        direct_offtopic = _feedback_says_offtopic(direct_feedback)
        pipeline_offtopic = _feedback_says_offtopic(pipeline_feedback)
        if expected == "relevant" and direct_offtopic:
            err_direct += 1
        if expected == "relevant" and pipeline_offtopic:
            err_pipeline += 1
        if _overscope_hit(question, direct_feedback):
            overscope_direct += 1
        if _overscope_hit(question, pipeline_feedback):
            overscope_pipeline += 1

        t_direct.append(t1 - t0)
        t_pipeline.append(t2 - t1)
        results.append(
            {
                "question": question,
                "answer": answer,
                "expected_label": expected,
                "direct_feedback": direct_feedback,
                "pipeline_feedback": pipeline_feedback,
                "direct_offtopic": direct_offtopic,
                "pipeline_offtopic": pipeline_offtopic,
            }
        )

    n = max(1, len(results))
    report = {
        "samples": len(results),
        "metrics": {
            "relevant_false_offtopic_rate_direct": round(err_direct / n, 4),
            "relevant_false_offtopic_rate_pipeline": round(err_pipeline / n, 4),
            "overscope_rate_direct": round(overscope_direct / n, 4),
            "overscope_rate_pipeline": round(overscope_pipeline / n, 4),
            "avg_latency_direct_sec": round(statistics.mean(t_direct) if t_direct else 0.0, 4),
            "avg_latency_pipeline_sec": round(statistics.mean(t_pipeline) if t_pipeline else 0.0, 4),
        },
        "examples": results[: min(20, len(results))],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline shadow evaluation for assignment feedback.")
    parser.add_argument("--dataset", default="training/data/assignment_feedback_mix/validation.jsonl")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--out", default="docs/eval/assignment_feedback_shadow.json")
    args = parser.parse_args()

    report = asyncio.run(main_async(args))
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(out_path.resolve()), "samples": report["samples"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

