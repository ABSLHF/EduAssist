import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.rag.pipeline import EmbeddingModelUnavailable, retrieve

TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")


@dataclass
class EvalRow:
    idx: int
    question: str
    knowledge_point: str
    difficulty: str
    gold_answer: str
    rag_answer: str
    rag_mode: str
    rag_score: float
    rag_hit: bool
    qa_answer: str
    qa_confidence: float | None
    qa_score: float
    qa_hit: bool


def normalize_text(text: str) -> str:
    text = text.replace("[Fallback]", " ").replace("[SmallQA]", " ")
    text = re.sub(r"Reference:\s*chunk_[^\s]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", "", text).lower().strip()
    return text


def token_set(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(normalize_text(text))}


def char_bigrams(text: str) -> set[str]:
    t = "".join(TOKEN_RE.findall(normalize_text(text)))
    if len(t) < 2:
        return {t} if t else set()
    return {t[i : i + 2] for i in range(len(t) - 1)}


def text_score(pred: str, gold: str) -> float:
    p = normalize_text(pred)
    g = normalize_text(gold)
    if not p or not g:
        return 0.0
    if p in g or g in p:
        return 1.0
    pt = token_set(pred) | char_bigrams(pred)
    gt = token_set(gold) | char_bigrams(gold)
    if not pt or not gt:
        return 0.0
    inter = len(pt & gt)
    precision = inter / max(1, len(pt))
    recall = inter / max(1, len(gt))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hit(score: float, threshold: float = 0.5) -> bool:
    return score >= threshold


def build_global_context(rows: list[dict[str, str]]) -> str:
    pairs: list[str] = []
    for r in rows:
        q = r["question"].strip()
        a = r["gold_answer"].strip()
        if q and a:
            pairs.append(f"Q: {q}\nA: {a}")
    return "\n\n".join(pairs)


def build_retrieved_context(course_id: int, question: str, top_k: int = 3) -> str:
    try:
        docs = retrieve(course_id, question, top_k=top_k)
    except EmbeddingModelUnavailable:
        return ""
    except Exception:
        return ""
    if not docs:
        return ""
    return "\n\n".join([doc for doc, _ in docs if doc])


def auth_token(client: httpx.Client, base_url: str, username: str, password: str) -> str:
    resp = client.post(
        f"{base_url}/auth/token",
        data={"username": username, "password": password},
        headers={"accept": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError("Missing access_token from /auth/token response")
    return token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 30-question teaching set against /qa and /model/qa_predict.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--course-id", type=int, required=True, help="Course id for /qa endpoint")
    parser.add_argument("--csv", default="training/data/teaching_eval_30.csv")
    parser.add_argument("--teacher-username", default="teacher1")
    parser.add_argument("--teacher-password", default="123456")
    parser.add_argument("--student-username", default="student1")
    parser.add_argument("--student-password", default="123456")
    parser.add_argument("--qa-model-path", default=None, help="Optional explicit model path for /model/qa_predict")
    parser.add_argument("--context-mode", choices=["retrieve", "global", "gold"], default="retrieve")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--hit-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise RuntimeError("Evaluation CSV is empty.")

    global_context = build_global_context(rows)
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = docs_dir / f"teaching_eval_report_{stamp}.json"
    out_md = docs_dir / f"teaching_eval_report_{stamp}.md"

    results: list[EvalRow] = []
    errors: list[dict[str, Any]] = []

    with httpx.Client(timeout=args.timeout, trust_env=False) as client:
        teacher_token = auth_token(client, base_url, args.teacher_username, args.teacher_password)
        student_token = auth_token(client, base_url, args.student_username, args.student_password)
        teacher_headers = {"Authorization": f"Bearer {teacher_token}", "accept": "application/json"}
        student_headers = {"Authorization": f"Bearer {student_token}", "accept": "application/json"}

        for i, row in enumerate(rows, start=1):
            q = row["question"].strip()
            gold = row["gold_answer"].strip()
            kp = row.get("knowledge_point", "").strip()
            diff = row.get("difficulty", "").strip()

            rag_answer = ""
            rag_mode = ""
            rag_score = 0.0
            rag_hit = False
            qa_answer = ""
            qa_confidence: float | None = None
            qa_score = 0.0
            qa_hit = False

            try:
                qa_payload = {"course_id": args.course_id, "question": q}
                rag_resp = client.post(f"{base_url}/qa/", json=qa_payload, headers=student_headers)
                rag_data = rag_resp.json()
                rag_answer = str(rag_data.get("answer", ""))
                rag_mode = str(rag_data.get("mode", ""))
                rag_score = text_score(rag_answer, gold)
                rag_hit = hit(rag_score, args.hit_threshold)
            except Exception as exc:
                errors.append({"idx": i, "stage": "qa", "question": q, "error": str(exc)})

            try:
                if args.context_mode == "gold":
                    context = gold
                elif args.context_mode == "retrieve":
                    context = build_retrieved_context(args.course_id, q, top_k=3)
                else:
                    context = global_context
                qa_predict_payload: dict[str, Any] = {"question": q, "context": context}
                if args.qa_model_path:
                    qa_predict_payload["model_path"] = args.qa_model_path
                pred_resp = client.post(f"{base_url}/model/qa_predict", json=qa_predict_payload, headers=teacher_headers)
                pred_data = pred_resp.json()
                qa_answer = str(pred_data.get("answer", ""))
                raw_conf = pred_data.get("confidence")
                qa_confidence = float(raw_conf) if raw_conf is not None else None
                qa_score = text_score(qa_answer, gold)
                qa_hit = hit(qa_score, args.hit_threshold)
            except Exception as exc:
                errors.append({"idx": i, "stage": "qa_predict", "question": q, "error": str(exc)})

            results.append(
                EvalRow(
                    idx=i,
                    question=q,
                    knowledge_point=kp,
                    difficulty=diff,
                    gold_answer=gold,
                    rag_answer=rag_answer,
                    rag_mode=rag_mode,
                    rag_score=round(rag_score, 4),
                    rag_hit=rag_hit,
                    qa_answer=qa_answer,
                    qa_confidence=qa_confidence,
                    qa_score=round(qa_score, 4),
                    qa_hit=qa_hit,
                )
            )

    total = len(results)
    rag_hits = sum(1 for r in results if r.rag_hit)
    qa_hits = sum(1 for r in results if r.qa_hit)
    rag_avg = round(sum(r.rag_score for r in results) / max(1, total), 4)
    qa_avg = round(sum(r.qa_score for r in results) / max(1, total), 4)
    rag_rate = round(rag_hits / max(1, total), 4)
    qa_rate = round(qa_hits / max(1, total), 4)

    payload = {
        "summary": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "base_url": base_url,
            "course_id": args.course_id,
            "context_mode": args.context_mode,
            "total": total,
            "rag": {"hit_count": rag_hits, "hit_rate": rag_rate, "avg_score": rag_avg},
            "qa_predict": {"hit_count": qa_hits, "hit_rate": qa_rate, "avg_score": qa_avg},
            "delta_hit_rate": round(qa_rate - rag_rate, 4),
            "errors": len(errors),
        },
        "results": [asdict(r) for r in results],
        "errors": errors,
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 教学评测对比报告（30题）",
        "",
        f"- 时间: {payload['summary']['timestamp']}",
        f"- 服务: `{base_url}`",
        f"- 课程ID: `{args.course_id}`",
        f"- 样本总数: `{total}`",
        f"- 上下文模式: `{args.context_mode}`",
        "",
        "## 总结",
        "",
        f"- RAG 命中: {rag_hits}/{total}，命中率 `{rag_rate}`，平均分 `{rag_avg}`",
        f"- QA抽取 命中: {qa_hits}/{total}，命中率 `{qa_rate}`，平均分 `{qa_avg}`",
        f"- 命中率增量（QA-RAG）: `{payload['summary']['delta_hit_rate']}`",
        f"- 错误条数: `{len(errors)}`",
        "",
        "## 明细",
        "",
        "| # | 难度 | 知识点 | RAG命中 | RAG分数 | QA命中 | QA分数 |",
        "|---|---|---|---|---:|---|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.idx} | {r.difficulty} | {r.knowledge_point} | {str(r.rag_hit)} | {r.rag_score} | {str(r.qa_hit)} | {r.qa_score} |"
        )

    if errors:
        lines.extend(["", "## 错误详情", ""])
        for e in errors:
            lines.append(f"- #{e['idx']} [{e['stage']}] {e['question']} -> {e['error']}")

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] JSON report: {out_json}")
    print(f"[OK] Markdown report: {out_md}")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
