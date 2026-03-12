import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")
INSUFFICIENT_PATTERNS = ("资料不足以确认", "资料不足以确定")
DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "data_structure": ("数据结构", "链表", "栈", "队列", "二叉树", "图", "哈希", "排序", "查找"),
    "database": ("数据库", "sql", "事务", "acid", "索引", "范式", "锁", "查询"),
    "operating_system": ("操作系统", "进程", "线程", "调度", "死锁", "分页", "分段", "内存管理"),
    "network": ("计算机网络", "网络", "tcp", "udp", "ip", "http", "dns", "路由"),
}


@dataclass
class EvalRow:
    set_name: str
    course_id: int
    idx: int
    question: str
    knowledge_point: str
    difficulty: str
    gold_answer: str
    rag_answer: str
    rag_mode: str
    rag_score: float
    rag_hit: bool
    rag_fallback: bool
    has_references: bool
    off_topic: bool
    insufficient_misfire: bool
    qa_answer: str
    qa_confidence: float | None
    qa_score: float
    qa_hit: bool


def safe_cell(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


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


def hit(score: float, threshold: float) -> bool:
    return score >= threshold


def infer_domain(text: str) -> str | None:
    sample = normalize_text(text)
    if not sample:
        return None
    best_domain = None
    best_score = 0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for keyword in keywords if normalize_text(keyword) in sample)
        if score > best_score:
            best_score = score
            best_domain = domain
    return best_domain if best_score > 0 else None


def is_off_topic(question: str, answer: str) -> bool:
    q_domain = infer_domain(question)
    if not q_domain:
        return False
    if any(flag in answer for flag in INSUFFICIENT_PATTERNS):
        return False
    a_domain = infer_domain(answer)
    if not a_domain:
        return False
    return a_domain != q_domain


def is_insufficient_misfire(answer: str, has_references: bool) -> bool:
    if not has_references:
        return False
    return any(flag in answer for flag in INSUFFICIENT_PATTERNS)


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
        raise RuntimeError("missing access_token from /auth/token response")
    return token


def ensure_join_course(client: httpx.Client, base_url: str, headers: dict[str, str], course_id: int) -> None:
    try:
        client.post(f"{base_url}/courses/{course_id}/join", headers=headers)
    except Exception:
        return


def build_retrieved_context(course_id: int, question: str, top_k: int = 5) -> str:
    try:
        from app.rag.pipeline import EmbeddingModelUnavailable, retrieve
    except Exception:
        return ""
    try:
        docs = retrieve(course_id, question, top_k=top_k)
    except EmbeddingModelUnavailable:
        return ""
    except Exception:
        return ""
    if not docs:
        return ""
    return "\n\n".join([doc for doc, _ in docs if doc])


def load_suite(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sets = payload.get("sets", [])
    if not isinstance(sets, list) or not sets:
        raise ValueError("suite file must contain non-empty 'sets'")
    validated: list[dict[str, Any]] = []
    for item in sets:
        if not isinstance(item, dict):
            continue
        name = str(item.get("course_name", "")).strip()
        csv_path = str(item.get("csv_path", "")).strip()
        course_id = item.get("course_id")
        if not name or not csv_path:
            continue
        try:
            course_id_int = int(course_id) if course_id is not None else None
        except Exception:
            course_id_int = None
        validated.append(
            {
                "course_name": name,
                "course_id": course_id_int,
                "csv_path": csv_path,
            }
        )
    if not validated:
        raise ValueError("suite file has no valid set entries")
    return validated


def summarize_rows(rows: list[EvalRow], threshold: float) -> dict[str, Any]:
    total = len(rows)
    rag_hits = sum(1 for row in rows if row.rag_hit)
    qa_hits = sum(1 for row in rows if row.qa_hit)
    rag_avg = round(sum(row.rag_score for row in rows) / max(1, total), 4)
    qa_avg = round(sum(row.qa_score for row in rows) / max(1, total), 4)
    fallback_ratio = round(sum(1 for row in rows if row.rag_fallback) / max(1, total), 4)
    reference_coverage = round(sum(1 for row in rows if row.has_references) / max(1, total), 4)
    off_topic_rate = round(sum(1 for row in rows if row.off_topic) / max(1, total), 4)
    insufficient_misfire_rate = round(sum(1 for row in rows if row.insufficient_misfire) / max(1, total), 4)
    return {
        "total": total,
        "hit_threshold": threshold,
        "rag": {
            "hit_count": rag_hits,
            "hit_rate": round(rag_hits / max(1, total), 4),
            "avg_score": rag_avg,
        },
        "qa_predict": {
            "hit_count": qa_hits,
            "hit_rate": round(qa_hits / max(1, total), 4),
            "avg_score": qa_avg,
        },
        "fallback_ratio": fallback_ratio,
        "reference_coverage": reference_coverage,
        "off_topic_rate": off_topic_rate,
        "insufficient_misfire_rate": insufficient_misfire_rate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CS suite against /qa and /model/qa_predict.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--suite", default="training/data/eval_cs_suite.json")
    parser.add_argument("--label", default="run", help="pre/post/run")
    parser.add_argument("--teacher-username", default="teacher1")
    parser.add_argument("--teacher-password", default="123456")
    parser.add_argument("--student-username", default="student1")
    parser.add_argument("--student-password", default="123456")
    parser.add_argument("--qa-model-path", default=None)
    parser.add_argument("--context-mode", choices=["retrieve", "gold"], default="retrieve")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--hit-threshold", type=float, default=0.5)
    parser.add_argument("--override-course-id", type=int, default=None)
    parser.add_argument("--auto-join", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    suite_path = Path(args.suite)
    if not suite_path.is_absolute():
        suite_path = ROOT_DIR / suite_path
    if not suite_path.exists():
        raise FileNotFoundError(f"suite file not found: {suite_path}")

    suite_sets = load_suite(suite_path)
    docs_dir = Path("docs") / "eval"
    docs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = docs_dir / f"{args.label}_{stamp}.json"
    out_md = docs_dir / f"{args.label}_{stamp}.md"

    all_rows: list[EvalRow] = []
    set_summaries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    with httpx.Client(timeout=args.timeout, trust_env=False) as client:
        teacher_token = auth_token(client, base_url, args.teacher_username, args.teacher_password)
        student_token = auth_token(client, base_url, args.student_username, args.student_password)
        teacher_headers = {"Authorization": f"Bearer {teacher_token}", "accept": "application/json"}
        student_headers = {"Authorization": f"Bearer {student_token}", "accept": "application/json"}

        for set_item in suite_sets:
            course_name = set_item["course_name"]
            course_id = args.override_course_id or set_item["course_id"]
            csv_path = Path(set_item["csv_path"])
            if not csv_path.is_absolute():
                csv_path = ROOT_DIR / csv_path
            if not csv_path.exists():
                errors.append({"set": course_name, "stage": "suite", "error": f"csv not found: {csv_path}"})
                continue
            if course_id is None:
                errors.append({"set": course_name, "stage": "suite", "error": "missing course_id"})
                continue
            if args.auto_join:
                ensure_join_course(client, base_url, student_headers, int(course_id))

            with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
                rows = list(csv.DictReader(fp))

            set_rows: list[EvalRow] = []
            for i, row in enumerate(rows, start=1):
                question = safe_cell(row, "question")
                gold = safe_cell(row, "gold_answer")
                kp = safe_cell(row, "knowledge_point")
                difficulty = safe_cell(row, "difficulty")
                if not question or not gold:
                    errors.append(
                        {
                            "set": course_name,
                            "idx": i,
                            "stage": "csv",
                            "error": "missing question or gold_answer",
                        }
                    )
                    continue

                rag_answer = ""
                rag_mode = ""
                rag_score = 0.0
                rag_hit = False
                rag_fallback = False
                has_references = False
                off_topic = False
                insufficient_misfire = False
                qa_answer = ""
                qa_confidence: float | None = None
                qa_score = 0.0
                qa_hit = False

                try:
                    qa_payload = {"course_id": int(course_id), "question": question}
                    rag_resp = client.post(f"{base_url}/qa/", json=qa_payload, headers=student_headers)
                    if rag_resp.status_code >= 400:
                        raise RuntimeError(f"/qa status={rag_resp.status_code} body={rag_resp.text[:200]}")
                    rag_data = rag_resp.json()
                    rag_answer = str(rag_data.get("answer", ""))
                    rag_mode = str(rag_data.get("mode", ""))
                    refs = rag_data.get("references", [])
                    if isinstance(refs, list):
                        has_references = len(refs) > 0
                    rag_fallback = "fallback" in rag_mode.lower()
                    off_topic = is_off_topic(question, rag_answer)
                    insufficient_misfire = is_insufficient_misfire(rag_answer, has_references)
                    rag_score = text_score(rag_answer, gold)
                    rag_hit = hit(rag_score, args.hit_threshold)
                except Exception as exc:
                    errors.append(
                        {
                            "set": course_name,
                            "idx": i,
                            "stage": "qa",
                            "question": question,
                            "error": str(exc),
                        }
                    )

                try:
                    if args.context_mode == "gold":
                        context = gold
                    else:
                        context = build_retrieved_context(int(course_id), question, top_k=max(1, args.top_k))
                    qa_predict_payload: dict[str, Any] = {"question": question, "context": context}
                    if args.qa_model_path:
                        qa_predict_payload["model_path"] = args.qa_model_path
                    pred_resp = client.post(
                        f"{base_url}/model/qa_predict",
                        json=qa_predict_payload,
                        headers=teacher_headers,
                    )
                    if pred_resp.status_code >= 400:
                        raise RuntimeError(f"/model/qa_predict status={pred_resp.status_code} body={pred_resp.text[:200]}")
                    pred_data = pred_resp.json()
                    qa_answer = str(pred_data.get("answer", ""))
                    raw_conf = pred_data.get("confidence")
                    qa_confidence = float(raw_conf) if raw_conf is not None else None
                    qa_score = text_score(qa_answer, gold)
                    qa_hit = hit(qa_score, args.hit_threshold)
                except Exception as exc:
                    errors.append(
                        {
                            "set": course_name,
                            "idx": i,
                            "stage": "qa_predict",
                            "question": question,
                            "error": str(exc),
                        }
                    )

                eval_row = EvalRow(
                    set_name=course_name,
                    course_id=int(course_id),
                    idx=i,
                    question=question,
                    knowledge_point=kp,
                    difficulty=difficulty,
                    gold_answer=gold,
                    rag_answer=rag_answer,
                    rag_mode=rag_mode,
                    rag_score=round(rag_score, 4),
                    rag_hit=rag_hit,
                    rag_fallback=rag_fallback,
                    has_references=has_references,
                    off_topic=off_topic,
                    insufficient_misfire=insufficient_misfire,
                    qa_answer=qa_answer,
                    qa_confidence=qa_confidence,
                    qa_score=round(qa_score, 4),
                    qa_hit=qa_hit,
                )
                set_rows.append(eval_row)
                all_rows.append(eval_row)

            metrics = summarize_rows(set_rows, args.hit_threshold)
            set_summaries.append(
                {
                    "course_name": course_name,
                    "course_id": int(course_id),
                    "csv_path": str(csv_path),
                    **metrics,
                }
            )

    overall = summarize_rows(all_rows, args.hit_threshold)
    payload = {
        "summary": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "label": args.label,
            "base_url": base_url,
            "suite_path": str(suite_path),
            "context_mode": args.context_mode,
            "set_count": len(set_summaries),
            **overall,
            "errors": len(errors),
        },
        "sets": set_summaries,
        "rows": [asdict(row) for row in all_rows],
        "errors": errors,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 高校CS通用评测报告",
        "",
        f"- 时间: {payload['summary']['timestamp']}",
        f"- 标签: `{args.label}`",
        f"- 服务: `{base_url}`",
        f"- 套件: `{suite_path}`",
        f"- 上下文模式: `{args.context_mode}`",
        f"- 样本总数: `{overall['total']}`",
        "",
        "## 总体指标",
        "",
        f"- RAG命中率: `{overall['rag']['hit_rate']}` (avg={overall['rag']['avg_score']})",
        f"- QA抽取命中率: `{overall['qa_predict']['hit_rate']}` (avg={overall['qa_predict']['avg_score']})",
        f"- RAG fallback占比: `{overall['fallback_ratio']}`",
        f"- 引用覆盖率: `{overall['reference_coverage']}`",
        f"- 答非所问率: `{overall['off_topic_rate']}`",
        f"- 资料不足误判率: `{overall['insufficient_misfire_rate']}`",
        f"- 错误条数: `{len(errors)}`",
        "",
        "## 分课程指标",
        "",
        "| 课程 | 样本 | RAG命中率 | QA命中率 | fallback占比 | 引用覆盖率 | 答非所问率 | 资料不足误判率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in set_summaries:
        lines.append(
            f"| {item['course_name']} | {item['total']} | {item['rag']['hit_rate']} | "
            f"{item['qa_predict']['hit_rate']} | {item['fallback_ratio']} | {item['reference_coverage']} | "
            f"{item['off_topic_rate']} | {item['insufficient_misfire_rate']} |"
        )

    if errors:
        lines.extend(["", "## 错误详情", ""])
        for err in errors[:80]:
            lines.append(f"- {json.dumps(err, ensure_ascii=False)}")
        if len(errors) > 80:
            lines.append(f"- ... 共 {len(errors)} 条错误，仅展示前80条")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] JSON report: {out_json}")
    print(f"[OK] Markdown report: {out_md}")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
