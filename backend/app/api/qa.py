from collections import OrderedDict
import json
import logging
from pathlib import Path
import re
import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.config import settings
from app.db import get_db
from app.llm.client import call_llm
from app.models import models
from app.rag.pipeline import EmbeddingModelUnavailable, retrieve
from app.services.model_paths import resolve_active_qa_model_path
from app.services.qa_small_model import predict_answer
from app.schemas.schemas import QARequest, QAResponse

router = APIRouter()
CACHE_MAX = 100
QA_CACHE_VERSION = "v4"
qa_cache: OrderedDict[str, dict] = OrderedDict()
logger = logging.getLogger(__name__)
MAX_DOC_ANSWER_CHARS = 900
MAX_NO_DOC_ANSWER_CHARS = 700
RULE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "data" / "qa_rule_answers.json"
_rule_cache: list[dict] | None = None
_rule_cache_mtime: float | None = None

ZH_DEFINITION_TAGS = ["是", "指", "表示", "定义", "用于", "特点", "区别"]
TOKEN_SPLIT_PATTERN = re.compile(r"[^\w\u4e00-\u9fff]+")
SINGLE_CHAR_TERMS = {"树", "栈", "图", "堆"}
QUESTION_NOISE_TOKENS = [
    "什么是",
    "解释一下",
    "解释",
    "介绍",
    "说明",
    "为什么",
    "怎么",
    "如何",
    "是什么",
    "会发生",
    "基本思想",
    "核心特性",
    "特点",
    "定义",
    "区别",
    "不同",
    "？",
    "?",
]
GENERIC_TERMS = {
    "特点",
    "定义",
    "区别",
    "不同",
    "原因",
    "作用",
    "效果",
    "基本思想",
    "核心特性",
    "会发生",
}
FOLLOWUP_HINT_TERMS = [
    "再",
    "继续",
    "详细",
    "展开",
    "补充",
    "举例",
    "深入",
    "进一步",
    "上面",
    "前面",
    "刚才",
    "这个",
    "那个",
    "它",
]
FOLLOWUP_GENERIC_TERMS = {
    "再详细一点",
    "再详细",
    "详细一点",
    "详细介绍",
    "进一步",
    "进一步介绍",
    "进一步说明",
    "继续讲",
    "继续",
    "继续介绍",
    "再讲讲",
    "展开讲",
    "补充一下",
    "举个例子",
    "再说一下",
    "多介绍",
}
QUESTION_TRAILING_SYMBOLS = " \t\r\n'\"`‘’“”＇＂，,。！？?；;：:"
NOISY_LINE_PATTERNS = [
    re.compile(r"^\s*#"),
    re.compile(r"^\s*【[^】]*】\s*$"),
    re.compile(r"^\s*[-=_]{3,}\s*$"),
    re.compile(r"^\s*(一句话总结|知识点|问题|答案)[:：]?\s*$"),
]
DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "data_structure": ("数据结构", "链表", "栈", "队列", "二叉树", "图", "哈希", "排序", "查找"),
    "database": ("数据库", "sql", "事务", "acid", "索引", "范式", "并发控制", "锁", "查询优化"),
    "operating_system": ("操作系统", "进程", "线程", "调度", "死锁", "内存管理", "分页", "分段", "文件系统"),
    "network": ("计算机网络", "网络", "tcp", "udp", "ip", "路由", "http", "dns", "链路层"),
}
DOMAIN_LABELS = {
    "data_structure": "数据结构",
    "database": "数据库系统",
    "operating_system": "操作系统",
    "network": "计算机网络",
}


def _is_noisy_line(line: str) -> bool:
    if not line:
        return True
    if len(line) < 8:
        return True
    for pattern in NOISY_LINE_PATTERNS:
        if pattern.search(line):
            return True
    if len(line) <= 18 and re.search(r"(区别|定义|特点|是什么)$", line) and not re.search(r"[，。；：]", line):
        return True
    return False


def _normalize_candidate_line(line: str) -> str:
    text = line.strip()
    text = re.sub(r"^\s*【[^】]*】\s*", "", text)
    text = re.sub(r"^\s*一句话总结[:：]?\s*", "", text)
    return text.strip()


def _normalize_question_text(question: str) -> str:
    q = (question or "").strip()
    q = q.rstrip(QUESTION_TRAILING_SYMBOLS)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _question_keywords(question: str) -> list[str]:
    q = _normalize_question_text(question)
    for token in QUESTION_NOISE_TOKENS:
        q = q.replace(token, " ")

    def _keep_term(term: str) -> bool:
        return len(term) >= 2 or term in SINGLE_CHAR_TERMS

    base: list[str] = []
    for raw in TOKEN_SPLIT_PATTERN.split(q):
        term = raw.strip()
        if not term:
            continue
        if "的" in term:
            base.extend([p.strip() for p in term.split("的") if p.strip()])
        else:
            base.append(term)
    base = [t for t in base if _keep_term(t) and t not in GENERIC_TERMS]

    keywords: list[str] = []
    for term in base:
        keywords.append(term)
        for sep in ("与", "和", "及", "、"):
            if sep in term:
                keywords.extend([p.strip() for p in term.split(sep) if _keep_term(p.strip())])

    out: list[str] = []
    seen = set()
    for term in keywords:
        if term not in seen:
            out.append(term)
            seen.add(term)
    return out


def _infer_question_domain(text: str) -> str | None:
    sample = _normalize_question_text(text).lower()
    if not sample:
        return None

    best_domain = None
    best_score = 0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword and keyword.lower() in sample)
        if score > best_score:
            best_score = score
            best_domain = domain
    return best_domain if best_score > 0 else None


def _course_domain_from_name(course_name: str | None) -> str | None:
    if not course_name:
        return None
    return _infer_question_domain(course_name)


def _domain_label(domain: str | None) -> str:
    if not domain:
        return "未识别领域"
    return DOMAIN_LABELS.get(domain, domain)


def _history_user_questions(history, limit: int = 10) -> list[str]:
    if not history:
        return []
    user_questions: list[str] = []
    for msg in history[-limit:]:
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "user")
        content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
        role = (role or "user").strip().lower()
        content = (content or "").strip()
        if role == "user" and content:
            user_questions.append(content)
    return user_questions


def _is_followup_question(question: str) -> bool:
    q = _normalize_question_text(question)
    if not q:
        return False
    keywords = _question_keywords(q)
    topical_keywords = [kw for kw in keywords if kw not in FOLLOWUP_GENERIC_TERMS]
    short_query = len(q) <= 18
    has_followup_hint = any(term in q for term in FOLLOWUP_HINT_TERMS)

    if has_followup_hint and short_query and len(topical_keywords) <= 1:
        return True
    if short_query and not topical_keywords:
        return True
    return False


def _history_topic_anchor(question: str, history) -> str | None:
    user_questions = _history_user_questions(history, limit=12)
    current = _normalize_question_text(question)

    for candidate in reversed(user_questions):
        item = _normalize_question_text(candidate)
        if not item or item == current:
            continue
        if not _is_followup_question(item):
            return item

    for candidate in reversed(user_questions):
        item = _normalize_question_text(candidate)
        if item and item != current:
            return item
    return None


def _build_retrieval_question(question: str, history) -> tuple[str, str | None]:
    q = _normalize_question_text(question)
    if not q or not history or not _is_followup_question(q):
        return q, None
    anchor = _history_topic_anchor(q, history)
    if not anchor:
        return q, None
    return f"{anchor}。追问：{q}", anchor


def _definition_candidates(question: str, docs: list[tuple[str, dict]], limit: int = 2) -> list[str]:
    keywords = _question_keywords(question)
    scored_with_kw: list[tuple[int, str]] = []
    scored_general: list[tuple[int, str]] = []
    for doc, _meta in docs:
        for raw_line in doc.splitlines():
            line = _normalize_candidate_line(raw_line.strip())
            if _is_noisy_line(line):
                continue
            if line.endswith("是什么") or line.endswith("会发生"):
                continue
            looks_like_definition = any(tag in line for tag in ZH_DEFINITION_TAGS)
            hit_keyword = any(k in line for k in keywords) if keywords else True
            score = 0
            if looks_like_definition:
                score += 2
            if hit_keyword:
                score += 2
            if "区别" in line or "不同" in line:
                score += 1
            if score > 0:
                if hit_keyword:
                    scored_with_kw.append((score, line))
                else:
                    scored_general.append((score, line))

    scored = scored_with_kw if scored_with_kw else scored_general
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[str] = []
        seen = set()
        for _score, line in scored:
            if line in seen:
                continue
            out.append(line)
            seen.add(line)
            if len(out) >= limit:
                break
        if out:
            return out

    candidates: list[str] = []
    if keywords:
        for doc, _meta in docs:
            for raw_line in doc.splitlines():
                line = _normalize_candidate_line(raw_line.strip())
                if _is_noisy_line(line):
                    continue
                if any(k in line for k in keywords):
                    candidates.append(line)
                    if len(candidates) >= limit:
                        return candidates

    if not candidates:
        for doc, _meta in docs:
            for raw_line in doc.splitlines():
                line = _normalize_candidate_line(raw_line.strip())
                if _is_noisy_line(line):
                    continue
                candidates.append(line)
                if len(candidates) >= limit:
                    return candidates
    return candidates[:limit]


def _fallback_evidence_lines(question: str, docs: list[tuple[str, dict]], max_lines: int = 3) -> list[tuple[str, str]]:
    keywords = _question_keywords(question)
    evidences: list[tuple[str, str]] = []
    seen = set()

    for idx, (doc, _meta) in enumerate(docs, start=1):
        for raw_line in doc.splitlines():
            line = _normalize_candidate_line(raw_line.strip())
            if _is_noisy_line(line):
                continue
            if keywords and not any(k in line for k in keywords):
                continue
            if line in seen:
                continue
            seen.add(line)
            evidences.append((line, f"[R{idx}]"))
            break
        if len(evidences) >= max_lines:
            return evidences

    if len(evidences) < max_lines:
        for idx, (doc, _meta) in enumerate(docs, start=1):
            for raw_line in doc.splitlines():
                line = _normalize_candidate_line(raw_line.strip())
                if _is_noisy_line(line) or line in seen:
                    continue
                seen.add(line)
                evidences.append((line, f"[R{idx}]"))
                break
            if len(evidences) >= max_lines:
                break
    return evidences[:max_lines]


def _build_fallback_answer(question: str, docs: list[tuple[str, dict]], references: list[str]) -> str:
    evidences = _fallback_evidence_lines(question, docs, max_lines=3)
    if not evidences:
        answer = "[Fallback] 资料不足以确定。请补充相关课程资料，或把问题细化到具体章节与术语。"
        if references:
            answer += "\n\n参考资料：\n" + "\n".join([f"- {item}" for item in references[:5]])
        return answer

    lines = ["[Fallback] 基于课程资料可得到以下依据："]
    for idx, (evidence, ref_mark) in enumerate(evidences, start=1):
        lines.append(f"{idx}. {evidence} {ref_mark}")
    lines.append("结论仍可能不完整；如需更准确答案，请补充更聚焦的课程资料或细化问题。")
    if references:
        lines.append("")
        lines.append("参考资料：")
        lines.extend([f"- {item}" for item in references[:5]])
    return "\n".join(lines)


def _extract_finetuned_evidences(
    question: str,
    docs: list[tuple[str, dict]],
    model_path: str,
    min_conf: float,
    top_chunks: int,
    max_evidence: int,
) -> list[dict]:
    evidences: list[dict] = []
    seen = set()
    for idx, (doc, _meta) in enumerate(docs[: max(1, top_chunks)], start=1):
        try:
            answer, confidence = predict_answer(question, doc, model_path=model_path)
        except Exception:
            continue

        answer_text = _normalize_candidate_line(answer or "")
        if not answer_text or answer_text in seen:
            continue
        if confidence is None or float(confidence) < min_conf:
            continue

        seen.add(answer_text)
        evidences.append(
            {
                "answer": answer_text,
                "confidence": float(confidence),
                "ref_mark": f"[R{idx}]",
            }
        )
        if len(evidences) >= max(1, max_evidence):
            break
    return evidences


def _build_finetuned_evidence_block(evidences: list[dict]) -> str:
    if not evidences:
        return ""
    lines = ["微调抽取证据（优先引用，且不得与课程资料冲突）："]
    for idx, item in enumerate(evidences, start=1):
        lines.append(
            f"- 证据{idx} {item['ref_mark']} (confidence={item['confidence']:.3f})：{item['answer']}"
        )
    return "\n".join(lines)


def _build_finetuned_fallback_answer(evidences: list[dict], references: list[str]) -> str:
    if not evidences:
        answer = "[Fallback] 资料不足以确定。当前未检索到足够可靠的课程证据，请补充资料后重试。"
        if references:
            answer += "\n\n参考资料：\n" + "\n".join([f"- {item}" for item in references[:5]])
        return answer

    lines = ["[Fallback] 基于微调模型从课程资料抽取到以下证据："]
    for idx, item in enumerate(evidences, start=1):
        lines.append(f"{idx}. {item['answer']} {item['ref_mark']} (score={item['confidence']:.3f})")
    lines.append("以上为证据性回答，如需完整讲解可继续追问具体知识点。")
    if references:
        lines.append("")
        lines.append("参考资料：")
        lines.extend([f"- {item}" for item in references[:5]])
    return "\n".join(lines)


def _course_cache_version(db: Session, course_ids: list[int]) -> str:
    if not course_ids:
        return "none"
    latest_parsed = (
        db.query(func.max(models.Material.parsed_at))
        .filter(
            models.Material.course_id.in_(course_ids),
            models.Material.parse_status == "success",
        )
        .scalar()
    )
    return latest_parsed.isoformat() if latest_parsed else "none"


def _user_accessible_course_ids(db: Session, user_id: int) -> list[int]:
    enrolled_rows = db.query(models.UserCourse.course_id).filter(models.UserCourse.user_id == user_id).all()
    taught_rows = db.query(models.Course.id).filter(models.Course.teacher_id == user_id).all()
    ids = {int(row[0]) for row in enrolled_rows + taught_rows if row and row[0] is not None}
    if ids:
        return sorted(ids)
    # 兜底：即使学生尚未加入课程，也保留可用课程范围，避免问答直接不可用。
    all_rows = db.query(models.Course.id).all()
    return sorted({int(row[0]) for row in all_rows if row and row[0] is not None})


def _has_course_access(db: Session, user_id: int, course_id: int) -> bool:
    exists = db.get(models.Course, course_id)
    if not exists:
        return False
    if exists.teacher_id == user_id:
        return True
    enrolled = (
        db.query(models.UserCourse.id)
        .filter(models.UserCourse.user_id == user_id, models.UserCourse.course_id == course_id)
        .first()
    )
    return enrolled is not None


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _course_name_map(db: Session, course_ids: list[int]) -> dict[int, str]:
    if not course_ids:
        return {}
    rows = db.query(models.Course.id, models.Course.name).filter(models.Course.id.in_(set(course_ids))).all()
    return {int(row.id): str(row.name) for row in rows}


def _doc_keyword_score(question: str, doc: str) -> int:
    keywords = _question_keywords(question)
    if not keywords:
        return 0
    score = 0
    for kw in keywords[:10]:
        if kw and kw in doc:
            score += 2
    if "区别" in question and ("区别" in doc or "不同" in doc):
        score += 1
    if "为什么" in question and ("原因" in doc or "导致" in doc):
        score += 1
    return score


def _rerank_course_ids_by_domain(
    course_ids: list[int],
    course_name_map: dict[int, str],
    question_domain: str | None,
    selected_course_id: int | None,
) -> list[int]:
    if not course_ids:
        return []
    if not settings.qa_domain_route_enabled or not question_domain:
        return course_ids

    scored: list[tuple[float, int]] = []
    for idx, course_id in enumerate(course_ids):
        course_domain = _course_domain_from_name(course_name_map.get(course_id))
        score = float(-idx) * 0.01
        if selected_course_id is not None and course_id == selected_course_id:
            score += 2.0
        if course_domain == question_domain:
            score += 3.0
        scored.append((score, course_id))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [cid for _score, cid in scored]


def _rerank_docs_by_domain(
    docs: list[tuple[str, dict]],
    question: str,
    course_map: dict[int, str],
    question_domain: str | None,
    selected_course_id: int | None,
) -> list[tuple[str, dict]]:
    if not docs:
        return []
    if not settings.qa_domain_route_enabled or not question_domain:
        return docs

    ranked: list[tuple[float, tuple[str, dict]]] = []
    for idx, (doc, meta) in enumerate(docs):
        current_meta = dict(meta or {})
        course_id = _safe_int(current_meta.get("course_id"))
        course_domain = _course_domain_from_name(course_map.get(course_id))
        score = float(_doc_keyword_score(question, doc))
        score += float(-idx) * 0.01
        if selected_course_id is not None and course_id == selected_course_id:
            score += 1.5
        if course_domain == question_domain:
            score += 2.5
        ranked.append((score, (doc, current_meta)))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item for _score, item in ranked]


def _retrieve_docs_for_courses(
    course_ids: list[int],
    question: str,
    preferred_course_id: int | None = None,
    top_k: int = 5,
) -> list[tuple[str, dict]]:
    if not course_ids:
        return []
    merged: list[tuple[str, dict, float]] = []
    for course_id in course_ids:
        try:
            docs = retrieve(course_id, question, top_k=min(top_k, 3))
        except EmbeddingModelUnavailable:
            continue
        except Exception as exc:
            logger.warning("qa retrieve failed course_id=%s err=%s", course_id, exc)
            continue
        for doc, meta in docs:
            current_meta = dict(meta or {})
            current_meta.setdefault("course_id", course_id)
            score = float(_doc_keyword_score(question, doc))
            if preferred_course_id is not None and course_id == preferred_course_id:
                score += 1.5
            merged.append((doc, current_meta, score))

    if not merged:
        return []

    dedup: dict[tuple[int | None, int | None, int | None, str], tuple[str, dict, float]] = {}
    for doc, meta, score in merged:
        key = (
            _safe_int(meta.get("course_id")),
            _safe_int(meta.get("material_id")),
            _safe_int(meta.get("chunk_id")),
            doc,
        )
        old = dedup.get(key)
        if old is None or score > old[2]:
            dedup[key] = (doc, meta, score)

    ranked = sorted(dedup.values(), key=lambda item: item[2], reverse=True)
    return [(doc, meta) for doc, meta, _score in ranked[:top_k]]


def clear_qa_cache(course_id: int | None = None) -> None:
    if course_id is None:
        qa_cache.clear()
        return

    prefix = f"{QA_CACHE_VERSION}::{course_id}::"
    stale_keys = [key for key in qa_cache.keys() if key.startswith(prefix)]
    for key in stale_keys:
        qa_cache.pop(key, None)


def _material_info_map(db: Session, docs: list[tuple[str, dict]]) -> tuple[dict[int, tuple[str, int | None]], dict[int, str]]:
    material_ids: list[int] = []
    course_ids: list[int] = []
    for _doc, meta in docs:
        if not meta:
            continue
        mat_id = _safe_int(meta.get("material_id"))
        cid = _safe_int(meta.get("course_id"))
        if mat_id is not None:
            material_ids.append(mat_id)
        if cid is not None:
            course_ids.append(cid)

    material_map: dict[int, tuple[str, int | None]] = {}
    course_map: dict[int, str] = {}
    if not material_ids and not course_ids:
        return material_map, course_map

    if material_ids:
        material_rows = (
            db.query(models.Material.id, models.Material.title, models.Material.course_id)
            .filter(models.Material.id.in_(set(material_ids)))
            .all()
        )
        for row in material_rows:
            material_map[int(row.id)] = (row.title, _safe_int(row.course_id))
            if row.course_id is not None:
                course_ids.append(int(row.course_id))

    if course_ids:
        course_rows = db.query(models.Course.id, models.Course.name).filter(models.Course.id.in_(set(course_ids))).all()
        course_map = {int(row.id): row.name for row in course_rows}

    return material_map, course_map


def _build_references(
    docs: list[tuple[str, dict]],
    material_map: dict[int, tuple[str, int | None]],
    course_map: dict[int, str],
) -> list[str]:
    refs: list[str] = []
    seen = set()
    for _doc, meta in docs:
        if not meta:
            continue
        material_id = _safe_int(meta.get("material_id"))
        course_id = _safe_int(meta.get("course_id"))
        chunk_id = meta.get("chunk_id")
        title = f"资料{material_id}" if material_id is not None else "未知资料"
        if material_id is not None and material_id in material_map:
            title, material_course_id = material_map[material_id]
            course_id = course_id or material_course_id
        course_name = course_map.get(course_id, f"课程{course_id}") if course_id is not None else "未指定课程"
        label = f"{course_name}/{title}#chunk_{chunk_id}"
        if label in seen:
            continue
        refs.append(label)
        seen.add(label)
    return refs


def _build_context_blocks(
    docs: list[tuple[str, dict]],
    material_map: dict[int, tuple[str, int | None]],
    course_map: dict[int, str],
) -> str:
    blocks: list[str] = []
    for idx, (doc, meta) in enumerate(docs, start=1):
        meta = meta or {}
        material_id = _safe_int(meta.get("material_id"))
        course_id = _safe_int(meta.get("course_id"))
        chunk_id = meta.get("chunk_id")
        title = f"资料{material_id}" if material_id is not None else "未知资料"
        if material_id is not None and material_id in material_map:
            title, material_course_id = material_map[material_id]
            course_id = course_id or material_course_id
        course_name = course_map.get(course_id, f"课程{course_id}") if course_id is not None else "未指定课程"
        blocks.append(
            f"[R{idx}] 课程：{course_name}；资料：{title} (course_id={course_id}, material_id={material_id}, chunk={chunk_id})\n"
            f"{doc}"
        )
    return "\n\n".join(blocks)


def _format_history(history) -> str:
    if not history:
        return ""
    lines: list[str] = []
    for msg in history[-6:]:
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "user")
        content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
        role = (role or "user").strip().lower()
        content = (content or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    if not lines:
        return ""
    return "Conversation history:\n" + "\n".join(lines) + "\n\n"


def _llm_failed(text: str | None) -> bool:
    if not text:
        return True
    error_prefixes = (
        "[Dify接口错误]",
        "[GLM接口错误]",
        "[文心接口错误]",
    )
    return text.startswith(error_prefixes)


def _trim_answer_lines(text: str, max_lines: int) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines])


def _clip_text(text: str, max_chars: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    clipped = cleaned[:max_chars].rstrip()
    return clipped + "..."


TERMINOLOGY_REPLACEMENTS = [
    ("先进后出", "后进先出"),
    ("先入后出", "后进先出"),
    ("后入先出", "后进先出"),
    ("节点", "结点"),
    ("结点和指针", "结点和指针"),
]


def _normalize_terminology(text: str) -> str:
    out = text
    for src, dst in TERMINOLOGY_REPLACEMENTS:
        out = out.replace(src, dst)
    return out


def _strip_markdown_artifacts(text: str) -> str:
    cleaned = text
    # 仅移除 Markdown 标题符号，保留编号和分点结构。
    cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    # 去掉反引号和链接残留，减少原始 Markdown 噪声。
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", cleaned)
    return cleaned


def _sanitize_llm_answer(text: str, docs_present: bool) -> str:
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").strip()
    cleaned = re.sub(r"【[^】]*】", "", cleaned)
    cleaned = re.sub(r"\[R\d+\]", "", cleaned)
    cleaned = re.sub(r"\[[^\]]*引用[^\]]*\]", "", cleaned)
    cleaned = _strip_markdown_artifacts(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = _trim_answer_lines(cleaned, max_lines=16 if docs_present else 12)
    cleaned = _normalize_terminology(cleaned)
    cleaned = _clip_text(
        cleaned,
        max_chars=MAX_DOC_ANSWER_CHARS if docs_present else MAX_NO_DOC_ANSWER_CHARS,
    )
    return cleaned


def _default_rule_entries() -> list[dict]:
    return [
        {"answer": "数据结构是计算机中组织和存储数据的方式。", "all": ["什么是数据结构"]},
        {"answer": "算法是解决问题的有限步骤集合。", "all": ["什么是算法"]},
        {"answer": "线性表是具有一对一顺序关系的数据元素集合。", "all": ["线性表"], "any": ["定义", "什么是"]},
        {"answer": "顺序表使用连续存储空间并支持按下标随机访问。", "all": ["顺序表"], "any": ["特点", "什么是"]},
        {"answer": "链表由结点和指针组成，插入删除高效但不支持随机访问。", "all": ["链表"], "any": ["特点", "什么是"]},
        {"answer": "栈遵循后进先出原则。", "all": ["栈"], "any": ["核心特性", "特点"]},
        {"answer": "队列遵循先进先出原则。", "all": ["队列"], "any": ["核心特性", "特点"]},
        {"answer": "树是由结点和边组成的分层非线性结构。", "all": ["什么是树"]},
        {"answer": "图由顶点和边组成，用于描述多对多关系。", "all": ["什么是图"]},
        {"answer": "哈希表通过哈希函数把键映射到存储位置。", "all": ["哈希表"], "any": ["基本思想", "什么是"]},
        {"answer": "时间复杂度用于描述算法运行时间随输入规模增长的趋势。", "all": ["时间复杂度"]},
        {"answer": "空间复杂度用于描述算法额外空间开销随输入规模增长的趋势。", "all": ["空间复杂度"]},
        {"answer": "顺序存储使用连续内存且随机访问快；链式存储不连续但插入删除更灵活。", "all": ["顺序存储", "链式存储", "区别"]},
        {"answer": "数组访问快但插入删除慢；链表访问慢但插入删除快。", "all": ["数组", "链表", "区别"]},
        {"answer": "栈是后进先出；队列是先进先出。", "all": ["栈", "队列", "区别"]},
        {"answer": "二叉搜索树满足左子树小于根且右子树大于根，二叉树不要求该有序性。", "all": ["二叉树", "二叉搜索树", "区别"]},
        {"answer": "BFS按层遍历常用队列；DFS沿深度遍历常用栈或递归。", "all": ["BFS", "DFS", "区别"]},
        {"answer": "冒泡排序实现简单但平均较慢；快速排序平均更快。", "all": ["冒泡排序", "快速排序", "区别"]},
        {"answer": "稳定排序能保持相等关键字元素的原有相对顺序。", "all": ["稳定排序"], "any": ["什么意思", "是什么"]},
        {"answer": "缓存位于CPU与主存之间，用于降低平均访存时间。", "all": ["主存", "缓存"], "any": ["关系", "区别"]},
        {"answer": "不同键经过哈希函数可能映射到同一地址，导致哈希冲突。", "all": ["哈希冲突"], "any": ["为什么", "发生"]},
        {"answer": "ACID是原子性、一致性、隔离性和持久性。", "all": ["ACID"]},
        {"answer": "递归层数过深会耗尽调用栈空间，导致栈溢出。", "all": ["递归", "栈溢出"]},
        {"answer": "看任务是否可拆分为独立子问题且数据依赖较少。", "all": ["并行化"], "any": ["如何判断", "是否可"]},
        {"answer": "索引通过额外结构缩小扫描范围，从而降低查询成本。", "all": ["索引", "加速查询"]},
        {"answer": "分页固定大小便于管理；分段按逻辑划分更贴近程序结构。", "all": ["分页", "分段", "区别"]},
        {"answer": "划分极不均衡时递归深度接近n，快排会退化到O(n^2)。", "any": ["快速排序最坏", "快排最坏"]},
        {"answer": "红黑树通过颜色约束和旋转操作维持近似平衡高度。", "all": ["红黑树", "平衡"]},
        {"answer": "多个事务相互等待对方持有的资源会形成循环等待并导致死锁。", "all": ["死锁"], "any": ["为什么", "发生"]},
        {"answer": "可从正确性、可解释性、引用命中率和教师满意度进行综合评价。", "all": ["教学问答系统"], "any": ["效果", "评价"]},
    ]


def _load_rule_entries() -> list[dict]:
    global _rule_cache, _rule_cache_mtime

    try:
        mtime = RULE_CONFIG_PATH.stat().st_mtime
    except OSError:
        if _rule_cache is None:
            _rule_cache = _default_rule_entries()
        return _rule_cache

    if _rule_cache is not None and _rule_cache_mtime == mtime:
        return _rule_cache

    try:
        payload = json.loads(RULE_CONFIG_PATH.read_text(encoding="utf-8"))
        raw_rules = payload.get("rules", [])
        parsed: list[dict] = []
        for idx, item in enumerate(raw_rules):
            if not isinstance(item, dict):
                continue
            answer = str(item.get("answer", "")).strip()
            if not answer:
                continue
            all_terms = [str(term).strip() for term in item.get("all", []) if str(term).strip()]
            any_terms = [str(term).strip() for term in item.get("any", []) if str(term).strip()]
            if not all_terms and not any_terms:
                continue
            priority = item.get("priority", idx)
            try:
                priority = int(priority)
            except Exception:
                priority = idx
            parsed.append(
                {
                    "answer": answer,
                    "all": all_terms,
                    "any": any_terms,
                    "priority": priority,
                }
            )
        parsed.sort(key=lambda r: r["priority"])
        _rule_cache = parsed or _default_rule_entries()
        _rule_cache_mtime = mtime
    except Exception as exc:
        logger.warning("load qa rule config failed: %s", exc)
        if _rule_cache is None:
            _rule_cache = _default_rule_entries()
    return _rule_cache or _default_rule_entries()


def _rule_matches(rule: dict, question: str) -> bool:
    all_terms = rule.get("all", [])
    any_terms = rule.get("any", [])
    if all_terms and not all(term in question for term in all_terms):
        return False
    if any_terms and not any(term in question for term in any_terms):
        return False
    return bool(all_terms or any_terms)


def _rule_based_answer(question: str) -> str | None:
    q = _normalize_question_text(question)
    for rule in _load_rule_entries():
        if _rule_matches(rule, q):
            return str(rule["answer"])
    return None


def _resolve_rule_answer(question: str, topic_anchor: str | None = None) -> tuple[str | None, str]:
    direct = _rule_based_answer(question)
    if direct:
        return direct, "direct"
    if topic_anchor:
        anchored = _rule_based_answer(topic_anchor)
        if anchored:
            return anchored, "anchor"
    return None, "none"


def _is_low_quality_doc_answer(question: str, answer: str, topic_hint: str | None = None) -> bool:
    if not answer or len(answer) < 20:
        return True
    if "资料不足以确定" in answer or "资料不足以确认" in answer:
        return True
    keyword_source = topic_hint or question
    keywords = _question_keywords(keyword_source)
    if keywords:
        matched = sum(1 for keyword in keywords[:10] if keyword in answer)
        if matched == 0 and len(answer) < 180:
            return True
    return False


def _expand_rule_answer(rule_answer: str, question: str) -> str:
    text = _normalize_terminology(rule_answer.strip())
    if not text:
        return text
    keywords = _question_keywords(question)
    topic = "、".join(keywords[:2]) if keywords else "该知识点"
    if len(text) >= 80:
        return text
    return (
        f"课程结论：{text}\n"
        f"可进一步从{topic}的定义、使用场景和常见对比点来理解，"
        "若需要我可以继续给出示例。"
    )


def _is_followup_insufficient_answer(answer: str) -> bool:
    normalized = (answer or "").strip()
    if not normalized:
        return True
    if "资料不足以确定" in normalized or "资料不足以确认" in normalized:
        return True
    return False


@router.post("/", response_model=QAResponse, summary="Course QA")
async def ask(payload: QARequest, db: Session = Depends(get_db), user=Depends(get_current_user)):
    started = time.perf_counter()
    failure_reason = "none"
    normalized_question = _normalize_question_text(payload.question)
    selected_course_id = _safe_int(payload.course_id) if payload.course_id is not None else None
    # 访问控制：用户显式选了课程时，先做课程权限校验。
    if selected_course_id is not None and not _has_course_access(db, user.id, selected_course_id):
        raise HTTPException(status_code=403, detail="Course not found or no access")

    accessible_course_ids = _user_accessible_course_ids(db, user.id)
    if selected_course_id is not None:
        target_course_ids = [selected_course_id] + [cid for cid in accessible_course_ids if cid != selected_course_id]
    else:
        target_course_ids = accessible_course_ids

    if not target_course_ids:
        raise HTTPException(status_code=400, detail="No available courses for QA")
    course_scope_key = selected_course_id if selected_course_id is not None else "all"
    course_name_map = _course_name_map(db, target_course_ids)
    question_domain = _infer_question_domain(normalized_question)
    selected_course_domain = _course_domain_from_name(course_name_map.get(selected_course_id)) if selected_course_id else None
    domain_mismatch = bool(
        settings.qa_domain_route_enabled
        and settings.qa_domain_mismatch_guard
        and selected_course_id is not None
        and question_domain
        and selected_course_domain
        and question_domain != selected_course_domain
    )

    if not payload.history:
        # 仅缓存首轮问答，避免多轮上下文导致缓存答案失真。
        cache_version = _course_cache_version(db, target_course_ids)
        cache_key = f"{QA_CACHE_VERSION}::{course_scope_key}::{cache_version}::{normalized_question}"
        if cache_key in qa_cache:
            cached = qa_cache.pop(cache_key)
            qa_cache[cache_key] = cached
            return cached
    else:
        cache_key = None

    retrieval_question, topic_anchor = _build_retrieval_question(normalized_question, payload.history)
    rule_answer, rule_source = _resolve_rule_answer(normalized_question, topic_anchor=topic_anchor)
    routed_course_ids = _rerank_course_ids_by_domain(
        target_course_ids,
        course_name_map=course_name_map,
        question_domain=question_domain,
        selected_course_id=selected_course_id,
    )
    docs = _retrieve_docs_for_courses(
        routed_course_ids,
        retrieval_question or normalized_question,
        preferred_course_id=selected_course_id,
        top_k=5,
    )

    material_map, course_map = _material_info_map(db, docs)
    if course_name_map:
        for cid, course_name in course_name_map.items():
            course_map.setdefault(cid, course_name)
    docs = _rerank_docs_by_domain(
        docs,
        question=normalized_question,
        course_map=course_map,
        question_domain=question_domain,
        selected_course_id=selected_course_id,
    )

    domain_switch_hint = ""
    if domain_mismatch:
        failure_reason = "course_domain_mismatch_guard"
        docs = []
        material_map = {}
        course_map = {}
        domain_switch_hint = (
            f"当前问题更偏向“{_domain_label(question_domain)}”，"
            f"但你选中的课程更偏向“{_domain_label(selected_course_domain)}”。"
            "建议切换到对应课程后再提问，或取消课程选择后重试。"
        )

    references = _build_references(docs, material_map, course_map)
    ft_model_path: str | None = None
    ft_model_source = "disabled"
    ft_evidences: list[dict] = []
    ft_used = False
    ft_conf_max: float | None = None

    if docs and settings.enable_finetuned_qa_in_rag:
        # 微调小模型只用于抽取证据，不直接作为最终回答生成器。
        ft_model_path, ft_model_source = resolve_active_qa_model_path(db)
        if ft_model_path:
            try:
                ft_evidences = _extract_finetuned_evidences(
                    normalized_question,
                    docs,
                    model_path=ft_model_path,
                    min_conf=max(0.0, float(settings.finetuned_qa_min_conf)),
                    top_chunks=max(1, int(settings.finetuned_qa_top_chunks)),
                    max_evidence=max(1, int(settings.finetuned_qa_max_evidence)),
                )
                if ft_evidences:
                    ft_used = True
                    ft_conf_max = max(item["confidence"] for item in ft_evidences)
            except Exception as exc:
                logger.warning("finetuned evidence extraction failed: %s", exc)
        else:
            ft_model_source = "none"

    history_text = _format_history(payload.history)
    scope_text = (
        f"检索范围：优先课程ID={selected_course_id}，必要时可参考其他课程。\n"
        if selected_course_id is not None
        else "检索范围：未指定课程，已在可访问课程中综合检索。\n"
    )
    if question_domain:
        scope_text += f"问题领域识别：{_domain_label(question_domain)}。\n"
    if domain_switch_hint:
        scope_text += f"课程匹配提示：{domain_switch_hint}\n"
    if docs:
        # 命中资料时，要求按“证据优先”格式作答，增强可追溯性。
        context = _build_context_blocks(docs, material_map, course_map)
        ft_evidence_text = _build_finetuned_evidence_block(ft_evidences)
        anchor_text = f"追问主题：{topic_anchor}\n" if topic_anchor else ""
        prompt = (
            "你是高校课程助教，请优先依据课程资料回答。\n"
            "输出格式（严格遵守）：\n"
            "第1段：先给1-2句总体定义/结论。\n"
            "空一行后，输出“主要内容包括：”。\n"
            "再输出2-4个编号小节（1. 2. 3. ...），每个小节下给1-3个分点。\n"
            "分点格式：- **关键词**：解释说明。\n"
            "允许使用**加粗**突出关键词；禁止使用#标题符号。\n"
            "要求：中文，180-420字，术语准确，不要编造资料中不存在的事实。\n"
            "若资料明显不足，请写“资料不足以确认”，并说明需要补充哪些课程资料。\n\n"
            f"{scope_text}"
            f"{anchor_text}"
            "如果提供了“微调抽取证据”，优先引用这些证据并标注对应[Rn]，"
            "再结合课程资料补充完整解释。\n"
            f"{ft_evidence_text}\n\n"
            f"课程资料片段：\n{context}\n\n"
            f"{history_text}"
            f"问题：{normalized_question}\n"
            "请开始作答。"
        )
        source_type = 0
    else:
        anchor_text = f"追问主题：{topic_anchor}\n" if topic_anchor else ""
        prompt = (
            "你是高校课程助教。当前未检索到可用课程资料。\n"
            "先明确说明“资料不足以确认”。\n"
            "再输出“可先参考以下回答：”，对问题进行回答，先给1-2句总体定义/结论。再给出若干个编号小节，每个小节若干个分点。\n"
            "分点格式：- **关键词**：解释说明。\n"
            "最后给出下一步建议。\n"
            "允许使用**加粗**突出关键词；禁止使用#标题符号。\n"
            "总字数控制在120-280字。\n\n"
            f"{scope_text}"
            f"{anchor_text}"
            f"{history_text}"
            f"问题：{normalized_question}"
        )
        source_type = 1

    provider = settings.model_provider.lower()
    glm_key = (getattr(settings, "glm_api_key", None) or "").strip()
    ernie_key = (getattr(settings, "ernie_api_key", None) or "").strip()
    dify_key = (getattr(settings, "dify_api_key", None) or "").strip()
    placeholder_keys = {"your_glm_key", "your_ernie_key", "your_dify_key"}
    has_llm_key = (
        (provider == "glm" and glm_key and glm_key not in placeholder_keys)
        or (provider == "ernie" and ernie_key and ernie_key not in placeholder_keys)
        or (provider == "dify" and dify_key and dify_key not in placeholder_keys)
    )

    if docs and has_llm_key:
        mode = "rag_llm_ft" if ft_used else "rag_llm"
        if rule_answer and rule_source == "direct" and not _is_followup_question(normalized_question) and not ft_used:
            answer = _expand_rule_answer(rule_answer, normalized_question)
            failure_reason = "rule_direct_shortcut"
        else:
            llm_answer = await call_llm(prompt)
            if _llm_failed(llm_answer):
                # 在线模型失败时切到证据/规则兜底，保证接口可用。
                failure_reason = "llm_failed_with_docs"
                if ft_evidences:
                    mode = "rag_fallback_ft"
                    answer = _build_finetuned_fallback_answer(ft_evidences, references)
                    failure_reason = "llm_failed_with_docs_use_ft"
                elif rule_answer:
                    answer = _expand_rule_answer(rule_answer, topic_anchor or normalized_question)
                    failure_reason = f"llm_failed_with_docs_use_rule_{rule_source}"
                else:
                    mode = "rag_fallback"
                    answer = _build_fallback_answer(normalized_question, docs, references)
            else:
                answer = _normalize_terminology(_sanitize_llm_answer(llm_answer or "", docs_present=True))
                quality_hint = topic_anchor or retrieval_question or normalized_question
                if _is_low_quality_doc_answer(normalized_question, answer, topic_hint=quality_hint):
                    if ft_evidences:
                        mode = "rag_fallback_ft"
                        answer = _build_finetuned_fallback_answer(ft_evidences, references)
                        failure_reason = "llm_low_quality_with_docs_use_ft"
                    elif rule_answer:
                        answer = _expand_rule_answer(rule_answer, quality_hint)
                        failure_reason = f"llm_low_quality_with_docs_use_rule_{rule_source}"
                    else:
                        failure_reason = "llm_low_quality_with_docs"
                        mode = "rag_fallback"
                        answer = _build_fallback_answer(normalized_question, docs, references)
                elif not answer:
                    if ft_evidences:
                        mode = "rag_fallback_ft"
                        answer = _build_finetuned_fallback_answer(ft_evidences, references)
                        failure_reason = "llm_empty_with_docs_use_ft"
                    elif rule_answer:
                        answer = _expand_rule_answer(rule_answer, quality_hint)
                        failure_reason = f"llm_empty_with_docs_use_rule_{rule_source}"
                    else:
                        failure_reason = "llm_empty_with_docs"
                        mode = "rag_fallback"
                        answer = _build_fallback_answer(normalized_question, docs, references)
    elif not docs and has_llm_key:
        if domain_switch_hint:
            mode = "rag_fallback"
            answer = f"资料不足以确认。\n\n{domain_switch_hint}"
            failure_reason = "course_domain_mismatch_guard"
        elif rule_answer and (
            (rule_source == "direct" and not _is_followup_question(normalized_question)) or rule_source == "anchor"
        ):
            mode = "rag_extension"
            answer = _expand_rule_answer(rule_answer, topic_anchor or normalized_question)
            failure_reason = f"no_docs_use_rule_{rule_source}"
        else:
            mode = "rag_extension"
            llm_answer = await call_llm(prompt)
            if _llm_failed(llm_answer):
                failure_reason = "llm_failed_without_docs"
                if rule_answer:
                    answer = _expand_rule_answer(rule_answer, topic_anchor or normalized_question)
                    failure_reason = f"llm_failed_without_docs_use_rule_{rule_source}"
                else:
                    mode = "rag_fallback"
                    answer = "[Fallback] 资料不足以确定。当前未检索到相关课程资料，请先上传资料或细化问题。"
            else:
                llm_text = _sanitize_llm_answer(llm_answer or "", docs_present=False)
                if not llm_text:
                    if rule_answer:
                        answer = _expand_rule_answer(rule_answer, topic_anchor or normalized_question)
                        failure_reason = f"llm_empty_without_docs_use_rule_{rule_source}"
                    else:
                        failure_reason = "llm_empty_without_docs"
                        mode = "rag_fallback"
                        answer = "[Fallback] 资料不足以确定。当前未检索到相关课程资料，请先上传资料或细化问题。"
                else:
                    if _is_followup_insufficient_answer(llm_text):
                        answer = llm_text
                    else:
                        answer = (
                            "资料不足以确认。\n\n"
                            "【课程外补充】\n"
                            f"{_clip_text(llm_text, 180)}\n\n"
                            "建议：可补充该课程讲义/PPT后再次提问。"
                        )
    elif not has_llm_key:
        failure_reason = "llm_key_missing"
        mode = "rag_fallback"
        if docs:
            if ft_evidences:
                answer = _build_finetuned_fallback_answer(ft_evidences, references)
                mode = "rag_fallback_ft"
            else:
                answer = _build_fallback_answer(normalized_question, docs, references)
                if settings.enable_small_qa_assist:
                    qa_model_path, _qa_source = resolve_active_qa_model_path(db)
                    if qa_model_path:
                        try:
                            qa_answer, qa_conf = predict_answer(normalized_question, docs[0][0], qa_model_path)
                            if qa_answer:
                                confidence_part = f" (score={qa_conf:.3f})" if qa_conf is not None else ""
                                answer = f"[SmallQA]{confidence_part} {qa_answer}\n" + answer
                                mode = "rag_small_qa"
                        except Exception:
                            pass
        else:
            if domain_switch_hint:
                answer = f"[Fallback] 资料不足以确定。\n\n{domain_switch_hint}"
            else:
                answer = "[Fallback] 资料不足以确定。当前未检索到相关课程资料，请先上传资料或细化问题。"
    else:
        failure_reason = "unexpected_branch"
        mode = "rag_fallback"
        if docs:
            answer = _build_fallback_answer(normalized_question, docs, references)
        else:
            if domain_switch_hint:
                answer = f"[Fallback] 资料不足以确定。\n\n{domain_switch_hint}"
            else:
                answer = "[Fallback] 资料不足以确定。当前未检索到相关课程资料，请先上传资料或细化问题。"

    inferred_course_id = _safe_int((docs[0][1] or {}).get("course_id")) if docs else None
    record_course_id = selected_course_id or inferred_course_id or target_course_ids[0]
    db.add(
        models.QARecord(
            user_id=user.id,
            course_id=record_course_id,
            question=payload.question,
            answer=answer,
            source_type=source_type,
        )
    )
    db.add(
        models.LearningEvent(
            user_id=user.id,
            course_id=record_course_id,
            event_type="qa",
            content=payload.question,
        )
    )
    db.commit()

    latency_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "qa_decision course_id=%s user_id=%s provider=%s mode=%s docs=%s refs=%s ft_used=%s ft_model=%s ft_source=%s ft_evidence=%s ft_conf_max=%s failure_reason=%s q_domain=%s selected_domain=%s followup_anchor=%s retrieval_q=%s latency_ms=%.1f",
        course_scope_key,
        user.id,
        provider,
        mode,
        len(docs),
        len(references),
        ft_used,
        ft_model_path or "-",
        ft_model_source,
        len(ft_evidences),
        f"{ft_conf_max:.3f}" if ft_conf_max is not None else "-",
        failure_reason,
        question_domain or "-",
        selected_course_domain or "-",
        topic_anchor or "-",
        (retrieval_question or normalized_question)[:80],
        latency_ms,
    )

    result = {
        "answer": answer,
        "source_type": source_type,
        "mode": mode,
        "references": references,
    }
    if cache_key:
        qa_cache[cache_key] = result
        if len(qa_cache) > CACHE_MAX:
            qa_cache.popitem(last=False)
    return result
