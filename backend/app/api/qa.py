from collections import OrderedDict
import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.config import settings
from app.db import get_db
from app.llm.client import call_llm
from app.models import models
from app.rag.pipeline import EmbeddingModelUnavailable, retrieve
from app.services.qa_small_model import predict_answer
from app.schemas.schemas import QARequest, QAResponse

router = APIRouter()
CACHE_MAX = 100
qa_cache: OrderedDict[str, dict] = OrderedDict()

ZH_WHAT_IS = "\u4ec0\u4e48\u662f"
ZH_QMARK = "\uFF1F"
ZH_DEFINITION_TAGS = ["\u662f", "\u6307", "\u8868\u793a", "\u5b9a\u4e49", "\u7528\u4e8e", "\u7279\u70b9", "\u533a\u522b"]
TOKEN_SPLIT_PATTERN = re.compile(r"[^\w\u4e00-\u9fff]+")
SINGLE_CHAR_TERMS = {"\u6811", "\u6808", "\u56fe", "\u5806"}
QUESTION_NOISE_TOKENS = [
    "\u4ec0\u4e48\u662f",
    "\u89e3\u91ca\u4e00\u4e0b",
    "\u89e3\u91ca",
    "\u4ecb\u7ecd",
    "\u8bf4\u660e",
    "\u4e3a\u4ec0\u4e48",
    "\u600e\u4e48",
    "\u5982\u4f55",
    "\u662f\u4ec0\u4e48",
    "\u4f1a\u53d1\u751f",
    "\u57fa\u672c\u601d\u60f3",
    "\u6838\u5fc3\u7279\u6027",
    "\u7279\u70b9",
    "\u5b9a\u4e49",
    "\u533a\u522b",
    "\u4e0d\u540c",
    "\uFF1F",
    "?",
]
GENERIC_TERMS = {
    "\u7279\u70b9",
    "\u5b9a\u4e49",
    "\u533a\u522b",
    "\u4e0d\u540c",
    "\u539f\u56e0",
    "\u4f5c\u7528",
    "\u6548\u679c",
    "\u57fa\u672c\u601d\u60f3",
    "\u6838\u5fc3\u7279\u6027",
    "\u4f1a\u53d1\u751f",
}
NOISY_LINE_PATTERNS = [
    re.compile(r"^\s*#"),
    re.compile(r"^\s*【[^】]*】\s*$"),
    re.compile(r"^\s*[-=_]{3,}\s*$"),
    re.compile(r"^\s*(一句话总结|知识点|问题|答案)[:：]?\s*$"),
]


def _is_noisy_line(line: str) -> bool:
    if not line:
        return True
    if len(line) < 8:
        return True
    for pattern in NOISY_LINE_PATTERNS:
        if pattern.search(line):
            return True
    # Skip short heading-style lines, e.g. "数组和链表的区别".
    if len(line) <= 18 and re.search(r"(区别|定义|特点|是什么)$", line) and not re.search(r"[，。；：]", line):
        return True
    return False


def _normalize_candidate_line(line: str) -> str:
    text = line.strip()
    # Strip bracketed labels like "【知识点】"
    text = re.sub(r"^\s*【[^】]*】\s*", "", text)
    # Strip summary prefixes
    text = re.sub(r"^\s*一句话总结[:：]?\s*", "", text)
    return text.strip()


def _question_keywords(question: str) -> list[str]:
    q = question.strip()
    for token in QUESTION_NOISE_TOKENS:
        q = q.replace(token, " ")
    def _keep_term(term: str) -> bool:
        return len(term) >= 2 or term in SINGLE_CHAR_TERMS

    base: list[str] = []
    for raw in TOKEN_SPLIT_PATTERN.split(q):
        term = raw.strip()
        if not term:
            continue
        if "\u7684" in term:
            base.extend([p.strip() for p in term.split("\u7684") if p.strip()])
        else:
            base.append(term)
    base = [t for t in base if _keep_term(t) and t not in GENERIC_TERMS]

    keywords: list[str] = []
    for term in base:
        keywords.append(term)
        for sep in ("\u4e0e", "\u548c", "\u53ca", "\u3001"):
            if sep in term:
                keywords.extend([p.strip() for p in term.split(sep) if _keep_term(p.strip())])

    out: list[str] = []
    seen = set()
    for term in keywords:
        if term not in seen:
            out.append(term)
            seen.add(term)
    return out


def _definition_candidates(question: str, docs: list[tuple[str, dict]], limit: int = 2) -> list[str]:
    keywords = _question_keywords(question)
    scored_with_kw: list[tuple[int, str]] = []
    scored_general: list[tuple[int, str]] = []
    for doc, _meta in docs:
        for raw_line in doc.splitlines():
            line = _normalize_candidate_line(raw_line.strip())
            if _is_noisy_line(line):
                continue
            # Skip plain question-title style lines.
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


def _latest_qa_model_path(db: Session) -> str | None:
    run = (
        db.query(models.ModelRun)
        .filter(models.ModelRun.task_type == "qa_extractive_hf", models.ModelRun.status == "success")
        .order_by(models.ModelRun.updated_at.desc())
        .first()
    )
    return run.model_path if run else None


def _course_cache_version(db: Session, course_id: int) -> str:
    latest_parsed = (
        db.query(func.max(models.Material.parsed_at))
        .filter(
            models.Material.course_id == course_id,
            models.Material.parse_status == "success",
        )
        .scalar()
    )
    return latest_parsed.isoformat() if latest_parsed else "none"


def clear_qa_cache(course_id: int | None = None) -> None:
    if course_id is None:
        qa_cache.clear()
        return

    prefix = f"{course_id}::"
    stale_keys = [key for key in qa_cache.keys() if key.startswith(prefix)]
    for key in stale_keys:
        qa_cache.pop(key, None)


@router.post("/", response_model=QAResponse, summary="Course QA")
async def ask(payload: QARequest, db: Session = Depends(get_db), user=Depends(get_current_user)):
    course = db.query(models.Course).get(payload.course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    cache_version = _course_cache_version(db, payload.course_id)
    cache_key = f"{payload.course_id}::{cache_version}::{payload.question.strip()}"
    if cache_key in qa_cache:
        cached = qa_cache.pop(cache_key)
        qa_cache[cache_key] = cached
        return cached

    try:
        docs = retrieve(payload.course_id, payload.question, top_k=5)
    except EmbeddingModelUnavailable:
        docs = []

    references = [
        f"chunk_{meta.get('material_id')}_{meta.get('chunk_id')}"
        for _, meta in docs
        if meta is not None
    ]

    if docs:
        context = "\n\n".join([f"[Chunk {i}] {doc}" for i, (doc, _) in enumerate(docs)])
        prompt = (
            "You are a teaching assistant. Answer strictly based on provided course materials.\n"
            f"{context}\n\n"
            f"Question: {payload.question}\n"
            "Return a short answer and cite chunk ids."
        )
        source_type = 0
    else:
        prompt = (
            "You are a teaching assistant. No course context was found. "
            "Give a short extension answer and mark it as extension knowledge.\n"
            f"Question: {payload.question}"
        )
        source_type = 1

    glm_key = (getattr(settings, "glm_api_key", None) or "").strip()
    ernie_key = (getattr(settings, "ernie_api_key", None) or "").strip()
    placeholder_keys = {"your_glm_key", "your_ernie_key"}
    has_llm_key = (glm_key and glm_key not in placeholder_keys) or (ernie_key and ernie_key not in placeholder_keys)

    if not has_llm_key:
        mode = "rag_fallback"
        if docs:
            candidates = _definition_candidates(payload.question, docs, limit=1)
            brief = candidates[0] if candidates else "No direct definition sentence found."
            first_ref = references[0] if references else "none"
            answer = "[Fallback] " + brief + "\n" + f"Reference: {first_ref}"

            if settings.enable_small_qa_assist:
                qa_model_path = _latest_qa_model_path(db)
                if qa_model_path:
                    try:
                        qa_answer, qa_conf = predict_answer(payload.question, docs[0][0], qa_model_path)
                        if qa_answer:
                            confidence_part = f" (score={qa_conf:.3f})" if qa_conf is not None else ""
                            answer = (
                                f"[SmallQA]{confidence_part} {qa_answer}\n"
                                + answer
                            )
                            mode = "rag_small_qa"
                    except Exception:
                        # Keep fallback answer if small QA model is unavailable.
                        pass
        else:
            answer = "[Fallback] No relevant course chunks found. Try another keyword or upload material."
    else:
        mode = "rag_llm"
        answer = await call_llm(prompt)

    db.add(
        models.QARecord(
            user_id=user.id,
            course_id=payload.course_id,
            question=payload.question,
            answer=answer,
            source_type=source_type,
        )
    )
    db.add(
        models.LearningEvent(
            user_id=user.id,
            course_id=payload.course_id,
            event_type="qa",
            content=payload.question,
        )
    )
    db.commit()

    result = {
        "answer": answer,
        "source_type": source_type,
        "mode": mode,
        "references": references,
    }
    qa_cache[cache_key] = result
    if len(qa_cache) > CACHE_MAX:
        qa_cache.popitem(last=False)
    return result
