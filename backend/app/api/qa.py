from collections import OrderedDict
import re

from fastapi import APIRouter, Depends, HTTPException
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


def _question_keywords(question: str) -> list[str]:
    q = question.strip()
    for token in [ZH_WHAT_IS, "\u89e3\u91ca\u4e00\u4e0b", "\u89e3\u91ca", "\u4ecb\u7ecd", "\u8bf4\u660e", ZH_QMARK, "?"]:
        q = q.replace(token, " ")
    base = [t.strip() for t in TOKEN_SPLIT_PATTERN.split(q) if len(t.strip()) >= 2]

    keywords: list[str] = []
    for term in base:
        keywords.append(term)
        for sep in ("\u4e0e", "\u548c", "\u53ca", "\u3001"):
            if sep in term:
                keywords.extend([p.strip() for p in term.split(sep) if len(p.strip()) >= 2])

    out: list[str] = []
    seen = set()
    for term in keywords:
        if term not in seen:
            out.append(term)
            seen.add(term)
    return out


def _definition_candidates(question: str, docs: list[tuple[str, dict]], limit: int = 2) -> list[str]:
    keywords = _question_keywords(question)

    candidates: list[str] = []
    for doc, _meta in docs:
        for raw_line in doc.splitlines():
            line = raw_line.strip()
            if not line or len(line) < 8:
                continue
            looks_like_definition = any(tag in line for tag in ZH_DEFINITION_TAGS)
            hit_keyword = any(k in line for k in keywords) if keywords else True
            if hit_keyword and looks_like_definition:
                candidates.append(line)
            if len(candidates) >= limit:
                return candidates

    # Second pass: any line containing keywords
    if not candidates and keywords:
        for doc, _meta in docs:
            for raw_line in doc.splitlines():
                line = raw_line.strip()
                if len(line) < 8:
                    continue
                if any(k in line for k in keywords):
                    candidates.append(line)
                    if len(candidates) >= limit:
                        return candidates

    if not candidates:
        for doc, _meta in docs:
            for raw_line in doc.splitlines():
                line = raw_line.strip()
                if len(line) >= 8:
                    candidates.append(line)
                    break
            if len(candidates) >= limit:
                break
    return candidates[:limit]


def _latest_qa_model_path(db: Session) -> str | None:
    run = (
        db.query(models.ModelRun)
        .filter(models.ModelRun.task_type == "qa_extractive_hf", models.ModelRun.status == "success")
        .order_by(models.ModelRun.updated_at.desc())
        .first()
    )
    return run.model_path if run else None


@router.post("/", response_model=QAResponse, summary="Course QA")
async def ask(payload: QARequest, db: Session = Depends(get_db), user=Depends(get_current_user)):
    course = db.query(models.Course).get(payload.course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    cache_key = f"{payload.course_id}::{payload.question.strip()}"
    if cache_key in qa_cache:
        cached = qa_cache.pop(cache_key)
        qa_cache[cache_key] = cached
        return cached

    try:
        docs = retrieve(payload.course_id, payload.question, top_k=3)
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
            candidates = _definition_candidates(payload.question, docs, limit=2)
            brief = " ; ".join(candidates) if candidates else "No direct definition sentence found."
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
