from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import unicodedata
from typing import Any, Awaitable, Callable, Literal, Protocol

from app.config import settings
from app.services.assignment_feedback_model import predict_feedback_quality
from app.services.assignment_relevance_model import (
    predict_relevance_probability,
    predict_reranker_probability,
)
from app.services.model_paths import (
    resolve_active_assignment_feedback_model_path,
    resolve_active_assignment_relevance_model_path,
)

try:
    from app.rag.pipeline import retrieve
except Exception:  # pragma: no cover
    retrieve = None

try:
    import jieba.posseg as jieba_posseg
except Exception:  # pragma: no cover
    jieba_posseg = None


logger = logging.getLogger(__name__)

RelevanceLabel = Literal["relevant", "ambiguous", "off_topic"]
DimensionLevel = Literal["good", "ok", "weak"]


class AssignmentLike(Protocol):
    title: str
    description: str | None
    keywords: str | None


@dataclass
class RelevanceResult:
    label: RelevanceLabel
    reason: str
    focus_terms: list[str]
    core_terms: list[str]
    matched_focus_terms: list[str]
    matched_core_terms: list[str]
    missing_focus_terms: list[str]
    coverage: float
    model_score: float | None = None
    model_source: str | None = None


@dataclass
class GradingDimension:
    name: str
    level: DimensionLevel
    summary: str


@dataclass
class StructuredGrading:
    dimensions: list[GradingDimension]
    strengths: list[str]
    issues: list[str]
    suggestions: list[str]
    risk_hint: str
    evidence_refs: list[str]
    model_signal: dict[str, Any] | None = None


_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{1,12}|[A-Za-z]{2,24}|[A-Za-z]+[0-9]+")
_TOKEN_CLEAN_RE = re.compile(r"^[\W_]+|[\W_]+$")
_KEYWORD_SPLIT_RE = re.compile(r"[,，、;；\s\n\r\t]+")
_GENERIC_TITLE_RE = re.compile(r"^(?:作业|练习|任务|实验)\s*[0-9一二三四五六七八九十]+$", re.IGNORECASE)
_TITLE_NOISE_TOKEN_RE = re.compile(
    r"^(?:test|hw|lab|task|lesson|unit|week|chapter)[_-]?\d+$|^(?:作业|练习|实验|任务|第?[0-9一二三四五六七八九十]+次?)$",
    re.IGNORECASE,
)
_QUESTION_PATTERNS = (
    re.compile(r"(?:什么是|何为|请解释|解释|简述|定义|说明)\s*([A-Za-z0-9\u4e00-\u9fff]{1,20})"),
    re.compile(r"([A-Za-z0-9\u4e00-\u9fff]{1,20})\s*(?:是什么|的定义|指的是什么)"),
)
_QUESTION_WORDS = {
    "什么",
    "如何",
    "为什么",
    "请",
    "请解释",
    "解释",
    "简述",
    "定义",
    "说明",
    "作业",
    "问题",
    "要求",
    "课程",
    "内容",
    "相关",
}
_STOPWORDS = {
    "什么",
    "如何",
    "为什么",
    "这个",
    "那个",
    "请",
    "请解释",
    "解释",
    "简述",
    "定义",
    "说明",
    "问题",
    "作业",
    "课程",
    "内容",
    "相关",
    "进行",
    "分析",
    "描述",
}
_LLM_FAILURE_MARKERS = (
    "接口错误",
    "占位答复",
    "timeout",
    "timed out",
    "error",
    "service unavailable",
)
_AMBIG_LABEL_RE = re.compile(r'"label"\s*:\s*"(relevant|off_topic)"', re.IGNORECASE)

# 清洗 LLM 输出中可能出现的分数信息
_SCORE_LINE_RE = re.compile(r"^(?:评分|得分|score)\s*[:：]\s*\d{1,3}\s*$", re.IGNORECASE)
_SCORE_INLINE_RE = re.compile(r"(?:评分|得分|score)\s*[:：]\s*\d{1,3}", re.IGNORECASE)
_FEEDBACK_PREFIX_RE = re.compile(r"^(?:评语|反馈)\s*[:：]\s*", re.IGNORECASE)
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*")
_MARKDOWN_BULLET_RE = re.compile(r"^\s*[-*+]\s+")
_SECTION_HEADER_RE = re.compile(r"(优点：|问题：|改进建议：)")
_OVER_SCOPE_HINTS = (
    "时间复杂度",
    "空间复杂度",
    "应用场景",
    "优缺点",
    "选择标准",
    "复杂度分析",
    "比较分析",
)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.lower().strip()
    return re.sub(r"\s+", "", normalized)


def _clean_term(token: str) -> str:
    cleaned = _TOKEN_CLEAN_RE.sub("", (token or "").strip())
    return _normalize_text(cleaned)


def _is_noise_term(term: str) -> bool:
    if not term:
        return True
    if term in {"ipv4", "ipv6"}:
        return False
    if term in _STOPWORDS or term in _QUESTION_WORDS:
        return True
    if _TITLE_NOISE_TOKEN_RE.fullmatch(term):
        return True
    if re.fullmatch(r"[a-z]{2,8}\d{1,3}", term, flags=re.IGNORECASE):
        return True
    if term.isdigit():
        return True
    if len(term) == 1 and not re.fullmatch(r"[a-zA-Z]", term):
        return False
    return len(term) < 2


def _tokenize_terms(text: str) -> list[str]:
    if not text:
        return []
    terms: list[str] = []
    if jieba_posseg is not None:
        try:
            for pair in jieba_posseg.cut(text):
                word = _clean_term(getattr(pair, "word", "") or "")
                flag = (getattr(pair, "flag", "") or "").lower()
                if word and flag.startswith(("n", "vn", "nz", "eng")):
                    terms.append(word)
        except Exception:
            terms = []
    if not terms:
        terms = [_clean_term(raw) for raw in _TOKEN_RE.findall(text)]

    output: list[str] = []
    for term in terms:
        if _is_noise_term(term):
            continue
        if term not in output:
            output.append(term)
    return output


def _parse_keywords(keywords_raw: str | None) -> list[str]:
    if not keywords_raw:
        return []
    output: list[str] = []
    for piece in _KEYWORD_SPLIT_RE.split(keywords_raw):
        token = _clean_term(piece)
        if _is_noise_term(token):
            continue
        if token not in output:
            output.append(token)
    return output


def _extract_core_terms(text: str) -> list[str]:
    if not text:
        return []
    found: list[str] = []
    for pattern in _QUESTION_PATTERNS:
        for match in pattern.finditer(text):
            raw = (match.group(1) or "").strip()
            candidates = [_clean_term(raw)] + _tokenize_terms(raw)
            for piece in re.split(r"[的和与及、，；\s]+", raw):
                token = _clean_term(piece)
                if token:
                    candidates.append(token)
            for token in candidates:
                if not token or token in _QUESTION_WORDS or len(token) > 10:
                    continue
                if token not in found:
                    found.append(token)
    return found


def _build_focus_terms(assignment: AssignmentLike) -> tuple[list[str], list[str]]:
    weighted: dict[str, float] = {}
    core_terms: list[str] = []

    desc = (assignment.description or "").strip()
    title = (assignment.title or "").strip()
    kw = _parse_keywords(assignment.keywords)

    for term in _extract_core_terms(desc) + _extract_core_terms(title):
        if term not in core_terms:
            core_terms.append(term)
        weighted[term] = weighted.get(term, 0.0) + 3.6

    for term in _tokenize_terms(desc):
        weighted[term] = weighted.get(term, 0.0) + 2.2

    for term in kw:
        weighted[term] = weighted.get(term, 0.0) + 1.8

    if title and not _GENERIC_TITLE_RE.fullmatch(title):
        for term in _tokenize_terms(title):
            weighted[term] = weighted.get(term, 0.0) + 1.0

    ranked = sorted(weighted.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    return [term for term, _ in ranked[:12]], core_terms


def _leading_defined_term(text: str) -> str:
    match = re.search(r"^\s*([A-Za-z0-9\u4e00-\u9fff]{1,12})\s*(?:是|指|表示)", text or "")
    if not match:
        return ""
    return _clean_term(match.group(1))


def _core_definition_hit(content_norm: str, core_terms: list[str]) -> bool:
    for term in core_terms:
        if term and (f"{term}是" in content_norm or f"{term}指" in content_norm or f"{term}表示" in content_norm):
            return True
    return False


def _evaluate_relevance_rule_only(assignment: AssignmentLike, content: str) -> RelevanceResult:
    focus_terms, core_terms = _build_focus_terms(assignment)
    content_norm = _normalize_text(content)

    if not content_norm:
        return RelevanceResult("off_topic", "empty_content", focus_terms, core_terms, [], [], focus_terms, 0.0)
    if not focus_terms:
        return RelevanceResult("ambiguous", "no_focus_terms", [], core_terms, [], [], [], 0.0)

    matched_focus = [t for t in focus_terms if t in content_norm]
    matched_core = [t for t in core_terms if t in content_norm]
    missing_focus = [t for t in focus_terms if t not in matched_focus]
    coverage = len(matched_focus) / max(1, len(focus_terms))

    leading = _leading_defined_term(content)
    definition_hit = _core_definition_hit(content_norm, core_terms)

    if matched_core and definition_hit:
        return RelevanceResult("relevant", "core_definition_hit", focus_terms, core_terms, matched_focus, matched_core, missing_focus, coverage)

    if core_terms and matched_core and leading and leading not in core_terms and not definition_hit and len(matched_core) <= 1:
        return RelevanceResult("off_topic", "leading_term_mismatch", focus_terms, core_terms, matched_focus, matched_core, missing_focus, coverage)

    if len(matched_focus) >= 2 and coverage >= 0.28:
        return RelevanceResult("relevant", "focus_hit", focus_terms, core_terms, matched_focus, matched_core, missing_focus, coverage)

    if core_terms and not matched_core and len(matched_focus) == 0 and coverage <= 0.10:
        return RelevanceResult("off_topic", "core_miss_low_coverage", focus_terms, core_terms, matched_focus, matched_core, missing_focus, coverage)

    return RelevanceResult("ambiguous", "borderline", focus_terms, core_terms, matched_focus, matched_core, missing_focus, coverage)


def _build_assignment_query_text(assignment: AssignmentLike) -> str:
    desc = (assignment.description or "").strip()
    kw = (assignment.keywords or "").strip()
    title = (assignment.title or "").strip()

    parts: list[str] = []
    if desc:
        parts.append(desc)
    if kw:
        parts.append(f"关键词：{kw}")
    if title and not _GENERIC_TITLE_RE.fullmatch(title):
        title_terms = [t for t in _tokenize_terms(title) if not _TITLE_NOISE_TOKEN_RE.fullmatch(t)]
        if title_terms:
            parts.append(f"标题：{'、'.join(title_terms[:6])}")
    if not parts and title:
        parts.append(title)
    return "\n".join(parts)


def _score_with_assignment_relevance_model(assignment: AssignmentLike, content: str, db=None) -> tuple[float | None, str | None]:
    if not settings.enable_assignment_relevance_model:
        return None, None

    query = _build_assignment_query_text(assignment)
    if not query or not (content or "").strip():
        return None, None

    model_path, model_source = resolve_active_assignment_relevance_model_path(db)
    if not model_path:
        return None, None

    base_score = predict_relevance_probability(query, content, model_path=model_path, max_length=384)
    blended = float(base_score)

    if settings.assignment_relevance_use_reranker:
        reranker_name = (settings.assignment_relevance_reranker_model or "").strip()
        if reranker_name:
            try:
                rerank = predict_reranker_probability(query, content, model_name=reranker_name, max_length=512)
                weight = min(max(float(settings.assignment_relevance_reranker_weight), 0.0), 0.8)
                blended = (1.0 - weight) * blended + weight * rerank
            except Exception as exc:
                logger.warning("assignment reranker scoring failed: %s", exc)

    return min(max(blended, 0.0), 1.0), model_source


def _label_from_score(score: float, threshold_hi: float, threshold_lo: float) -> RelevanceLabel:
    if score >= threshold_hi:
        return "relevant"
    if score <= threshold_lo:
        return "off_topic"
    return "ambiguous"


def evaluate_relevance(assignment: AssignmentLike, content: str, db=None) -> RelevanceResult:
    rule = _evaluate_relevance_rule_only(assignment, content)

    try:
        model_score, model_source = _score_with_assignment_relevance_model(assignment, content, db=db)
    except Exception as exc:
        logger.warning("assignment relevance model unavailable: %s", exc)
        return rule

    if model_score is None:
        return rule

    hi = min(max(float(settings.assignment_relevance_threshold_hi), 0.0), 1.0)
    lo = min(max(float(settings.assignment_relevance_threshold_lo), 0.0), 1.0)
    if lo > hi:
        lo, hi = hi, lo

    model_label = _label_from_score(model_score, hi, lo)

    if model_label == "ambiguous":
        final_label = rule.label
    elif model_label == "relevant":
        final_label = "ambiguous" if rule.label == "off_topic" else "relevant"
    else:
        final_label = "ambiguous" if rule.label == "relevant" else "off_topic"

    return RelevanceResult(
        label=final_label,
        reason=f"model_score={model_score:.3f}|model_label={model_label}|rule={rule.reason}",
        focus_terms=rule.focus_terms,
        core_terms=rule.core_terms,
        matched_focus_terms=rule.matched_focus_terms,
        matched_core_terms=rule.matched_core_terms,
        missing_focus_terms=rule.missing_focus_terms,
        coverage=rule.coverage,
        model_score=model_score,
        model_source=model_source,
    )


def _llm_failed(text: str) -> bool:
    normalized = (text or "").strip()
    if not normalized:
        return True
    lowered = normalized.lower()
    return any(marker in lowered for marker in _LLM_FAILURE_MARKERS)


def parse_llm_feedback(response: str) -> str:
    text = (response or "").strip()
    if not text or _llm_failed(text):
        return ""

    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("```"):
            continue
        line = _MARKDOWN_HEADING_RE.sub("", line)
        line = _MARKDOWN_BULLET_RE.sub("", line)
        line = line.replace("**", "").replace("__", "").replace("`", "")
        if _SCORE_LINE_RE.search(line):
            continue
        line = _SCORE_INLINE_RE.sub("", line).strip()
        line = line.replace("作业批改评语", "").strip()
        line = _FEEDBACK_PREFIX_RE.sub("", line)
        if line:
            lines.append(line)

    cleaned = "\n".join(lines).strip()
    cleaned = _SCORE_INLINE_RE.sub("", cleaned).strip()
    cleaned = _FEEDBACK_PREFIX_RE.sub("", cleaned).strip()
    cleaned = _SECTION_HEADER_RE.sub(r"\n\1", cleaned).strip()
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    if cleaned.startswith("："):
        cleaned = cleaned[1:].strip()
    return cleaned


def _scope_has_requirement(assignment: AssignmentLike, hint: str) -> bool:
    scope_text = " ".join(
        [
            (assignment.title or "").lower(),
            (assignment.description or "").lower(),
            (assignment.keywords or "").lower(),
        ]
    )
    return hint.lower() in scope_text


def _enforce_feedback_scope(assignment: AssignmentLike, feedback: str, structured: StructuredGrading) -> str:
    lines = [line.strip() for line in (feedback or "").splitlines() if line.strip()]
    if not lines:
        return feedback

    drop_hints = [hint for hint in _OVER_SCOPE_HINTS if (hint in feedback and not _scope_has_requirement(assignment, hint))]
    if not drop_hints:
        return feedback

    kept_lines: list[str] = []
    for line in lines:
        if line.startswith("问题：") or line.startswith("改进建议："):
            if any(h in line for h in drop_hints):
                continue
        kept_lines.append(line)

    if not any(x.startswith("问题：") for x in kept_lines):
        issue = "；".join(structured.issues[:1]) or "答案与题干要求对齐还不够充分。"
        kept_lines.append(f"问题：{issue}")
    if not any(x.startswith("改进建议：") for x in kept_lines):
        suggestion = "；".join(structured.suggestions[:2]) or "围绕题干要求补充核心定义与关键要点。"
        kept_lines.append(f"改进建议：{suggestion}")

    normalized = "\n".join(kept_lines)
    normalized = re.sub(r"\n{2,}", "\n", normalized).strip()
    return normalized


def _mode() -> str:
    mode = (getattr(settings, "assignment_feedback_mode", "legacy") or "legacy").strip().lower()
    if mode in {"legacy", "shadow", "v2"}:
        return mode
    return "legacy"


def _shadow_log_path() -> Path:
    value = (getattr(settings, "assignment_feedback_shadow_log_path", "") or "").strip()
    return Path(value) if value else Path("logs/assignment_feedback_shadow.jsonl")


def _append_shadow_log(payload: dict[str, Any]) -> None:
    try:
        path = _shadow_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover
        logger.warning("assignment feedback shadow log write failed: %s", exc)


def _assignment_course_id(assignment: AssignmentLike) -> int | None:
    raw = getattr(assignment, "course_id", None)
    try:
        return int(raw) if raw is not None else None
    except Exception:
        return None


def _retrieve_evidence(assignment: AssignmentLike, content: str, top_k: int = 4) -> list[dict[str, Any]]:
    if retrieve is None:
        return []

    course_id = _assignment_course_id(assignment)
    if course_id is None:
        return []

    query = _build_assignment_query_text(assignment)
    if (content or "").strip():
        query = f"{query}\n学生答案：{content[:220]}"
    if not query.strip():
        return []

    try:
        docs = retrieve(course_id, query, top_k=top_k)
    except Exception as exc:
        logger.warning("assignment evidence retrieve failed: %s", exc)
        return []

    rows: list[dict[str, Any]] = []
    for idx, (doc, meta) in enumerate(docs[:top_k], start=1):
        text = re.sub(r"\s+", " ", str(doc or "")).strip()
        if not text:
            continue
        meta = meta or {}
        rows.append(
            {
                "ref": f"R{idx}",
                "text": text[:220],
                "material_id": meta.get("material_id"),
                "chunk_id": meta.get("chunk_id"),
            }
        )
    return rows


def _dedupe_keep_order(items: list[str]) -> list[str]:
    output: list[str] = []
    for item in items:
        text = (item or "").strip()
        if not text:
            continue
        if text not in output:
            output.append(text)
    return output


def _downgrade_level(level: DimensionLevel, target: DimensionLevel) -> DimensionLevel:
    order = {"weak": 0, "ok": 1, "good": 2}
    return target if order[target] < order[level] else level


def _grade_structured(content: str, relevance: RelevanceResult, evidence: list[dict[str, Any]]) -> StructuredGrading:
    length = len((content or "").strip())
    sentence_count = len(re.findall(r"[。！？!?]", content or ""))
    dimensions: list[GradingDimension] = []

    if relevance.label == "relevant":
        dimensions.append(GradingDimension("概念正确性", "good", "核心概念方向基本正确。"))
    elif relevance.label == "ambiguous":
        dimensions.append(GradingDimension("概念正确性", "ok", "概念方向部分相关，但重心不够稳定。"))
    else:
        dimensions.append(GradingDimension("概念正确性", "weak", "概念方向偏离题干。"))

    if relevance.coverage >= 0.45:
        dimensions.append(GradingDimension("要点覆盖", "good", "关键要点覆盖较完整。"))
    elif relevance.coverage >= 0.20:
        dimensions.append(GradingDimension("要点覆盖", "ok", "覆盖了部分关键要点。"))
    else:
        dimensions.append(GradingDimension("要点覆盖", "weak", "关键要点覆盖不足。"))

    core_hits = len(relevance.matched_core_terms)
    focus_hits = len(relevance.matched_focus_terms)
    if core_hits >= 1 and focus_hits >= 3:
        dimensions.append(GradingDimension("术语准确性", "good", "术语使用较准确。"))
    elif focus_hits >= 2:
        dimensions.append(GradingDimension("术语准确性", "ok", "术语有覆盖，但准确性一般。"))
    else:
        dimensions.append(GradingDimension("术语准确性", "weak", "课程术语命中较少。"))

    if length >= 80 and sentence_count >= 2:
        dimensions.append(GradingDimension("表达清晰度", "good", "表达清晰且结构较完整。"))
    elif length >= 30 and sentence_count >= 1:
        dimensions.append(GradingDimension("表达清晰度", "ok", "表达可读，但结构还可加强。"))
    else:
        dimensions.append(GradingDimension("表达清晰度", "weak", "表达过于简略。"))

    strengths = [d.summary for d in dimensions if d.level == "good"]
    if not strengths and focus_hits > 0:
        strengths = ["已覆盖部分课程术语，具备继续完善的基础。"]

    issues = [d.summary for d in dimensions if d.level == "weak"]
    if relevance.label != "relevant":
        issues.append("回答与题干主问法存在偏差。")

    suggestions: list[str] = []
    missing = relevance.missing_focus_terms[:3]
    if missing:
        suggestions.append(f"围绕“{'、'.join(missing)}”补充定义和关键特征。")
    suggestions.append("按“定义 -> 要点 -> 示例”三步重写答案。")
    suggestions.append("使用课程术语并结合课程资料中的证据来支撑结论。")

    risk_hint = ""
    if relevance.label == "off_topic":
        risk_hint = "偏题风险较高：当前回答未对齐题干核心概念。"
    elif relevance.label == "ambiguous":
        risk_hint = "存在偏题风险：建议先明确题干概念再展开。"

    refs = [str(item.get("ref")) for item in evidence if item.get("ref")]
    return StructuredGrading(
        dimensions=dimensions,
        strengths=_dedupe_keep_order(strengths),
        issues=_dedupe_keep_order(issues),
        suggestions=_dedupe_keep_order(suggestions)[:3],
        risk_hint=risk_hint,
        evidence_refs=refs,
    )


def _reference_answer_from_evidence(assignment: AssignmentLike, evidence: list[dict[str, Any]]) -> str:
    snippets = [str(item.get("text") or "").strip() for item in evidence[:3]]
    snippets = [x for x in snippets if x]
    if snippets:
        return " ".join(snippets)
    keyword_text = (assignment.keywords or "").strip()
    return keyword_text or (assignment.description or "").strip()


def _apply_feedback_model_signal(
    assignment: AssignmentLike,
    content: str,
    structured: StructuredGrading,
    evidence: list[dict[str, Any]],
    db=None,
) -> None:
    if not settings.enable_assignment_feedback_model:
        return

    model_path, model_source = resolve_active_assignment_feedback_model_path(db)
    if not model_path:
        return

    question = (assignment.description or "").strip() or assignment.title
    student_answer = (content or "").strip()
    if not question or not student_answer:
        return
    reference_answer = _reference_answer_from_evidence(assignment, evidence)

    try:
        quality, confidence = predict_feedback_quality(
            question=question,
            reference_answer=reference_answer,
            student_answer=student_answer,
            model_path=model_path,
            max_length=384,
        )
    except Exception as exc:
        logger.warning("assignment feedback model inference failed: %s", exc)
        return

    structured.model_signal = {
        "quality": quality,
        "confidence": round(float(confidence), 4),
        "model_source": model_source,
        "model_path": model_path,
    }

    # 模型只做辅助信号，不直接替代规则判断
    if quality == "weak":
        structured.issues.append("模型判定：答案整体质量偏弱，存在关键概念或要点缺失。")
        structured.suggestions.append("先围绕题干给出准确定义，再补充2-3个关键要点。")
        for dim in structured.dimensions:
            if dim.name in {"概念正确性", "要点覆盖"}:
                dim.level = _downgrade_level(dim.level, "ok")
    elif quality == "good" and confidence >= 0.55:
        structured.strengths.append("模型判定：答案与题目语义较一致，基础较好。")

    structured.strengths = _dedupe_keep_order(structured.strengths)
    structured.issues = _dedupe_keep_order(structured.issues)
    structured.suggestions = _dedupe_keep_order(structured.suggestions)[:3]


def _build_legacy_feedback(assignment: AssignmentLike, relevance: RelevanceResult, llm_reason: str) -> str:
    missing_terms = [t for t in relevance.missing_focus_terms if not _TITLE_NOISE_TOKEN_RE.fullmatch(t)]
    matched_terms = [t for t in relevance.matched_focus_terms if not _TITLE_NOISE_TOKEN_RE.fullmatch(t)]
    missing = "、".join(missing_terms[:4]) or "题目核心知识点"
    matched = "、".join(matched_terms[:3])
    reason_text = f"判定依据：{llm_reason}。" if llm_reason else ""
    matched_text = f"已覆盖：{matched}。" if matched else ""
    return (
        f"你的回答与题目《{assignment.title}》的要求不一致。{reason_text}{matched_text}"
        f"当前答案未围绕“{missing}”展开。\n"
        "建议按以下结构重写：\n"
        "1) 先给出题目概念的准确定义；\n"
        "2) 补充关键特征/组成；\n"
        "3) 结合课程术语给出简短示例或对比。"
    )


def _build_relevant_prompt(assignment: AssignmentLike, content: str, relevance: RelevanceResult) -> str:
    return (
        "你是高校课程助教。请生成作业批改评语。\n"
        "要求：必须包含“优点、问题、改进建议(至少2条)”，不要给分数。\n"
        f"题目：{assignment.title}\n"
        f"描述：{assignment.description or '无'}\n"
        f"关键词：{assignment.keywords or '无'}\n"
        f"命中术语：{','.join(relevance.matched_focus_terms) or '无'}\n"
        f"学生答案：{content}\n"
    )


async def _resolve_ambiguous_with_llm(
    assignment: AssignmentLike,
    content: str,
    relevance: RelevanceResult,
    llm_call: Callable[[str], Awaitable[str]],
) -> tuple[RelevanceLabel, str]:
    prompt = (
        "你是作业相关性判定器，请判断学生回答是否围绕题目。\n"
        "注意：题目名称里如果包含 test1、作业1 等编号词，不能据此判偏题。\n"
        "仅输出JSON：{\"label\":\"relevant|off_topic\",\"reason\":\"一句话\"}\n"
        f"题目：{assignment.title}\n"
        f"描述：{assignment.description or '无'}\n"
        f"命中术语：{','.join(relevance.matched_focus_terms) or '无'}\n"
        f"缺失术语：{','.join(relevance.missing_focus_terms[:6]) or '无'}\n"
        f"答案：{content}\n"
    )

    llm_output = (await llm_call(prompt)).strip()
    if _llm_failed(llm_output):
        return "relevant", "llm_unavailable_fallback_relevant"

    try:
        payload_match = re.search(r"\{.*\}", llm_output, flags=re.DOTALL)
        payload = json.loads(payload_match.group(0) if payload_match else llm_output)
        label = str(payload.get("label", "")).strip().lower()
        reason = str(payload.get("reason", "")).strip()
        if label in {"relevant", "off_topic"}:
            return label, reason or "llm_json_result"
    except Exception:
        pass

    keyword_match = _AMBIG_LABEL_RE.search(llm_output)
    if keyword_match and keyword_match.group(1).lower() in {"relevant", "off_topic"}:
        return keyword_match.group(1).lower(), "llm_regex_result"

    if "偏题" in llm_output or "不相关" in llm_output:
        return "off_topic", "llm_keyword_off_topic"

    return "relevant", "llm_fallback_relevant"


async def _generate_legacy_feedback(
    assignment: AssignmentLike,
    content: str,
    llm_call: Callable[[str], Awaitable[str]],
    db=None,
) -> tuple[str, dict[str, Any]]:
    relevance = evaluate_relevance(assignment, content, db=db)
    final_label = relevance.label
    llm_reason = ""

    if relevance.label == "ambiguous":
        final_label, llm_reason = await _resolve_ambiguous_with_llm(assignment, content, relevance, llm_call)

    if final_label == "off_topic":
        return _build_legacy_feedback(assignment, relevance, llm_reason), {"final_label": final_label, "relevance": asdict(relevance)}

    llm_output = await llm_call(_build_relevant_prompt(assignment, content, relevance))
    feedback = parse_llm_feedback(llm_output)
    if feedback:
        return feedback, {"final_label": final_label, "relevance": asdict(relevance)}

    fallback = "优点：回答覆盖了部分题干术语。\n问题：论述不够完整。\n改进建议：按定义、要点、示例三步补充，并使用课程术语。"
    return fallback, {"final_label": final_label, "relevance": asdict(relevance), "fallback": True}


def _build_v2_prompt(
    assignment: AssignmentLike,
    content: str,
    relevance: RelevanceResult,
    structured: StructuredGrading,
    evidence: list[dict[str, Any]],
) -> str:
    evidence_block = "\n".join(f"[{item['ref']}] {item['text']}" for item in evidence) or "无"
    dim_json = json.dumps([asdict(item) for item in structured.dimensions], ensure_ascii=False)
    model_signal_json = json.dumps(structured.model_signal or {}, ensure_ascii=False)

    return (
        "你是高校课程助教，请基于结构化批改结果生成评语。\n"
        "必须严格输出以下三段：\n"
        "优点：\n"
        "问题：\n"
        "改进建议：\n"
        "禁止输出评分、分数、得分、百分比。\n"
        "问题与建议必须严格围绕题干显式要求，不能额外引入未要求维度（例如复杂度比较、应用场景扩展）。\n"
        "改进建议至少2条，且可执行。\n"
        f"题目：{assignment.title}\n"
        f"描述：{assignment.description or '无'}\n"
        f"关键词：{assignment.keywords or '无'}\n"
        f"学生答案：{content}\n"
        f"相关性：{relevance.label} ({relevance.reason})\n"
        f"结构化结果：{dim_json}\n"
        f"模型信号：{model_signal_json}\n"
        f"偏题风险提示：{structured.risk_hint or '无'}\n"
        f"课程证据：\n{evidence_block}\n"
    )


def _fallback_v2_feedback(relevance: RelevanceResult, structured: StructuredGrading) -> str:
    _ = relevance
    strengths = "；".join(structured.strengths[:2]) or "回答包含部分可用信息。"
    issues = "；".join(structured.issues[:2]) or "答案还不够完整。"
    if structured.risk_hint and structured.risk_hint not in issues:
        issues = f"{issues}；{structured.risk_hint}"
    suggestions = "；".join(structured.suggestions[:3]) or "建议补充定义、要点与示例。"
    return f"优点：{strengths}\n问题：{issues}\n改进建议：{suggestions}"


async def _generate_v2_feedback(
    assignment: AssignmentLike,
    content: str,
    llm_call: Callable[[str], Awaitable[str]],
    db=None,
) -> tuple[str, dict[str, Any]]:
    relevance = evaluate_relevance(assignment, content, db=db)
    evidence = _retrieve_evidence(assignment, content, top_k=4)
    structured = _grade_structured(content, relevance, evidence)
    _apply_feedback_model_signal(assignment, content, structured, evidence, db=db)

    llm_output = await llm_call(_build_v2_prompt(assignment, content, relevance, structured, evidence))
    feedback = parse_llm_feedback(llm_output)
    if not feedback:
        feedback = _fallback_v2_feedback(relevance, structured)
    feedback = _enforce_feedback_scope(assignment, feedback, structured)

    if structured.risk_hint and structured.risk_hint not in feedback:
        if "问题：" in feedback:
            feedback = feedback.replace("问题：", f"问题：{structured.risk_hint} ", 1)
        else:
            feedback = f"{feedback}\n问题：{structured.risk_hint}"

    if "改进建议：" not in feedback:
        feedback = f"{feedback}\n改进建议：按定义、要点、示例三步重写答案。"

    if not all(x in feedback for x in ("优点：", "问题：", "改进建议：")):
        feedback = _fallback_v2_feedback(relevance, structured)

    return feedback, {
        "relevance": asdict(relevance),
        "structured": {
            "dimensions": [asdict(item) for item in structured.dimensions],
            "strengths": structured.strengths,
            "issues": structured.issues,
            "suggestions": structured.suggestions,
            "risk_hint": structured.risk_hint,
            "evidence_refs": structured.evidence_refs,
            "model_signal": structured.model_signal,
        },
        "evidence_count": len(evidence),
    }


async def generate_text_assignment_feedback(
    assignment: AssignmentLike,
    content: str,
    llm_call: Callable[[str], Awaitable[str]],
    db=None,
) -> str:
    mode = _mode()

    if mode == "legacy":
        feedback, _ = await _generate_legacy_feedback(assignment, content, llm_call, db=db)
        return feedback

    if mode == "v2":
        feedback, _ = await _generate_v2_feedback(assignment, content, llm_call, db=db)
        return feedback

    # shadow: 返回 legacy，但后台记录 legacy/v2 对比
    legacy_feedback, legacy_debug = await _generate_legacy_feedback(assignment, content, llm_call, db=db)
    try:
        v2_feedback, v2_debug = await _generate_v2_feedback(assignment, content, llm_call, db=db)
    except Exception as exc:
        v2_feedback = ""
        v2_debug = {"error": str(exc)}
        logger.warning("assignment v2 shadow generation failed: %s", exc)

    _append_shadow_log(
        {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "mode": "shadow",
            "assignment": {
                "title": assignment.title,
                "description": assignment.description,
                "keywords": assignment.keywords,
                "course_id": _assignment_course_id(assignment),
            },
            "content": content,
            "legacy_feedback": legacy_feedback,
            "v2_feedback": v2_feedback,
            "legacy_debug": legacy_debug,
            "v2_debug": v2_debug,
            "returned": "legacy",
        }
    )
    return legacy_feedback
