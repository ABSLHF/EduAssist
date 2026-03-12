from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
import unicodedata
from typing import Awaitable, Callable, Literal, Protocol

from app.config import settings
from app.services.assignment_relevance_model import (
    predict_relevance_probability,
    predict_reranker_probability,
)
from app.services.model_paths import resolve_active_assignment_relevance_model_path

try:
    import jieba.posseg as jieba_posseg
except Exception:  # pragma: no cover - optional dependency fallback
    jieba_posseg = None

logger = logging.getLogger(__name__)


RelevanceLabel = Literal["relevant", "ambiguous", "off_topic"]


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


_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{1,12}|[A-Za-z]{2,24}|[A-Za-z]+[0-9]+")
_TOKEN_CLEAN_RE = re.compile(r"^[\W_]+|[\W_]+$")
_SCORE_LINE_RE = re.compile(r"^(?:评分|得分)\s*[:：]", re.IGNORECASE)
_SCORE_INLINE_RE = re.compile(r"(?:评分|得分)\s*[:：]\s*\d{1,3}", re.IGNORECASE)
_SCORE_LEADING_RE = re.compile(r"^(?:评分|得分)\s*[:：]?\s*\d{1,3}\s*", re.IGNORECASE)
_FEEDBACK_PREFIX_RE = re.compile(r"^(?:评语|反馈)\s*[:：]\s*", re.IGNORECASE)
_KEYWORD_SPLIT_RE = re.compile(r"[,，;；、\s\n\r\t]+")
_GENERIC_TITLE_RE = re.compile(r"^(?:作业|练习|任务|实验)\s*[0-9一二三四五六七八九十]+$", re.IGNORECASE)
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
    "相关",
    "内容",
    "进行",
    "分析",
    "描述",
}
_LLM_FAILURE_MARKERS = ("接口错误", "占位答复", "timeout", "timed out", "error")
_AMBIG_LABEL_RE = re.compile(r'"label"\s*:\s*"(relevant|off_topic)"', re.IGNORECASE)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.lower().strip()
    return re.sub(r"\s+", "", normalized)


def _clean_term(token: str) -> str:
    cleaned = _TOKEN_CLEAN_RE.sub("", token.strip())
    cleaned = _normalize_text(cleaned)
    return cleaned


def _is_noise_term(term: str) -> bool:
    if not term:
        return True
    if term in _STOPWORDS or term in _QUESTION_WORDS:
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
        accepted_flags = ("n", "vn", "nz", "eng")
        try:
            for pair in jieba_posseg.cut(text):
                word = _clean_term(getattr(pair, "word", "") or "")
                flag = (getattr(pair, "flag", "") or "").lower()
                if not word:
                    continue
                if flag.startswith(accepted_flags):
                    terms.append(word)
        except Exception:
            terms = []

    if not terms:
        for raw in _TOKEN_RE.findall(text):
            terms.append(_clean_term(raw))

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
    terms: list[str] = []
    for piece in _KEYWORD_SPLIT_RE.split(keywords_raw):
        token = _clean_term(piece)
        if _is_noise_term(token):
            continue
        if token not in terms:
            terms.append(token)
    return terms


def _extract_core_terms(text: str) -> list[str]:
    if not text:
        return []
    found: list[str] = []
    for pattern in _QUESTION_PATTERNS:
        for match in pattern.finditer(text):
            raw_token = (match.group(1) or "").strip()
            token = _clean_term(raw_token)
            candidate_terms: list[str] = []
            if token:
                candidate_terms.append(token)

            # 把“死锁的四个必要条件”拆成更稳定的核心词（如“死锁”）。
            for piece in re.split(r"[的和与及、,，;；\s]+", raw_token):
                cleaned_piece = _clean_term(piece)
                if cleaned_piece:
                    candidate_terms.append(cleaned_piece)

            for piece in _tokenize_terms(raw_token):
                if piece:
                    candidate_terms.append(piece)

            for candidate in candidate_terms:
                if not candidate or candidate in _QUESTION_WORDS:
                    continue
                if len(candidate) > 10:
                    continue
                if candidate not in found:
                    found.append(candidate)
    return found


def _leading_defined_term(text: str) -> str:
    match = re.search(r"^\s*([A-Za-z0-9\u4e00-\u9fff]{1,12})\s*(?:是|指|表示)", text or "")
    if not match:
        return ""
    return _clean_term(match.group(1))


def _core_definition_hit(content_norm: str, core_terms: list[str]) -> bool:
    for term in core_terms:
        if not term:
            continue
        if f"{term}是" in content_norm or f"{term}指" in content_norm or f"{term}表示" in content_norm:
            return True
    return False


def _build_focus_terms(assignment: AssignmentLike) -> tuple[list[str], list[str]]:
    weighted: dict[str, float] = {}
    core_terms: list[str] = []

    description = (assignment.description or "").strip()
    title = (assignment.title or "").strip()
    keywords = _parse_keywords(assignment.keywords)

    description_core = _extract_core_terms(description)
    title_core = _extract_core_terms(title)
    for term in description_core + title_core:
        if term not in core_terms:
            core_terms.append(term)
        weighted[term] = weighted.get(term, 0.0) + 3.6

    for term in _tokenize_terms(description):
        weighted[term] = weighted.get(term, 0.0) + 2.2

    for term in keywords:
        weighted[term] = weighted.get(term, 0.0) + 1.9

    if title and not _GENERIC_TITLE_RE.fullmatch(title):
        for term in _tokenize_terms(title):
            weighted[term] = weighted.get(term, 0.0) + 1.1

    ranked = sorted(weighted.items(), key=lambda item: (item[1], len(item[0])), reverse=True)
    focus_terms = [term for term, _ in ranked[:12]]
    return focus_terms, core_terms


def _evaluate_relevance_rule_only(assignment: AssignmentLike, content: str) -> RelevanceResult:
    focus_terms, core_terms = _build_focus_terms(assignment)
    content_norm = _normalize_text(content)

    if not content_norm:
        return RelevanceResult(
            label="off_topic",
            reason="empty_content",
            focus_terms=focus_terms,
            core_terms=core_terms,
            matched_focus_terms=[],
            matched_core_terms=[],
            missing_focus_terms=focus_terms,
            coverage=0.0,
        )

    if not focus_terms:
        return RelevanceResult(
            label="ambiguous",
            reason="no_focus_terms",
            focus_terms=[],
            core_terms=[],
            matched_focus_terms=[],
            matched_core_terms=[],
            missing_focus_terms=[],
            coverage=0.0,
        )

    matched_focus = [term for term in focus_terms if term in content_norm]
    matched_core = [term for term in core_terms if term in content_norm]
    missing_focus = [term for term in focus_terms if term not in matched_focus]
    coverage = len(matched_focus) / max(1, len(focus_terms))

    leading_term = _leading_defined_term(content)
    definition_hit = _core_definition_hit(content_norm, core_terms)

    # 命中核心且出现“核心术语 + 定义动词”时，直接视为相关。
    if matched_core and definition_hit:
        return RelevanceResult(
            label="relevant",
            reason="core_definition_hit",
            focus_terms=focus_terms,
            core_terms=core_terms,
            matched_focus_terms=matched_focus,
            matched_core_terms=matched_core,
            missing_focus_terms=missing_focus,
            coverage=coverage,
        )

    # 出现“线程是...”这类首句定义其他概念，且核心仅被顺带提及，判为偏题。
    if (
        core_terms
        and matched_core
        and leading_term
        and leading_term not in core_terms
        and not definition_hit
        and len(matched_core) <= 1
    ):
        return RelevanceResult(
            label="off_topic",
            reason="leading_term_mismatch",
            focus_terms=focus_terms,
            core_terms=core_terms,
            matched_focus_terms=matched_focus,
            matched_core_terms=matched_core,
            missing_focus_terms=missing_focus,
            coverage=coverage,
        )

    if len(matched_focus) >= 2 and coverage >= 0.28:
        return RelevanceResult(
            label="relevant",
            reason="focus_hit",
            focus_terms=focus_terms,
            core_terms=core_terms,
            matched_focus_terms=matched_focus,
            matched_core_terms=matched_core,
            missing_focus_terms=missing_focus,
            coverage=coverage,
        )

    # 只有“核心全未命中 + 覆盖极低”才判偏题。
    if core_terms and not matched_core and len(matched_focus) == 0 and coverage <= 0.10:
        return RelevanceResult(
            label="off_topic",
            reason="core_miss_low_coverage",
            focus_terms=focus_terms,
            core_terms=core_terms,
            matched_focus_terms=matched_focus,
            matched_core_terms=matched_core,
            missing_focus_terms=missing_focus,
            coverage=coverage,
        )

    return RelevanceResult(
        label="ambiguous",
        reason="borderline",
        focus_terms=focus_terms,
        core_terms=core_terms,
        matched_focus_terms=matched_focus,
        matched_core_terms=matched_core,
        missing_focus_terms=missing_focus,
        coverage=coverage,
    )


def _build_assignment_query_text(assignment: AssignmentLike) -> str:
    description = (assignment.description or "").strip()
    keywords = (assignment.keywords or "").strip()
    title = (assignment.title or "").strip()
    parts: list[str] = []
    if description:
        parts.append(description)
    if keywords:
        parts.append(f"关键词：{keywords}")
    if title and not _GENERIC_TITLE_RE.fullmatch(title):
        parts.append(f"标题：{title}")
    if not parts and title:
        parts.append(title)
    return "\n".join(parts)


def _score_with_assignment_relevance_model(
    assignment: AssignmentLike,
    content: str,
    db=None,
) -> tuple[float | None, str | None]:
    if not settings.enable_assignment_relevance_model:
        return None, None

    query_text = _build_assignment_query_text(assignment)
    if not query_text or not content.strip():
        return None, None

    model_path, model_source = resolve_active_assignment_relevance_model_path(db)
    if not model_path:
        return None, None

    base_score = predict_relevance_probability(query_text, content, model_path=model_path, max_length=384)
    blended_score = float(base_score)

    if settings.assignment_relevance_use_reranker:
        reranker_name = (settings.assignment_relevance_reranker_model or "").strip()
        if reranker_name:
            try:
                reranker_score = predict_reranker_probability(
                    query_text,
                    content,
                    model_name=reranker_name,
                    max_length=512,
                )
                weight = float(settings.assignment_relevance_reranker_weight)
                weight = min(max(weight, 0.0), 0.8)
                blended_score = (1.0 - weight) * base_score + weight * reranker_score
            except Exception as exc:
                logger.warning("assignment reranker scoring failed, fallback to classifier score: %s", exc)

    return min(max(float(blended_score), 0.0), 1.0), model_source


def _label_from_score(score: float, threshold_hi: float, threshold_lo: float) -> RelevanceLabel:
    if score >= threshold_hi:
        return "relevant"
    if score <= threshold_lo:
        return "off_topic"
    return "ambiguous"


def _merge_rule_and_model(rule_result: RelevanceResult, model_label: RelevanceLabel) -> RelevanceLabel:
    if model_label == "ambiguous":
        return rule_result.label

    if model_label == "relevant":
        if rule_result.label == "off_topic":
            return "ambiguous"
        return "relevant"

    if rule_result.label == "relevant":
        return "ambiguous"
    return "off_topic"


def evaluate_relevance(assignment: AssignmentLike, content: str, db=None) -> RelevanceResult:
    rule_result = _evaluate_relevance_rule_only(assignment, content)

    try:
        model_score, model_source = _score_with_assignment_relevance_model(assignment, content, db=db)
    except Exception as exc:
        logger.warning("assignment relevance model unavailable, fallback to rule engine: %s", exc)
        return rule_result

    if model_score is None:
        return rule_result

    threshold_hi = float(settings.assignment_relevance_threshold_hi)
    threshold_lo = float(settings.assignment_relevance_threshold_lo)
    threshold_hi = min(max(threshold_hi, 0.0), 1.0)
    threshold_lo = min(max(threshold_lo, 0.0), 1.0)
    if threshold_lo > threshold_hi:
        threshold_lo, threshold_hi = threshold_hi, threshold_lo

    model_label = _label_from_score(model_score, threshold_hi=threshold_hi, threshold_lo=threshold_lo)
    final_label = _merge_rule_and_model(rule_result, model_label=model_label)

    return RelevanceResult(
        label=final_label,
        reason=f"model_score={model_score:.3f}|model_label={model_label}|rule={rule_result.reason}",
        focus_terms=rule_result.focus_terms,
        core_terms=rule_result.core_terms,
        matched_focus_terms=rule_result.matched_focus_terms,
        matched_core_terms=rule_result.matched_core_terms,
        missing_focus_terms=rule_result.missing_focus_terms,
        coverage=rule_result.coverage,
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
        if _SCORE_LINE_RE.search(line):
            line = _SCORE_INLINE_RE.sub("", line).strip()
            line = _SCORE_LEADING_RE.sub("", line).strip()
            line = _FEEDBACK_PREFIX_RE.sub("", line)
            if not line:
                continue
        line = _FEEDBACK_PREFIX_RE.sub("", line)
        lines.append(line)

    cleaned = "\n".join(lines).strip()
    if not cleaned:
        cleaned = _FEEDBACK_PREFIX_RE.sub("", text)
    cleaned = _SCORE_INLINE_RE.sub("", cleaned).strip()
    cleaned = _SCORE_LEADING_RE.sub("", cleaned).strip()
    cleaned = _FEEDBACK_PREFIX_RE.sub("", cleaned).strip()
    return cleaned


def _build_ambiguous_check_prompt(assignment: AssignmentLike, content: str, relevance: RelevanceResult) -> str:
    return (
        "你是作业相关性判断器，请判断学生回答是否围绕题目。\n"
        "仅输出 JSON，不要输出其他内容：\n"
        '{"label":"relevant|off_topic","reason":"一句话说明"}\n\n'
        f"作业标题：{assignment.title}\n"
        f"作业描述：{assignment.description or '无'}\n"
        f"关键词：{assignment.keywords or '无'}\n"
        f"命中术语：{','.join(relevance.matched_focus_terms) or '无'}\n"
        f"缺失术语：{','.join(relevance.missing_focus_terms[:6]) or '无'}\n"
        f"学生答案：{content}\n"
    )


async def _resolve_ambiguous_with_llm(
    assignment: AssignmentLike,
    content: str,
    relevance: RelevanceResult,
    llm_call: Callable[[str], Awaitable[str]],
) -> tuple[RelevanceLabel, str]:
    prompt = _build_ambiguous_check_prompt(assignment, content, relevance)
    llm_output = (await llm_call(prompt)).strip()
    if _llm_failed(llm_output):
        # 低误判优先：LLM不可用时回退为相关。
        return "relevant", "llm_unavailable_fallback_relevant"

    try:
        payload_match = re.search(r"\{.*\}", llm_output, flags=re.DOTALL)
        payload_text = payload_match.group(0) if payload_match else llm_output
        payload = json.loads(payload_text)
        label = str(payload.get("label", "")).strip().lower()
        reason = str(payload.get("reason", "")).strip()
        if label in {"relevant", "off_topic"}:
            return label, reason or "llm_json_result"
    except Exception:
        pass

    keyword_match = _AMBIG_LABEL_RE.search(llm_output)
    if keyword_match:
        label = keyword_match.group(1).lower()
        if label in {"relevant", "off_topic"}:
            return label, "llm_regex_result"

    if "偏题" in llm_output or "不相关" in llm_output:
        return "off_topic", "llm_keyword_off_topic"
    if "相关" in llm_output:
        return "relevant", "llm_keyword_relevant"

    return "relevant", "llm_uncertain_fallback_relevant"


def _build_relevant_feedback_prompt(assignment: AssignmentLike, content: str, relevance: RelevanceResult) -> str:
    return (
        "你是高校课程助教。学生回答已经与题目相关，请生成批改评语。\n"
        "要求：\n"
        "1) 必须包含：优点、存在问题、可执行修改建议（至少2条）。\n"
        "2) 不要给分数，不要出现“评分/得分”等字样。\n"
        "3) 控制在100-220字，语气客观具体。\n\n"
        f"作业标题：{assignment.title}\n"
        f"作业描述：{assignment.description or '无'}\n"
        f"关键词：{assignment.keywords or '无'}\n"
        f"命中术语：{','.join(relevance.matched_focus_terms) or '无'}\n"
        f"学生答案：{content}\n"
    )


def _build_off_topic_feedback(assignment: AssignmentLike, relevance: RelevanceResult, llm_reason: str) -> str:
    missing = "、".join(relevance.missing_focus_terms[:4]) or "题目核心知识点"
    matched = "、".join(relevance.matched_focus_terms[:3])
    matched_text = f"已覆盖：{matched}。" if matched else ""
    reason_text = f"判定依据：{llm_reason}。" if llm_reason else ""
    return (
        f"你的回答与题目《{assignment.title}》的要求不一致。{reason_text}{matched_text}"
        f"当前答案未围绕“{missing}”展开。\n"
        "建议按以下结构重写：\n"
        "1) 先给出题目概念的准确定义；\n"
        "2) 补充关键特征/组成；\n"
        "3) 结合课程术语给出简短示例或对比。"
    )


def _build_keyword_fallback_feedback(assignment: AssignmentLike, relevance: RelevanceResult) -> str:
    if relevance.label == "off_topic":
        return _build_off_topic_feedback(assignment, relevance, "")
    if not relevance.focus_terms:
        return "回答已提交。建议围绕题干概念补充定义、关键特征与简要示例，使表达更完整。"
    hit = "、".join(relevance.matched_focus_terms[:4]) or "无"
    missing = "、".join(relevance.missing_focus_terms[:4]) or "无"
    return f"你已覆盖术语：{hit}。建议继续补充：{missing}，并用1-2句示例说明该概念的应用场景。"


async def generate_text_assignment_feedback(
    assignment: AssignmentLike,
    content: str,
    llm_call: Callable[[str], Awaitable[str]],
    db=None,
) -> str:
    relevance = evaluate_relevance(assignment, content, db=db)
    logger.info(
        "assignment_grading_decision label=%s reason=%s model_score=%s model_source=%s focus_hits=%s",
        relevance.label,
        relevance.reason,
        f"{relevance.model_score:.4f}" if relevance.model_score is not None else "n/a",
        relevance.model_source or "none",
        len(relevance.matched_focus_terms),
    )
    llm_reason = ""
    final_label = relevance.label

    if relevance.label == "ambiguous":
        final_label, llm_reason = await _resolve_ambiguous_with_llm(assignment, content, relevance, llm_call)

    if final_label == "off_topic":
        return _build_off_topic_feedback(assignment, relevance, llm_reason)

    prompt = _build_relevant_feedback_prompt(assignment, content, relevance)
    llm_output = await llm_call(prompt)
    feedback = parse_llm_feedback(llm_output)
    if feedback:
        return feedback
    return _build_keyword_fallback_feedback(assignment, relevance)
