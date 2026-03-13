from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
import re
import unicodedata
from typing import Any, Awaitable, Callable, Literal, Protocol

from app.config import settings
from app.services.assignment_feedback_generator import generate_feedback_text
from app.services.model_paths import resolve_active_assignment_feedback_sft_model_path

try:
    import jieba.posseg as jieba_posseg
except Exception:  # pragma: no cover
    jieba_posseg = None


logger = logging.getLogger(__name__)

RelevanceLabel = Literal["relevant", "ambiguous", "off_topic"]
FeedbackTier = Literal["invalid", "off_topic", "partial", "good"]


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


@dataclass
class FeedbackDiagnostic:
    tier: FeedbackTier
    reasons: list[str]
    must_cover: list[str]
    missing: list[str]
    forbidden_hints: list[str]
    relevance: RelevanceResult


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
_SCORE_LINE_RE = re.compile(r"^(?:评分|得分|score)\s*[:：]\s*\d{1,3}\s*$", re.IGNORECASE)
_SCORE_INLINE_RE = re.compile(r"(?:评分|得分|score)\s*[:：]\s*\d{1,3}", re.IGNORECASE)
_FEEDBACK_PREFIX_RE = re.compile(r"^(?:评语|反馈)\s*[:：]\s*", re.IGNORECASE)
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*")
_MARKDOWN_BULLET_RE = re.compile(r"^\s*[-*+]\s+")
_SECTION_HEADER_RE = re.compile(r"(优点：|问题：|改进建议：)")
_LLM_FAILURE_MARKERS = (
    "接口错误",
    "占位答复",
    "timeout",
    "timed out",
    "service unavailable",
)

_FORBIDDEN_HINTS = {
    "复杂度",
    "时间复杂度",
    "空间复杂度",
    "应用场景",
    "实际应用",
    "优缺点",
    "比较",
    "对比",
    "选择标准",
    "工程实践",
}

_DEFINITION_PATTERNS = (r"什么是", r"定义", r"解释", r"简述", r"含义")
_LISTING_PATTERNS = (r"有哪些", r"列举", r"写出", r"请写出", r"请列举")
_ALLOW_COMPLEXITY_PATTERNS = (r"复杂度", r"时间复杂度", r"空间复杂度")
_ALLOW_COMPARISON_PATTERNS = (r"比较", r"对比", r"优缺点", r"区别", r"异同")
_ALLOW_APPLY_PATTERNS = (r"应用", r"场景", r"实例", r"案例")


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
        weighted[term] = weighted.get(term, 0.0) + 3.4

    for term in _tokenize_terms(desc):
        weighted[term] = weighted.get(term, 0.0) + 2.1

    for term in kw:
        weighted[term] = weighted.get(term, 0.0) + 1.9

    if title and not _GENERIC_TITLE_RE.fullmatch(title):
        for term in _tokenize_terms(title):
            weighted[term] = weighted.get(term, 0.0) + 0.9

    ranked = sorted(weighted.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    return [term for term, _ in ranked[:12]], core_terms


def _assignment_scope_text(assignment: AssignmentLike) -> str:
    return " ".join(
        [
            (assignment.title or "").strip().lower(),
            (assignment.description or "").strip().lower(),
            (assignment.keywords or "").strip().lower(),
        ]
    ).strip()


def _contains_any_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _scope_forbidden_hints(assignment: AssignmentLike) -> list[str]:
    scope = _assignment_scope_text(assignment)
    forbidden = set(_FORBIDDEN_HINTS)

    if _contains_any_pattern(scope, _ALLOW_COMPLEXITY_PATTERNS):
        forbidden -= {"复杂度", "时间复杂度", "空间复杂度"}
    if _contains_any_pattern(scope, _ALLOW_COMPARISON_PATTERNS):
        forbidden -= {"优缺点", "比较", "对比", "选择标准"}
    if _contains_any_pattern(scope, _ALLOW_APPLY_PATTERNS):
        forbidden -= {"应用场景", "实际应用", "工程实践"}

    return sorted(forbidden)


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


def evaluate_relevance(assignment: AssignmentLike, content: str, db=None) -> RelevanceResult:
    _ = db
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

    # 定义题里若答案起句定义了非核心概念（如题问进程，答线程），优先判偏题。
    if core_terms and leading and leading not in core_terms and not matched_core:
        return RelevanceResult("off_topic", "leading_term_not_in_core", focus_terms, core_terms, matched_focus, matched_core, missing_focus, coverage)

    if core_terms and matched_core and leading and leading not in core_terms and not definition_hit and len(matched_core) <= 1:
        return RelevanceResult("off_topic", "leading_term_mismatch", focus_terms, core_terms, matched_focus, matched_core, missing_focus, coverage)

    if len(matched_focus) >= 2 and coverage >= 0.26:
        return RelevanceResult("relevant", "focus_hit", focus_terms, core_terms, matched_focus, matched_core, missing_focus, coverage)

    if core_terms and not matched_core and len(matched_focus) == 0 and coverage <= 0.10:
        return RelevanceResult("off_topic", "core_miss_low_coverage", focus_terms, core_terms, matched_focus, matched_core, missing_focus, coverage)

    return RelevanceResult("ambiguous", "borderline", focus_terms, core_terms, matched_focus, matched_core, missing_focus, coverage)


def _invalid_text_reason(content: str, relevance: RelevanceResult) -> str | None:
    raw = (content or "").strip()
    if not raw:
        return "未提交有效内容"

    pure = re.sub(r"\s+", "", raw)
    zh_count = len(re.findall(r"[\u4e00-\u9fff]", pure))
    token_count = len(_TOKEN_RE.findall(raw))

    if len(pure) < 4:
        return "答案过短，无法判断掌握情况"
    if zh_count == 0 and token_count <= 2:
        return "答案缺少有效学术内容"
    if relevance.coverage <= 0.01 and zh_count < 3 and token_count < 4:
        return "答案信息量过低，无法对应题干"
    if not relevance.matched_core_terms and relevance.coverage < 0.2 and len(pure) < 18:
        return "答案信息不足且未对齐题干核心概念"
    return None


def _build_diagnostic(assignment: AssignmentLike, content: str, db=None) -> FeedbackDiagnostic:
    relevance = evaluate_relevance(assignment, content, db=db)
    must_cover = list(dict.fromkeys((relevance.core_terms + relevance.focus_terms)[:8]))
    missing = relevance.missing_focus_terms[:4]
    forbidden_hints = _scope_forbidden_hints(assignment)

    reasons: list[str] = []
    invalid_reason = _invalid_text_reason(content, relevance)
    if invalid_reason:
        reasons.append(invalid_reason)
        return FeedbackDiagnostic(
            tier="invalid",
            reasons=reasons,
            must_cover=must_cover,
            missing=missing,
            forbidden_hints=forbidden_hints,
            relevance=relevance,
        )

    if relevance.label == "off_topic":
        reasons.append("回答与题干核心概念不一致")
        return FeedbackDiagnostic(
            tier="off_topic",
            reasons=reasons,
            must_cover=must_cover,
            missing=missing,
            forbidden_hints=forbidden_hints,
            relevance=relevance,
        )

    length = len((content or "").strip())
    if relevance.coverage >= 0.42 and length >= 40:
        reasons.append("核心要点覆盖较好")
        tier: FeedbackTier = "good"
    else:
        reasons.append("已覆盖部分要点，但仍有缺口")
        tier = "partial"

    return FeedbackDiagnostic(
        tier=tier,
        reasons=reasons,
        must_cover=must_cover,
        missing=missing,
        forbidden_hints=forbidden_hints,
        relevance=relevance,
    )


def _llm_failed(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return True
    return any(marker in normalized for marker in _LLM_FAILURE_MARKERS)


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
    return cleaned


def _split_sections(text: str) -> dict[str, list[str]]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    sections: dict[str, list[str]] = {"优点": [], "问题": [], "改进建议": []}
    current: str | None = None

    for line in lines:
        if line.startswith("优点："):
            current = "优点"
            tail = line[len("优点：") :].strip()
            if tail:
                sections[current].append(tail)
            continue
        if line.startswith("问题："):
            current = "问题"
            tail = line[len("问题：") :].strip()
            if tail:
                sections[current].append(tail)
            continue
        if line.startswith("改进建议："):
            current = "改进建议"
            tail = line[len("改进建议：") :].strip()
            if tail:
                sections[current].append(tail)
            continue
        if current:
            sections[current].append(line)

    return sections


def _dedupe_keep_order(items: list[str]) -> list[str]:
    output: list[str] = []
    for item in items:
        text = (item or "").strip("；;。 \t\r\n")
        if not text:
            continue
        if text not in output:
            output.append(text)
    return output


def _remove_forbidden(items: list[str], forbidden_hints: list[str]) -> list[str]:
    output: list[str] = []
    for item in items:
        parts = [p.strip() for p in re.split(r"[；;。]\s*", item) if p.strip()]
        if not parts:
            parts = [item.strip()]
        kept: list[str] = []
        for part in parts:
            if any(hint in part for hint in forbidden_hints):
                continue
            kept.append(part)
        if kept:
            output.append("；".join(kept))
    return _dedupe_keep_order(output)


def _fallback_feedback(diag: FeedbackDiagnostic) -> str:
    missing_text = "、".join(diag.missing[:3])
    must_text = "、".join(diag.must_cover[:3])

    if diag.tier == "invalid":
        issue = diag.reasons[0] if diag.reasons else "答案信息不足，无法进行有效批改"
        suggestion = f"请先给出清晰定义，并围绕“{must_text or '题干核心概念'}”补充至少2个关键要点。"
        return f"问题：{issue}\n改进建议：{suggestion}"

    if diag.tier == "off_topic":
        suggestion = f"请围绕题干要求重写答案，优先覆盖“{must_text or '题干核心概念'}”。"
        if missing_text:
            suggestion = f"请围绕题干要求重写答案，并补充“{missing_text}”等关键内容。"
        return f"问题：回答与题干核心概念不一致。\n改进建议：{suggestion}"

    if diag.tier == "partial":
        suggestion = f"先补齐“{missing_text or must_text or '题干要点'}”，再按“定义-要点-简短示例”组织答案。"
        return f"问题：答案覆盖了部分要点，但完整性不足。\n改进建议：{suggestion}"

    suggestion = f"在保持当前正确性的基础上，可再补充“{missing_text}”以提升完整度。" if missing_text else "可补充1-2个关键点并精炼表述。"
    return f"优点：回答与题干核心概念对齐，结构较清晰。\n改进建议：{suggestion}"


def _build_generation_prompt(assignment: AssignmentLike, content: str, diag: FeedbackDiagnostic) -> str:
    must_cover_text = "、".join(diag.must_cover[:8]) or "题干核心概念"
    missing_text = "、".join(diag.missing[:6]) or "无"
    forbidden_text = "、".join(diag.forbidden_hints) or "无"

    if diag.tier in {"invalid", "off_topic"}:
        style_rule = (
            "输出两段：\n"
            "问题：...\n"
            "改进建议：...\n"
            "不得输出“优点：”段。"
        )
    elif diag.tier == "partial":
        style_rule = (
            "可输出两段或三段。若没有明确亮点，可不写“优点：”。\n"
            "建议至少给出2条可执行改进建议。"
        )
    else:
        style_rule = "输出“优点：”和“改进建议：”两段，可选“问题：”。"

    return (
        "你是高校课程助教，请生成中文作业评语。\n"
        "硬约束：\n"
        "1) 不要使用Markdown符号；\n"
        "2) 不要出现评分、得分、百分比；\n"
        "3) 题干未要求的内容不能作为缺点或必改项。\n"
        f"4) 禁止引入：{forbidden_text}\n"
        f"5) {style_rule}\n"
        f"题目：{assignment.title}\n"
        f"描述：{assignment.description or '无'}\n"
        f"关键词：{assignment.keywords or '无'}\n"
        f"诊断档位：{diag.tier}\n"
        f"诊断原因：{'；'.join(diag.reasons) or '无'}\n"
        f"建议覆盖：{must_cover_text}\n"
        f"当前缺失：{missing_text}\n"
        f"学生答案：{content}\n"
    )


async def _generate_with_local_model(prompt: str, db=None) -> tuple[str, str]:
    if not settings.enable_assignment_feedback_sft_model:
        return "", "disabled"

    model_path, source = resolve_active_assignment_feedback_sft_model_path(db)
    if not model_path:
        return "", "model_not_found"

    try:
        text = await asyncio.to_thread(
            generate_feedback_text,
            prompt,
            model_path,
            int(getattr(settings, "assignment_feedback_sft_max_new_tokens", 220)),
            float(getattr(settings, "assignment_feedback_sft_temperature", 0.2)),
            float(getattr(settings, "assignment_feedback_sft_top_p", 0.9)),
        )
        return (text or "").strip(), source
    except Exception as exc:
        logger.warning("local feedback generation failed: %s", exc)
        return "", f"local_error:{exc}"


async def _generate_feedback_draft(
    assignment: AssignmentLike,
    content: str,
    diag: FeedbackDiagnostic,
    llm_call: Callable[[str], Awaitable[str]],
    db=None,
) -> tuple[str, str]:
    prompt = _build_generation_prompt(assignment, content, diag)

    local_text, local_source = await _generate_with_local_model(prompt, db=db)
    if local_text and not _llm_failed(local_text):
        return local_text, f"local:{local_source}"

    if getattr(settings, "assignment_feedback_external_fallback", True):
        llm_text = (await llm_call(prompt)).strip()
        if llm_text and not _llm_failed(llm_text):
            return llm_text, "external_llm"

    return "", "fallback_template"


def _sanitize_feedback(assignment: AssignmentLike, draft: str, diag: FeedbackDiagnostic) -> str:
    cleaned = parse_llm_feedback(draft)
    if not cleaned:
        return _fallback_feedback(diag)

    cleaned = _SECTION_HEADER_RE.sub(r"\n\1", cleaned).strip()
    sections = _split_sections(cleaned)

    pros = _dedupe_keep_order(sections["优点"])
    issues = _remove_forbidden(sections["问题"], diag.forbidden_hints)
    suggestions = _remove_forbidden(sections["改进建议"], diag.forbidden_hints)

    if diag.tier == "invalid":
        issue = issues[0] if issues else (diag.reasons[0] if diag.reasons else "答案信息不足，无法判断掌握情况")
        suggestion = suggestions[0] if suggestions else f"请围绕“{'、'.join(diag.must_cover[:3]) or '题干核心概念'}”重新作答。"
        return f"问题：{issue}\n改进建议：{suggestion}"

    if diag.tier == "off_topic":
        issue = issues[0] if issues else "回答与题干核心概念不一致。"
        suggestion = suggestions[0] if suggestions else f"请围绕“{'、'.join(diag.must_cover[:3]) or '题干核心概念'}”重写答案。"
        return f"问题：{issue}\n改进建议：{suggestion}"

    if diag.tier == "partial":
        lines: list[str] = []
        if pros:
            lines.append(f"优点：{'；'.join(pros[:1])}")
        issue = issues[0] if issues else "答案覆盖了部分要点，但完整性不足。"
        lines.append(f"问题：{issue}")

        if not suggestions:
            missing = "、".join(diag.missing[:3])
            suggestions = [f"补齐“{missing or '题干关键要点'}”，并按“定义-要点-示例”组织答案。"]
        lines.append(f"改进建议：{'；'.join(suggestions[:2])}")
        return "\n".join(lines)

    # good
    lines = []
    if pros:
        lines.append(f"优点：{'；'.join(pros[:2])}")
    else:
        lines.append("优点：回答与题干核心概念对齐，结构较清晰。")

    if issues:
        lines.append(f"问题：{'；'.join(issues[:1])}")

    if not suggestions:
        if diag.missing:
            suggestions = [f"可补充“{'、'.join(diag.missing[:2])}”以提升完整度。"]
        else:
            suggestions = ["可补充1-2个关键细节并精炼表达。"]
    lines.append(f"改进建议：{'；'.join(suggestions[:2])}")
    return "\n".join(lines)


async def generate_text_assignment_feedback(
    assignment: AssignmentLike,
    content: str,
    llm_call: Callable[[str], Awaitable[str]],
    db=None,
) -> str:
    diag = _build_diagnostic(assignment, content, db=db)
    draft, source = await _generate_feedback_draft(assignment, content, diag, llm_call, db=db)
    feedback = _sanitize_feedback(assignment, draft, diag)
    if not feedback:
        feedback = _fallback_feedback(diag)

    logger.info(
        "assignment_feedback_pipeline tier=%s relevance=%s source=%s",
        diag.tier,
        diag.relevance.label,
        source,
    )
    return feedback
