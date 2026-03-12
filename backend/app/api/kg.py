from collections import Counter
import itertools
import logging
import re
import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_teacher
from app.config import settings
from app.db import get_db
from app.llm.client import call_llm
from app.models import models
from app.schemas.schemas import (
    KGCandidateBatchActionOut,
    KGCandidateBatchActionRequest,
    KGCandidateListOut,
    KGCandidateOut,
    KGCandidatesResponse,
    KGNodeBriefOut,
    KGResponse,
)
from app.services.keywords import (
    build_sentence_cooccurrence,
    extract_keywords,
    extract_keywords_with_meta,
    find_source_sentence,
    normalize_keyword,
    split_sentences,
)
from app.services.materials import extract_text

router = APIRouter()
logger = logging.getLogger(__name__)

_RELATION_CUE_WORDS = (
    "包括",
    "属于",
    "基于",
    "依赖",
    "组成",
    "包含",
    "使用",
    "通过",
    "用于",
    "实现",
    "由",
)

_KNOWN_NODE_BRIEFS = {
    "bfs": "BFS（广度优先搜索）是一种按层次逐层访问节点的图或树遍历算法，通常借助队列实现，可用于最短路径等问题。",
    "dfs": "DFS（深度优先搜索）是一种沿着分支尽可能深入再回溯的图或树遍历算法，通常通过递归或栈实现。",
    "tcp": "TCP（传输控制协议）是面向连接、可靠传输的传输层协议，提供顺序传输、重传与流量控制等机制。",
    "udp": "UDP（用户数据报协议）是无连接的传输层协议，开销小、时延低，但不保证可靠到达与有序性。",
    "http": "HTTP（超文本传输协议）是应用层协议，用于客户端与服务器之间进行请求与响应式通信。",
    "ip": "IP（互联网协议）是网络层核心协议，负责主机寻址与分组转发。",
    "队列": "队列是一种先进先出（FIFO）的线性结构，常用于任务调度、缓冲与广度优先搜索。",
    "栈": "栈是一种后进先出（LIFO）的线性结构，常用于函数调用、表达式求值与深度优先搜索。",
}


def _require_teacher_course(db: Session, teacher_id: int, course_id: int):
    course = db.query(models.Course).get(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    if course.teacher_id != teacher_id:
        raise HTTPException(status_code=403, detail="Permission denied")
    return course


def _require_material_of_course(db: Session, course_id: int, material_id: int):
    material = db.query(models.Material).get(material_id)
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    if material.course_id != course_id:
        raise HTTPException(status_code=400, detail="Material does not belong to the course")
    if material.parse_status != "success":
        raise HTTPException(status_code=400, detail="Material parse_status is not success")
    return material


def _clean_llm_definition(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    value = re.sub(r"^```(?:text|markdown|md|json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    value = re.sub(r"^\s*(定义|简要定义|知识点定义)[:：]\s*", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) > 180:
        value = value[:180].rstrip("，,;；。 ") + "。"
    return value


def _fallback_definition(node_name: str, course_name: str = "") -> str:
    key = (node_name or "").strip().lower()
    if key in _KNOWN_NODE_BRIEFS:
        return _KNOWN_NODE_BRIEFS[key]
    course_text = course_name or "本课程"
    return f"{node_name} 是《{course_text}》中的核心知识点，建议结合课程资料中的定义、特征与应用场景理解。"


_PCB_PROCESS_TERMS = (
    "数据结构",
    "操作系统",
    "进程",
    "线程",
    "调度",
    "进程控制块",
    "上下文切换",
)
_PCB_CIRCUIT_TERMS = (
    "电路",
    "电路板",
    "印刷电路板",
    "电子",
    "元件",
    "导线",
    "焊盘",
)


def _compact_snippet(text: str, max_len: int = 130) -> str:
    value = re.sub(r"\s+", " ", (text or "").strip())
    if not value:
        return ""
    if len(value) <= max_len:
        return value
    return f"{value[:max_len].rstrip('，,;；。.!?？')}。"


def _collect_node_brief_evidence(
    db: Session,
    *,
    course_id: int,
    node_name: str,
    related_edges: list[models.KnowledgeRelation],
    limit: int = 6,
) -> list[str]:
    rows: list[tuple[int | None, str]] = []
    material_ids: set[int] = set()

    for edge in related_edges:
        snippet = _compact_snippet(edge.evidence_sentence or "")
        if not snippet:
            continue
        rows.append((edge.material_id, snippet))
        if edge.material_id:
            material_ids.add(int(edge.material_id))

    term_norm = normalize_keyword(node_name)
    candidate_query = db.query(models.KnowledgeCandidate.material_id, models.KnowledgeCandidate.source_sentence).filter(
        models.KnowledgeCandidate.course_id == course_id,
    )
    if term_norm:
        candidate_query = candidate_query.filter(models.KnowledgeCandidate.term_norm == term_norm)
    else:
        candidate_query = candidate_query.filter(func.lower(models.KnowledgeCandidate.term) == node_name.strip().lower())

    candidate_rows = (
        candidate_query.order_by(
            func.coalesce(models.KnowledgeCandidate.score, -1).desc(),
            models.KnowledgeCandidate.updated_at.desc(),
        )
        .limit(12)
        .all()
    )
    for material_id, source_sentence in candidate_rows:
        snippet = _compact_snippet(source_sentence or "")
        if not snippet:
            continue
        rows.append((material_id, snippet))
        if material_id:
            material_ids.add(int(material_id))

    title_map: dict[int, str] = {}
    if material_ids:
        material_rows = (
            db.query(models.Material.id, models.Material.title)
            .filter(models.Material.id.in_(material_ids))
            .all()
        )
        title_map = {int(item.id): (item.title or "").strip() for item in material_rows}

    out: list[str] = []
    seen: set[str] = set()
    for material_id, snippet in rows:
        normalized = normalize_keyword(snippet) or snippet.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        title = title_map.get(int(material_id), "") if material_id else ""
        prefix = f"[资料:{title}]" if title else "[资料片段]"
        out.append(f"{prefix} {snippet}")
        if len(out) >= max(1, limit):
            break
    return out


def _keyword_hit_count(text: str, keywords: tuple[str, ...]) -> int:
    base = (text or "").lower()
    return sum(1 for item in keywords if item and item.lower() in base)


def _resolve_contextual_definition(
    *,
    course_name: str,
    node_name: str,
    related_labels: list[str],
    evidence_lines: list[str],
) -> str | None:
    key = (node_name or "").strip().lower()
    context_text = " ".join(
        [course_name or "", " ".join(related_labels or []), " ".join(evidence_lines or [])]
    )

    if key == "pcb":
        process_score = _keyword_hit_count(context_text, _PCB_PROCESS_TERMS)
        circuit_score = _keyword_hit_count(context_text, _PCB_CIRCUIT_TERMS)
        if process_score >= max(2, circuit_score + 1):
            return (
                "PCB（进程控制块，Process Control Block）是操作系统中用于描述和管理进程的数据结构，"
                "记录进程状态、程序计数器、寄存器上下文及调度信息，是进程切换和调度的核心依据。"
            )
        if circuit_score >= max(2, process_score + 1):
            return (
                "PCB（印刷电路板，Printed Circuit Board）是用于承载电子元器件并实现电气连接的基础板材，"
                "通过导电线路连接各元件，是电子设备硬件电路的核心载体。"
            )
    return None


def _definition_conflicts_with_context(node_name: str, text: str, contextual_definition: str | None) -> bool:
    if not contextual_definition:
        return False
    key = (node_name or "").strip().lower()
    if key != "pcb":
        return False
    raw = (text or "").lower()
    if "进程控制块" in contextual_definition:
        return any(item in raw for item in ("印刷电路板", "电子元件", "导线", "焊盘", "板材"))
    if "印刷电路板" in contextual_definition:
        return any(item in raw for item in ("进程控制块", "进程状态", "上下文切换", "调度信息"))
    return False


def _build_node_brief_prompt(
    *,
    course_name: str,
    node_name: str,
    related_labels: list[str],
    evidence_lines: list[str],
    contextual_definition: str | None,
) -> str:
    related_text = "、".join(related_labels[:6]) if related_labels else "暂无明显关联"
    if evidence_lines:
        evidence_text = "\n".join([f"{idx + 1}. {line}" for idx, line in enumerate(evidence_lines[:6])])
    else:
        evidence_text = "无可用资料证据"
    preferred_text = contextual_definition or "无"
    return (
        "你是课程知识图谱助教。请为知识点生成‘定义型简介’。\n"
        "要求：\n"
        "1) 必须优先依据课程名称、关联知识点和资料证据，禁止脱离课程语境给通用解释；\n"
        "2) 直接给定义，不要写‘与谁相关’的关系总结；\n"
        "3) 1-2句，40-120字，不使用Markdown和分点；\n"
        "4) 对缩写词先给中文全称（必要时补英文全称）。\n\n"
        f"课程：{course_name}\n"
        f"知识点：{node_name}\n"
        f"关联知识点：{related_text}\n"
        f"资料证据：\n{evidence_text}\n"
        f"推荐释义（若证据支持则优先采用）：{preferred_text}\n\n"
        "只输出最终定义文本。"
    )


def _candidate_status_stats(db: Session, course_id: int) -> dict[str, int]:
    rows = (
        db.query(models.KnowledgeCandidate.status, func.count(models.KnowledgeCandidate.id))
        .filter(models.KnowledgeCandidate.course_id == course_id)
        .group_by(models.KnowledgeCandidate.status)
        .all()
    )
    stats = Counter({status: count for status, count in rows})
    return {
        "pending": int(stats.get("pending", 0)),
        "approved": int(stats.get("approved", 0)),
        "rejected": int(stats.get("rejected", 0)),
    }


def _serialize_candidate(row: models.KnowledgeCandidate) -> KGCandidateOut:
    return KGCandidateOut(
        id=row.id,
        course_id=row.course_id,
        material_id=row.material_id,
        term=row.term,
        source_sentence=row.source_sentence,
        status=row.status,
        extractor=row.extractor,
        fallback_used=bool(row.fallback_used),
        score=row.score,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _edge_min_score() -> float:
    # 兼容旧配置：未配置新阈值时回退到旧字段。
    return float(max(0.0, settings.kg_edge_min_score if settings.kg_edge_min_score is not None else settings.kg_min_cooccur_count))


def _build_pair_evidence(text: str, terms: list[str]) -> dict[tuple[str, str], str]:
    evidence: dict[tuple[str, str], str] = {}
    unique_terms = sorted({normalize_keyword(term) for term in terms if normalize_keyword(term)})
    if len(unique_terms) < 2:
        return evidence

    sentence_pairs: list[tuple[str, set[str], str]] = []
    for sentence in split_sentences(text):
        normalized_sentence = normalize_keyword(sentence)
        if not normalized_sentence:
            sentence_pairs.append((sentence, set(), normalized_sentence))
            continue
        hits = {term for term in unique_terms if term in normalized_sentence}
        sentence_pairs.append((sentence, hits, normalized_sentence))
        if len(hits) < 2:
            continue
        for left in sorted(hits):
            for right in sorted(hits):
                if left >= right:
                    continue
                evidence.setdefault((left, right), sentence[:220])

    # 跨句软证据兜底：用于补充同句证据缺失的情况。
    for idx in range(len(sentence_pairs) - 1):
        left_sentence, left_hits, _ = sentence_pairs[idx]
        right_sentence, right_hits, _ = sentence_pairs[idx + 1]
        if not left_hits or not right_hits:
            continue
        for left in sorted(left_hits):
            for right in sorted(right_hits):
                if left == right:
                    continue
                key = tuple(sorted((left, right)))
                if key in evidence:
                    continue
                evidence[key] = f"{left_sentence[:110]} / {right_sentence[:110]}"
    return evidence


def _build_order_fallback_pairs(text: str, terms: list[str]) -> dict[tuple[str, str], float]:
    unique_terms = sorted({normalize_keyword(term) for term in terms if normalize_keyword(term)})
    if len(unique_terms) < 2:
        return {}

    pairs: dict[tuple[str, str], float] = {}
    max_distance = max(8, int(settings.kg_max_term_distance))
    for sentence in split_sentences(text):
        normalized_sentence = normalize_keyword(sentence)
        if not normalized_sentence:
            continue

        ordered_hits: list[tuple[int, str]] = []
        for term in unique_terms:
            start = normalized_sentence.find(term)
            while start >= 0:
                ordered_hits.append((start, term))
                start = normalized_sentence.find(term, start + 1)

        ordered_hits.sort(key=lambda x: x[0])
        if len(ordered_hits) < 2:
            continue

        for idx in range(len(ordered_hits) - 1):
            left_pos, left_term = ordered_hits[idx]
            for next_idx in range(idx + 1, min(idx + 3, len(ordered_hits))):
                right_pos, right_term = ordered_hits[next_idx]
                if left_term == right_term:
                    continue
                distance = max(1, right_pos - left_pos)
                if distance > max_distance * 2:
                    continue
                key = tuple(sorted((left_term, right_term)))
                # 顺序邻接仅作为弱证据，避免“仅靠顺序”的边被判成强关系。
                score = max(0.03, min(0.16, 12.0 / float(distance)))
                if score > pairs.get(key, 0.0):
                    pairs[key] = score
    return pairs


def _build_pair_context_support(text: str, terms: list[str]) -> dict[tuple[str, str], float]:
    unique_terms = sorted({normalize_keyword(term) for term in terms if normalize_keyword(term)})
    if len(unique_terms) < 2:
        return {}

    term_sentence_hits: Counter[str] = Counter()
    pair_sentence_hits: Counter[tuple[str, str]] = Counter()
    pair_cue_hits: Counter[tuple[str, str]] = Counter()

    for sentence in split_sentences(text):
        normalized_sentence = normalize_keyword(sentence)
        if not normalized_sentence:
            continue

        hits = sorted({term for term in unique_terms if term in normalized_sentence})
        if not hits:
            continue
        for term in hits:
            term_sentence_hits[term] += 1
        if len(hits) < 2:
            continue

        has_relation_cue = any(cue in normalized_sentence for cue in _RELATION_CUE_WORDS)
        for left, right in itertools.combinations(hits, 2):
            key = tuple(sorted((left, right)))
            pair_sentence_hits[key] += 1
            if has_relation_cue:
                pair_cue_hits[key] += 1

    support_scores: dict[tuple[str, str], float] = {}
    for (left, right), pair_count in pair_sentence_hits.items():
        left_hits = max(1, int(term_sentence_hits.get(left, 1)))
        right_hits = max(1, int(term_sentence_hits.get(right, 1)))
        overlap_ratio = pair_count / float(max(1, min(left_hits, right_hits)))
        cue_ratio = pair_cue_hits.get((left, right), 0) / float(pair_count)

        support = min(1.1, 0.42 * overlap_ratio + 0.18 * pair_count + 0.35 * cue_ratio)
        support_scores[(left, right)] = support
    return support_scores


def _blend_pair_scores(
    *,
    cooccur_scores: dict[tuple[str, str], float],
    order_scores: dict[tuple[str, str], float],
    context_scores: dict[tuple[str, str], float],
) -> dict[tuple[str, str], float]:
    blended: dict[tuple[str, str], float] = {}
    for pair in set(cooccur_scores) | set(order_scores) | set(context_scores):
        blended[pair] = (
            float(cooccur_scores.get(pair, 0.0))
            + float(order_scores.get(pair, 0.0))
            + float(context_scores.get(pair, 0.0))
        )
    return blended


def _build_pair_scores(text: str, terms: list[str]) -> dict[tuple[str, str], float]:
    cooccur_scores = build_sentence_cooccurrence(text=text, terms=terms)
    order_scores = (
        _build_order_fallback_pairs(text=text, terms=terms)
        if settings.kg_enable_order_fallback_pairs
        else {}
    )
    context_scores = _build_pair_context_support(text=text, terms=terms)
    return _blend_pair_scores(
        cooccur_scores=cooccur_scores,
        order_scores=order_scores,
        context_scores=context_scores,
    )


def _upsert_relation(
    *,
    course_id: int,
    source_id: int,
    target_id: int,
    relation: str,
    material_id: int | None,
    score: float | None,
    evidence_sentence: str | None,
    is_weak: bool,
    extractor: str | None,
    existing_edges: dict[tuple[int, int, str], models.KnowledgeRelation],
    db: Session,
) -> bool:
    source_id, target_id = sorted((source_id, target_id))
    if source_id == target_id:
        return False

    key = (source_id, target_id, relation)
    exists = existing_edges.get(key)
    if exists:
        # 手工维护关系不被自动流程覆盖。
        if (exists.extractor or "").lower() == "manual":
            return False
        current_score = float(exists.cooccur_score or 0.0)
        if score is not None and score >= current_score:
            exists.material_id = material_id
            exists.cooccur_score = score
            exists.evidence_sentence = evidence_sentence
            exists.is_weak = 1 if is_weak else 0
            exists.extractor = extractor
        return False

    row = models.KnowledgeRelation(
        course_id=course_id,
        source_id=source_id,
        target_id=target_id,
        relation=relation,
        material_id=material_id,
        cooccur_score=score,
        evidence_sentence=evidence_sentence,
        is_weak=1 if is_weak else 0,
        extractor=extractor,
    )
    db.add(row)
    db.flush()
    existing_edges[key] = row
    return True


def _build_edges_for_material(
    *,
    db: Session,
    course_id: int,
    material_id: int,
    points_by_norm: dict[str, models.KnowledgePoint],
    existing_edges: dict[tuple[int, int, str], models.KnowledgeRelation],
) -> dict[str, int | str | bool]:
    material = _require_material_of_course(db, course_id=course_id, material_id=material_id)
    approved_rows = (
        db.query(models.KnowledgeCandidate)
        .filter_by(course_id=course_id, material_id=material_id, status="approved")
        .all()
    )
    if not approved_rows:
        return {
            "approved_terms_in_material": 0,
            "edge_ready_pairs": 0,
            "edge_blocked_by_threshold": 0,
            "created_edges": 0,
            "weak_edges_created": 0,
            "extractor_used": "hybrid",
            "fallback_used": False,
        }

    extractor_counter = Counter((row.extractor or "hybrid") for row in approved_rows)
    extractor_used = extractor_counter.most_common(1)[0][0]
    fallback_used = any(bool(row.fallback_used) for row in approved_rows)

    material_terms = sorted({row.term for row in approved_rows if row.term})
    approved_norm_terms = sorted({row.term_norm or normalize_keyword(row.term) for row in approved_rows if row.term_norm or row.term})
    approved_terms_in_material = len([term for term in approved_norm_terms if term in points_by_norm])
    if len(material_terms) < 2:
        return {
            "approved_terms_in_material": approved_terms_in_material,
            "edge_ready_pairs": 0,
            "edge_blocked_by_threshold": 0,
            "created_edges": 0,
            "weak_edges_created": 0,
            "extractor_used": extractor_used,
            "fallback_used": fallback_used,
        }

    text = extract_text(material.file_path)
    cooccur_scores = build_sentence_cooccurrence(text=text, terms=material_terms)
    order_fallback = _build_order_fallback_pairs(text=text, terms=material_terms) if settings.kg_enable_order_fallback_pairs else {}
    context_scores = _build_pair_context_support(text=text, terms=material_terms)
    pair_scores = _blend_pair_scores(
        cooccur_scores=cooccur_scores,
        order_scores=order_fallback,
        context_scores=context_scores,
    )
    evidence_by_pair = _build_pair_evidence(text=text, terms=material_terms)
    min_score = _edge_min_score()
    # 强关系必须过阈值；被阈值拦截的候选对会返回给前端做诊断展示。
    ready_pairs = {pair: score for pair, score in pair_scores.items() if score >= min_score}
    blocked_by_threshold = max(0, len(pair_scores) - len(ready_pairs))

    created_edges = 0
    weak_edges_created = 0
    approved_norm_set = {term for term in approved_norm_terms if term in points_by_norm}
    term_degree = {term: 0 for term in approved_norm_set}
    point_id_to_norm = {point.id: norm for norm, point in points_by_norm.items() if norm in approved_norm_set}

    for edge in existing_edges.values():
        left_term = point_id_to_norm.get(edge.source_id)
        right_term = point_id_to_norm.get(edge.target_id)
        if not left_term or not right_term or left_term == right_term:
            continue
        term_degree[left_term] = term_degree.get(left_term, 0) + 1
        term_degree[right_term] = term_degree.get(right_term, 0) + 1

    for (left_key, right_key), score in sorted(ready_pairs.items(), key=lambda x: x[1], reverse=True):
        source = points_by_norm.get(left_key)
        target = points_by_norm.get(right_key)
        if not source or not target:
            continue
        evidence = evidence_by_pair.get((left_key, right_key))
        if not evidence and (left_key, right_key) in order_fallback:
            evidence = "顺序邻接弱证据（同文档近邻）"
        created = _upsert_relation(
            course_id=course_id,
            source_id=source.id,
            target_id=target.id,
            relation="relates_to",
            material_id=material_id,
            score=float(score),
            evidence_sentence=evidence,
            is_weak=False,
            extractor=extractor_used,
            existing_edges=existing_edges,
            db=db,
        )
        if created:
            created_edges += 1
            if left_key in term_degree:
                term_degree[left_key] += 1
            if right_key in term_degree:
                term_degree[right_key] += 1

    min_edges_target = max(0, int(settings.kg_min_edges_per_material))
    min_degree_target = max(0, int(settings.kg_min_degree_per_term))
    need_more_edges = created_edges < min_edges_target
    need_more_degree = min_degree_target > 0 and any(
        term_degree.get(term, 0) < min_degree_target for term in term_degree
    )

    if (need_more_edges or need_more_degree) and approved_terms_in_material >= 2:
        # 弱边兜底：语料较少时保证图连通性，提升演示稳定性。
        weak_top_n = max(1, int(settings.kg_weak_edge_top_n))
        weak_min_score = max(0.0, settings.kg_weak_edge_min_score)
        weak_limit = max(
            weak_top_n,
            max(0, min_edges_target - created_edges),
            min_degree_target * max(1, len(term_degree) // 2),
        )

        weak_candidates = [
            (pair, score)
            for pair, score in sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
            if score >= weak_min_score
            and pair not in ready_pairs
        ]

        for (left_key, right_key), score in weak_candidates:
            left_missing = term_degree.get(left_key, 0) < min_degree_target if min_degree_target > 0 else False
            right_missing = term_degree.get(right_key, 0) < min_degree_target if min_degree_target > 0 else False
            degree_unmet = left_missing or right_missing
            edges_unmet = created_edges < min_edges_target
            if not degree_unmet and min_degree_target > 0:
                if term_degree.get(left_key, 0) >= min_degree_target + 2 and term_degree.get(right_key, 0) >= min_degree_target + 2:
                    continue
            if weak_edges_created >= weak_limit and not degree_unmet and not edges_unmet:
                break

            source = points_by_norm.get(left_key)
            target = points_by_norm.get(right_key)
            if not source or not target:
                continue
            evidence = evidence_by_pair.get((left_key, right_key))
            if not evidence and (left_key, right_key) in order_fallback:
                evidence = "顺序邻接弱证据（同文档近邻）"
            created = _upsert_relation(
                course_id=course_id,
                source_id=source.id,
                target_id=target.id,
                relation="relates_to",
                material_id=material_id,
                score=float(score),
                evidence_sentence=evidence,
                is_weak=True,
                extractor=extractor_used,
                existing_edges=existing_edges,
                db=db,
            )
            if created:
                created_edges += 1
                weak_edges_created += 1
                if left_key in term_degree:
                    term_degree[left_key] += 1
                if right_key in term_degree:
                    term_degree[right_key] += 1

    return {
        "approved_terms_in_material": approved_terms_in_material,
        "edge_ready_pairs": len(ready_pairs),
        "edge_blocked_by_threshold": blocked_by_threshold,
        "created_edges": created_edges,
        "weak_edges_created": weak_edges_created,
        "extractor_used": extractor_used,
        "fallback_used": fallback_used,
    }


def _build_cross_material_edges(
    *,
    db: Session,
    course_id: int,
    points_by_norm: dict[str, models.KnowledgePoint],
    existing_edges: dict[tuple[int, int, str], models.KnowledgeRelation],
) -> int:
    rows = (
        db.query(
            models.KnowledgeCandidate.material_id,
            models.KnowledgeCandidate.term_norm,
        )
        .filter(
            models.KnowledgeCandidate.course_id == course_id,
            models.KnowledgeCandidate.status == "approved",
        )
        .all()
    )
    if not rows:
        return 0

    terms_by_material: dict[int, set[str]] = {}
    for material_id, term_norm in rows:
        norm = normalize_keyword(term_norm or "")
        if not norm or norm not in points_by_norm:
            continue
        terms_by_material.setdefault(int(material_id), set()).add(norm)

    if not terms_by_material:
        return 0

    pair_counter: Counter[tuple[str, str]] = Counter()
    for terms in terms_by_material.values():
        if len(terms) < 2:
            continue
        for left, right in itertools.combinations(sorted(terms), 2):
            pair_counter[(left, right)] += 1

    min_support = max(2, int(settings.kg_min_cooccur_count))
    weak_top_n = max(1, int(settings.kg_weak_edge_top_n))
    candidates = [
        (pair, count)
        for pair, count in sorted(pair_counter.items(), key=lambda x: x[1], reverse=True)
        if count >= min_support
    ][: max(weak_top_n * 2, 6)]

    created = 0
    for (left_key, right_key), count in candidates:
        source = points_by_norm.get(left_key)
        target = points_by_norm.get(right_key)
        if not source or not target:
            continue
        inserted = _upsert_relation(
            course_id=course_id,
            source_id=source.id,
            target_id=target.id,
            relation="cross_material_related",
            material_id=None,
            score=float(count),
            evidence_sentence=f"跨资料共现 {count} 次",
            is_weak=True,
            extractor="cross_material",
            existing_edges=existing_edges,
            db=db,
        )
        if inserted:
            created += 1
    return created


def _query_candidates_for_action(
    db: Session,
    course_id: int,
    payload: KGCandidateBatchActionRequest,
    only_status: str | None = None,
):
    query = db.query(models.KnowledgeCandidate).filter(models.KnowledgeCandidate.course_id == course_id)
    if payload.candidate_ids:
        query = query.filter(models.KnowledgeCandidate.id.in_(payload.candidate_ids))
    elif payload.material_id is not None:
        query = query.filter(models.KnowledgeCandidate.material_id == payload.material_id)
    else:
        raise HTTPException(status_code=422, detail="candidate_ids or material_id is required")

    if only_status:
        query = query.filter(models.KnowledgeCandidate.status == only_status)

    return query.all()


@router.get("/{course_id}", response_model=KGResponse, summary="获取知识图谱")
def get_kg(course_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    course = db.query(models.Course).get(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    nodes = db.query(models.KnowledgePoint).filter_by(course_id=course_id).all()
    edges = db.query(models.KnowledgeRelation).filter_by(course_id=course_id).all()
    if not edges:
        # Compatibility fallback for legacy table name knowledge_edges.
        # 兼容旧表 knowledge_edges，避免历史数据升级后直接丢失展示。
        try:
            legacy_rows = db.execute(
                text(
                    "SELECT source_id, target_id, relation FROM knowledge_edges WHERE course_id = :course_id"
                ),
                {"course_id": course_id},
            ).fetchall()
        except Exception:
            legacy_rows = []
        if legacy_rows:
            return {
                "nodes": [{"id": n.id, "label": n.name, "description": n.description} for n in nodes],
                "edges": [
                    {
                        "source": row.source_id,
                        "target": row.target_id,
                        "relation": row.relation,
                        "material_id": None,
                        "cooccur_score": None,
                        "evidence_sentence": None,
                        "is_weak": False,
                        "extractor": None,
                    }
                    for row in legacy_rows
                ],
            }
    return {
        "nodes": [{"id": n.id, "label": n.name, "description": n.description} for n in nodes],
        "edges": [
            {
                "source": e.source_id,
                "target": e.target_id,
                "relation": e.relation,
                "material_id": e.material_id,
                "cooccur_score": e.cooccur_score,
                "evidence_sentence": e.evidence_sentence,
                "is_weak": bool(e.is_weak),
                "extractor": e.extractor,
            }
            for e in edges
        ],
    }


@router.post(
    "/{course_id}/nodes/{node_id}/brief",
    response_model=KGNodeBriefOut,
    summary="生成或获取节点简要定义",
)
async def generate_node_brief(
    course_id: int,
    node_id: int,
    force_refresh: bool = False,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    course = db.query(models.Course).get(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    node = db.query(models.KnowledgePoint).filter_by(course_id=course_id, id=node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Knowledge point not found")

    existing_description = (node.description or "").strip()

    related_edges = (
        db.query(models.KnowledgeRelation)
        .filter(
            models.KnowledgeRelation.course_id == course_id,
            (models.KnowledgeRelation.source_id == node_id)
            | (models.KnowledgeRelation.target_id == node_id),
        )
        .limit(16)
        .all()
    )
    related_node_ids = {
        edge.source_id if edge.source_id != node_id else edge.target_id
        for edge in related_edges
        if edge.source_id != edge.target_id
    }
    related_nodes = (
        db.query(models.KnowledgePoint)
        .filter(models.KnowledgePoint.course_id == course_id, models.KnowledgePoint.id.in_(related_node_ids))
        .all()
        if related_node_ids
        else []
    )
    related_labels = [item.name for item in related_nodes][:8]
    evidence_lines = _collect_node_brief_evidence(
        db,
        course_id=course_id,
        node_name=node.name,
        related_edges=related_edges,
        limit=6,
    )
    contextual_definition = _resolve_contextual_definition(
        course_name=course.name or f"课程{course_id}",
        node_name=node.name,
        related_labels=related_labels,
        evidence_lines=evidence_lines,
    )
    if existing_description and not force_refresh:
        if contextual_definition and _definition_conflicts_with_context(node.name, existing_description, contextual_definition):
            node.description = contextual_definition
            db.commit()
            return {
                "node_id": node.id,
                "label": node.name,
                "description": contextual_definition,
                "generated": True,
                "from_cache": False,
            }
        return {
            "node_id": node.id,
            "label": node.name,
            "description": existing_description,
            "generated": False,
            "from_cache": True,
        }

    prompt = _build_node_brief_prompt(
        course_name=course.name or f"课程{course_id}",
        node_name=node.name,
        related_labels=related_labels,
        evidence_lines=evidence_lines,
        contextual_definition=contextual_definition,
    )
    llm_text = await call_llm(prompt)
    cleaned = _clean_llm_definition(llm_text)

    failed = (
        (not cleaned)
        or cleaned.startswith("[")
        or "接口错误" in cleaned
        or "占位答复" in cleaned
    )
    if contextual_definition and (failed or _definition_conflicts_with_context(node.name, cleaned, contextual_definition)):
        cleaned = contextual_definition
    elif failed:
        cleaned = _fallback_definition(node.name, course.name or f"课程{course_id}")

    node.description = cleaned
    db.commit()

    return {
        "node_id": node.id,
        "label": node.name,
        "description": cleaned,
        "generated": True,
        "from_cache": False,
    }


@router.get("/{course_id}/candidates", response_model=KGCandidateListOut, summary="查询知识图谱候选池")
def list_candidates(
    course_id: int,
    status: str | None = None,
    material_id: int | None = None,
    page: int = 1,
    page_size: int = 50,
    db: Session = Depends(get_db),
    teacher=Depends(require_teacher),
):
    _require_teacher_course(db, teacher.id, course_id)

    page = max(1, page)
    page_size = max(1, min(page_size, 200))

    query = db.query(models.KnowledgeCandidate).filter(models.KnowledgeCandidate.course_id == course_id)
    if status:
        query = query.filter(models.KnowledgeCandidate.status == status)
    if material_id is not None:
        query = query.filter(models.KnowledgeCandidate.material_id == material_id)

    total = query.count()
    rows = (
        query.order_by(models.KnowledgeCandidate.updated_at.desc(), models.KnowledgeCandidate.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return KGCandidateListOut(
        items=[_serialize_candidate(row) for row in rows],
        total=total,
        page=page,
        page_size=page_size,
        status_stats=_candidate_status_stats(db, course_id),
    )


@router.post("/{course_id}/points", summary="新增知识点")
def create_point(course_id: int, name: str, db: Session = Depends(get_db), teacher=Depends(require_teacher)):
    _require_teacher_course(db, teacher.id, course_id)
    cleaned_name = (name or "").strip()
    if not cleaned_name:
        raise HTTPException(status_code=422, detail="name is required")
    exists = db.query(models.KnowledgePoint).filter_by(course_id=course_id, name=cleaned_name).first()
    if exists:
        return {"id": exists.id, "name": exists.name}
    kp = models.KnowledgePoint(course_id=course_id, name=cleaned_name)
    db.add(kp)
    db.commit()
    db.refresh(kp)
    return {"id": kp.id, "name": kp.name}


@router.post("/{course_id}/edges", summary="新增知识关系")
def create_edge(
    course_id: int,
    source_id: int,
    target_id: int,
    relation: str = "relates_to",
    db: Session = Depends(get_db),
    teacher=Depends(require_teacher),
):
    _require_teacher_course(db, teacher.id, course_id)
    if source_id == target_id:
        raise HTTPException(status_code=422, detail="source_id and target_id cannot be the same")

    source = db.query(models.KnowledgePoint).filter_by(course_id=course_id, id=source_id).first()
    target = db.query(models.KnowledgePoint).filter_by(course_id=course_id, id=target_id).first()
    if not source or not target:
        raise HTTPException(status_code=404, detail="Knowledge point not found")

    rel = (relation or "relates_to").strip() or "relates_to"
    exists = (
        db.query(models.KnowledgeRelation)
        .filter_by(course_id=course_id, source_id=source_id, target_id=target_id, relation=rel)
        .first()
    )
    if exists:
        return {"id": exists.id}

    edge = models.KnowledgeRelation(
        course_id=course_id,
        source_id=source_id,
        target_id=target_id,
        relation=rel,
        material_id=None,
        cooccur_score=None,
        evidence_sentence=None,
        is_weak=0,
        extractor="manual",
    )
    db.add(edge)
    db.commit()
    db.refresh(edge)
    return {"id": edge.id}


@router.post("/{course_id}/extract", summary="自动抽取知识点")
def extract_points(course_id: int, material_id: int, top_k: int = 10, db: Session = Depends(get_db), teacher=Depends(require_teacher)):
    _require_teacher_course(db, teacher.id, course_id)
    material = _require_material_of_course(db, course_id=course_id, material_id=material_id)
    text = extract_text(material.file_path)
    effective_top_k = top_k if top_k > 0 else settings.kg_top_k
    keywords = extract_keywords(text, top_k=effective_top_k)
    created = []
    for kw in keywords:
        if not db.query(models.KnowledgePoint).filter_by(course_id=course_id, name=kw).first():
            kp = models.KnowledgePoint(course_id=course_id, name=kw)
            db.add(kp)
            db.flush()
            created.append(kw)
    db.commit()
    return {"created": created}


@router.post("/{course_id}/candidates", response_model=KGCandidatesResponse, summary="生成知识图谱候选点")
def build_candidates(
    course_id: int,
    material_id: int,
    top_k: int = 12,
    auto_create: bool = False,
    auto_link: bool = True,
    db: Session = Depends(get_db),
    teacher=Depends(require_teacher),
):
    _require_teacher_course(db, teacher.id, course_id)
    material = _require_material_of_course(db, course_id=course_id, material_id=material_id)
    text = extract_text(material.file_path)
    effective_top_k = top_k if top_k > 0 else settings.kg_top_k
    start_ts = time.perf_counter()
    extract_result = extract_keywords_with_meta(text, top_k=effective_top_k)
    extract_ms = int((time.perf_counter() - start_ts) * 1000)
    logger.info(
        "kg_extract_candidates material=%s extractor_used=%s fallback=%s extract_ms=%s",
        material_id,
        extract_result.extractor,
        extract_result.fallback_used,
        extract_ms,
    )
    candidates = extract_result.candidates

    created: list[str] = []
    candidate_ids: list[int] = []
    seen_norms: set[str] = set()

    # 兼容：参数仍保留，但本接口只生成候选，不直接改主图。
    # 这样可以先审核候选再入图，减少脏节点和伪关系。
    _ = (auto_create, auto_link)

    for idx, term in enumerate(candidates):
        term_norm = normalize_keyword(term)
        if not term_norm:
            continue
        if term_norm in seen_norms:
            continue
        seen_norms.add(term_norm)
        source_sentence = find_source_sentence(text, term)
        score = float(max(0, len(candidates) - idx))

        row = (
            db.query(models.KnowledgeCandidate)
            .filter_by(course_id=course_id, material_id=material_id, term_norm=term_norm)
            .first()
        )
        if not row:
            row = models.KnowledgeCandidate(
                course_id=course_id,
                material_id=material_id,
                term=term,
                term_norm=term_norm,
                source_sentence=source_sentence,
                status="pending",
                extractor=extract_result.extractor,
                fallback_used=1 if extract_result.fallback_used else 0,
                score=score,
            )
            db.add(row)
            db.flush()
            created.append(term)
        else:
            row.term = term
            row.source_sentence = source_sentence
            row.extractor = extract_result.extractor
            row.fallback_used = 1 if extract_result.fallback_used else 0
            row.score = score

        candidate_ids.append(row.id)

    db.commit()

    preview_scores = _build_pair_scores(text=text, terms=candidates)
    preview_cooccur_pairs = len(preview_scores)
    edge_ready_pairs = sum(1 for score in preview_scores.values() if score >= _edge_min_score())
    edge_blocked = max(0, preview_cooccur_pairs - edge_ready_pairs)
    return {
        "candidates": candidates,
        "created": created,
        "created_nodes": 0,
        "created_edges": 0,
        "filtered_noise": extract_result.filtered_noise,
        "cooccur_pairs": preview_cooccur_pairs,
        "extractor": extract_result.extractor,
        "fallback_used": extract_result.fallback_used,
        "candidate_ids": candidate_ids,
        "status_stats": _candidate_status_stats(db, course_id),
        "edge_ready_pairs": edge_ready_pairs,
        "edge_blocked_by_threshold": edge_blocked,
    }


@router.post("/{course_id}/candidates/approve", response_model=KGCandidateBatchActionOut, summary="批量通过候选并入图")
def approve_candidates(
    course_id: int,
    payload: KGCandidateBatchActionRequest,
    db: Session = Depends(get_db),
    teacher=Depends(require_teacher),
):
    _require_teacher_course(db, teacher.id, course_id)
    candidates = _query_candidates_for_action(db, course_id, payload, only_status="pending")
    if not candidates:
        return {"affected": 0, "created_nodes": 0, "created_edges": 0}

    points_by_norm = {
        normalize_keyword(point.name): point
        for point in db.query(models.KnowledgePoint).filter_by(course_id=course_id).all()
    }

    created_nodes = 0
    for candidate in candidates:
        candidate.status = "approved"
        norm = candidate.term_norm or normalize_keyword(candidate.term)
        if norm not in points_by_norm:
            point = models.KnowledgePoint(course_id=course_id, name=candidate.term)
            db.add(point)
            db.flush()
            points_by_norm[norm] = point
            created_nodes += 1

    existing_edges = {
        (row.source_id, row.target_id, row.relation): row
        for row in db.query(models.KnowledgeRelation).filter_by(course_id=course_id).all()
    }

    created_edges = 0
    approved_terms_in_material = 0
    edge_ready_pairs = 0
    edge_blocked_by_threshold = 0
    weak_edges_created = 0
    extractor_used_set: set[str] = set()
    fallback_used = False

    material_ids = sorted({candidate.material_id for candidate in candidates})
    # 先按资料内建边，再补跨资料弱连接，兼顾精度与整体连通性。
    for material_id in material_ids:
        start_ts = time.perf_counter()
        stats = _build_edges_for_material(
            db=db,
            course_id=course_id,
            material_id=material_id,
            points_by_norm=points_by_norm,
            existing_edges=existing_edges,
        )
        elapsed_ms = int((time.perf_counter() - start_ts) * 1000)
        logger.info(
            "kg_extract_relation material=%s extractor_used=%s fallback=%s extract_ms=%s",
            material_id,
            stats.get("extractor_used"),
            stats.get("fallback_used"),
            elapsed_ms,
        )
        created_edges += int(stats.get("created_edges", 0))
        approved_terms_in_material += int(stats.get("approved_terms_in_material", 0))
        edge_ready_pairs += int(stats.get("edge_ready_pairs", 0))
        edge_blocked_by_threshold += int(stats.get("edge_blocked_by_threshold", 0))
        weak_edges_created += int(stats.get("weak_edges_created", 0))
        extractor_used_set.add(str(stats.get("extractor_used", "hybrid")))
        fallback_used = fallback_used or bool(stats.get("fallback_used", False))

    cross_material_edges = _build_cross_material_edges(
        db=db,
        course_id=course_id,
        points_by_norm=points_by_norm,
        existing_edges=existing_edges,
    )
    if cross_material_edges > 0:
        created_edges += cross_material_edges
        weak_edges_created += cross_material_edges
        extractor_used_set.add("cross_material")

    db.commit()
    return {
        "affected": len(candidates),
        "created_nodes": created_nodes,
        "created_edges": created_edges,
        "approved_terms_in_material": approved_terms_in_material,
        "edge_ready_pairs": edge_ready_pairs,
        "edge_blocked_by_threshold": edge_blocked_by_threshold,
        "weak_edges_created": weak_edges_created,
        "extractor_used": "+".join(sorted(extractor_used_set)) if extractor_used_set else "hybrid",
        "fallback_used": fallback_used,
    }


@router.post("/{course_id}/relations/rebuild", response_model=KGCandidateBatchActionOut, summary="重建图谱关系")
def rebuild_relations(
    course_id: int,
    material_id: int | None = None,
    db: Session = Depends(get_db),
    teacher=Depends(require_teacher),
):
    _require_teacher_course(db, teacher.id, course_id)
    if material_id is not None:
        _require_material_of_course(db, course_id=course_id, material_id=material_id)
        material_ids = [material_id]
    else:
        material_ids = [
            row[0]
            for row in (
                db.query(models.KnowledgeCandidate.material_id)
                .filter_by(course_id=course_id, status="approved")
                .group_by(models.KnowledgeCandidate.material_id)
                .all()
            )
        ]

    if not material_ids:
        return {
            "affected": 0,
            "created_nodes": 0,
            "created_edges": 0,
            "approved_terms_in_material": 0,
            "edge_ready_pairs": 0,
            "edge_blocked_by_threshold": 0,
            "weak_edges_created": 0,
            "extractor_used": "hybrid",
            "fallback_used": False,
        }

    db.query(models.KnowledgeRelation).filter(
        models.KnowledgeRelation.course_id == course_id,
        models.KnowledgeRelation.material_id.in_(material_ids),
    ).delete(synchronize_session=False)
    db.flush()

    points_by_norm = {
        normalize_keyword(point.name): point
        for point in db.query(models.KnowledgePoint).filter_by(course_id=course_id).all()
    }
    existing_edges = {
        (row.source_id, row.target_id, row.relation): row
        for row in db.query(models.KnowledgeRelation).filter_by(course_id=course_id).all()
    }

    affected = 0
    created_edges = 0
    approved_terms_in_material = 0
    edge_ready_pairs = 0
    edge_blocked_by_threshold = 0
    weak_edges_created = 0
    extractor_used_set: set[str] = set()
    fallback_used = False

    for item_material_id in material_ids:
        affected += (
            db.query(models.KnowledgeCandidate)
            .filter_by(course_id=course_id, material_id=item_material_id, status="approved")
            .count()
        )
        start_ts = time.perf_counter()
        stats = _build_edges_for_material(
            db=db,
            course_id=course_id,
            material_id=item_material_id,
            points_by_norm=points_by_norm,
            existing_edges=existing_edges,
        )
        elapsed_ms = int((time.perf_counter() - start_ts) * 1000)
        logger.info(
            "kg_rebuild_relation material=%s extractor_used=%s fallback=%s extract_ms=%s",
            item_material_id,
            stats.get("extractor_used"),
            stats.get("fallback_used"),
            elapsed_ms,
        )
        created_edges += int(stats.get("created_edges", 0))
        approved_terms_in_material += int(stats.get("approved_terms_in_material", 0))
        edge_ready_pairs += int(stats.get("edge_ready_pairs", 0))
        edge_blocked_by_threshold += int(stats.get("edge_blocked_by_threshold", 0))
        weak_edges_created += int(stats.get("weak_edges_created", 0))
        extractor_used_set.add(str(stats.get("extractor_used", "hybrid")))
        fallback_used = fallback_used or bool(stats.get("fallback_used", False))

    cross_material_edges = _build_cross_material_edges(
        db=db,
        course_id=course_id,
        points_by_norm=points_by_norm,
        existing_edges=existing_edges,
    )
    if cross_material_edges > 0:
        created_edges += cross_material_edges
        weak_edges_created += cross_material_edges
        extractor_used_set.add("cross_material")

    db.commit()
    return {
        "affected": affected,
        "created_nodes": 0,
        "created_edges": created_edges,
        "approved_terms_in_material": approved_terms_in_material,
        "edge_ready_pairs": edge_ready_pairs,
        "edge_blocked_by_threshold": edge_blocked_by_threshold,
        "weak_edges_created": weak_edges_created,
        "extractor_used": "+".join(sorted(extractor_used_set)) if extractor_used_set else "hybrid",
        "fallback_used": fallback_used,
    }


@router.post("/{course_id}/candidates/reject", response_model=KGCandidateBatchActionOut, summary="批量驳回候选")
def reject_candidates(
    course_id: int,
    payload: KGCandidateBatchActionRequest,
    db: Session = Depends(get_db),
    teacher=Depends(require_teacher),
):
    _require_teacher_course(db, teacher.id, course_id)
    candidates = _query_candidates_for_action(db, course_id, payload, only_status="pending")
    for candidate in candidates:
        candidate.status = "rejected"
    if candidates:
        db.commit()
    return {
        "affected": len(candidates),
        "created_nodes": 0,
        "created_edges": 0,
    }





