from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
import re
import unicodedata

from sqlalchemy.orm import Session

from app.config import settings
from app.models import models


DEFAULT_WINDOW_DAYS = 14
DEFAULT_BUCKET_LIMIT = 3
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,10}|[A-Za-z]{3,24}")
_TOKEN_CLEAN_RE = re.compile(r"^[\W_]+|[\W_]+$")
_NOISE_PATH_RE = re.compile(r"(?:[a-zA-Z]:\\|/|\\).+")
_GENERIC_FALLBACK_POINTS = [
    "核心概念复习",
    "典型题型训练",
    "章节知识串联",
    "综合应用练习",
    "高频错题复盘",
    "课后拓展阅读",
]
_STOPWORDS = {
    "什么",
    "这个",
    "那个",
    "一个",
    "一种",
    "我们",
    "你们",
    "他们",
    "如何",
    "为什么",
    "可以",
    "应该",
    "关于",
    "进行",
    "课程",
    "学习",
    "知识",
    "问题",
    "系统",
    "数据",
    "模型",
    "内容",
    "说明",
}


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.lower().strip()
    return re.sub(r"\s+", "", normalized)


def _resolve_data_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return path
    backend_root = Path(__file__).resolve().parents[2]
    for candidate in (path, backend_root / path_value):
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=1)
def _load_stopwords() -> set[str]:
    out = set(_STOPWORDS)
    path = _resolve_data_path(settings.kg_stopwords_path)
    if not path:
        return out
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            out.add(_normalize_text(token))
    except Exception:
        return out
    return out


@lru_cache(maxsize=1)
def _load_domain_lexicon() -> set[str]:
    out: set[str] = set()
    path = _resolve_data_path(settings.kg_domain_lexicon_path)
    if not path:
        return out
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            out.add(_normalize_text(token))
    except Exception:
        return out
    return out


def _term_hit(term_norm: str, text: str) -> bool:
    if not term_norm or not text:
        return False
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return term_norm in normalized


def _safe_score(score) -> float | None:
    try:
        if score is None:
            return None
        return float(score)
    except Exception:
        return None


def _extract_fallback_topics(text: str, limit: int = 8) -> list[str]:
    stopwords = _load_stopwords()
    domain_lexicon = _load_domain_lexicon()
    counter: Counter[str] = Counter()
    for token in _TOKEN_RE.findall(text or ""):
        cleaned = _TOKEN_CLEAN_RE.sub("", token.strip())
        normalized = _normalize_text(cleaned)
        if not normalized:
            continue
        if len(normalized) < 2 or normalized.isdigit():
            continue
        if normalized in stopwords:
            continue
        if _NOISE_PATH_RE.match(cleaned):
            continue
        counter[normalized] += 1

    if not counter:
        return []

    ranked = sorted(
        counter.items(),
        key=lambda item: (
            1 if item[0] in domain_lexicon else 0,
            item[1],
            len(item[0]),
        ),
        reverse=True,
    )
    return [term for term, _ in ranked[: max(1, limit)]]


def _material_refs(material_ids: set[int], material_map: dict[int, dict], limit: int = 3) -> list[dict]:
    refs: list[dict] = []
    for material_id in sorted(material_ids):
        item = material_map.get(material_id)
        if not item:
            continue
        refs.append(
            {
                "id": item["id"],
                "title": item["title"],
                "parse_status": item["parse_status"],
            }
        )
        if len(refs) >= limit:
            break
    return refs


def _build_report_item(
    *,
    knowledge_point: str,
    priority: str,
    weakness_score: float,
    mastery_score: float,
    expansion_score: float,
    confidence: float,
    evidence: list[dict],
    material_ids: set[int],
    material_map: dict[int, dict],
) -> dict:
    reason_parts: list[str] = []
    if weakness_score >= mastery_score + 0.8:
        reason_parts.append("近期存在薄弱表现，建议优先复习")
    elif mastery_score >= weakness_score + 0.6:
        reason_parts.append("当前掌握较好，可适度拓展")
    else:
        reason_parts.append("建议继续巩固核心概念与典型题型")

    if confidence < 0.35:
        reason_parts.append("行为数据偏少，结论置信度较低")

    return {
        "knowledge_point": knowledge_point,
        "reason": "；".join(reason_parts),
        "priority": priority,
        "weakness_score": round(max(0.0, weakness_score), 4),
        "mastery_score": round(max(0.0, mastery_score), 4),
        "expansion_score": round(max(0.0, expansion_score), 4),
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "evidence": evidence[:4],
        "recommended_materials": _material_refs(material_ids, material_map, limit=3),
    }


def _build_fallback_report(
    *,
    corpus_text: str,
    bucket_limit: int,
    generated_at: datetime,
    window_days: int,
) -> dict:
    topics = _extract_fallback_topics(corpus_text, limit=max(bucket_limit * 2, 6))
    if not topics:
        topics = _GENERIC_FALLBACK_POINTS[: max(bucket_limit * 2, 4)]

    must_review: list[dict] = []
    need_explore: list[dict] = []
    need_consolidate: list[dict] = []

    for idx, topic in enumerate(topics):
        base = float(max(1, len(topics) - idx))
        item = {
            "knowledge_point": topic,
            "reason": "课程知识图谱尚未构建，基于近期问答与作业高频主题生成建议",
            "priority": "medium",
            "weakness_score": round(base * 0.6, 4),
            "mastery_score": round(base * 0.5, 4),
            "expansion_score": round(base * 0.4, 4),
            "confidence": 0.3,
            "evidence": [
                {
                    "source": "fallback",
                    "value": "缺少可用知识图谱，已启用高频主题兜底推荐",
                    "weight": 1.0,
                }
            ],
            "recommended_materials": [],
        }
        if idx % 2 == 0 and len(must_review) < bucket_limit:
            must_review.append(item)
        elif len(need_explore) < bucket_limit:
            need_explore.append(item)
        elif len(need_consolidate) < bucket_limit:
            need_consolidate.append(item)

    return {
        "summary": {
            "risk_level": "medium",
            "activity_level": "low",
            "main_weak_points": [item["knowledge_point"] for item in must_review[:2]],
            "generated_at": generated_at,
            "window_days": window_days,
            "note": "当前课程尚无稳定知识图谱，推荐结果基于兜底主题抽取。",
        },
        "must_review": must_review[:bucket_limit],
        "need_consolidate": need_consolidate[:bucket_limit],
        "need_explore": need_explore[:bucket_limit],
    }


def _compute_activity_level(*, qa_recent: int, submissions_recent: int, events_recent: int) -> str:
    score = qa_recent + submissions_recent * 2 + events_recent * 0.5
    if score >= 12:
        return "high"
    if score >= 5:
        return "medium"
    return "low"


def _compute_risk_level(must_review_count: int, avg_weakness: float) -> str:
    if must_review_count >= 3 or avg_weakness >= 3.0:
        return "high"
    if must_review_count >= 1 or avg_weakness >= 1.8:
        return "medium"
    return "low"


def build_learning_report(
    db: Session,
    user_id: int,
    course_id: int,
    *,
    bucket_limit: int = DEFAULT_BUCKET_LIMIT,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> dict:
    bucket_limit = max(1, bucket_limit)
    window_days = max(1, window_days)
    now = datetime.utcnow()
    recent_threshold = now - timedelta(days=window_days)

    qa_records = (
        db.query(models.QARecord)
        .filter(models.QARecord.user_id == user_id, models.QARecord.course_id == course_id)
        .order_by(models.QARecord.created_at.desc())
        .limit(240)
        .all()
    )
    learning_events = (
        db.query(models.LearningEvent)
        .filter(models.LearningEvent.user_id == user_id, models.LearningEvent.course_id == course_id)
        .order_by(models.LearningEvent.created_at.desc())
        .limit(300)
        .all()
    )
    submission_rows = (
        db.query(
            models.Submission.content,
            models.Submission.code,
            models.Submission.feedback,
            models.Submission.score,
            models.Submission.created_at,
            models.Assignment.title.label("assignment_title"),
            models.Assignment.keywords.label("assignment_keywords"),
        )
        .join(models.Assignment, models.Assignment.id == models.Submission.assignment_id)
        .filter(models.Submission.user_id == user_id, models.Assignment.course_id == course_id)
        .order_by(models.Submission.created_at.desc())
        .limit(200)
        .all()
    )
    points = db.query(models.KnowledgePoint).filter(models.KnowledgePoint.course_id == course_id).all()
    relations = db.query(models.KnowledgeRelation).filter(models.KnowledgeRelation.course_id == course_id).all()
    materials = (
        db.query(models.Material)
        .filter(models.Material.course_id == course_id)
        .order_by(models.Material.uploaded_at.desc())
        .all()
    )

    material_map = {
        item.id: {"id": item.id, "title": item.title, "parse_status": item.parse_status}
        for item in materials
    }

    qa_recent = sum(1 for row in qa_records if row.created_at and row.created_at >= recent_threshold)
    submissions_recent = sum(1 for row in submission_rows if row.created_at and row.created_at >= recent_threshold)
    events_recent = sum(1 for row in learning_events if row.created_at and row.created_at >= recent_threshold)
    activity_level = _compute_activity_level(
        qa_recent=qa_recent,
        submissions_recent=submissions_recent,
        events_recent=events_recent,
    )

    qa_corpus = [row.question for row in qa_records if row.question]
    event_corpus = [row.content for row in learning_events if row.content]
    submission_corpus = [
        "\n".join(
            [
                row.assignment_title or "",
                row.assignment_keywords or "",
                row.content or "",
                row.code or "",
                row.feedback or "",
            ]
        )
        for row in submission_rows
    ]

    if not points:
        merged = "\n".join(qa_corpus + event_corpus + submission_corpus)
        return _build_fallback_report(
            corpus_text=merged,
            bucket_limit=bucket_limit,
            generated_at=now,
            window_days=window_days,
        )

    relation_degree: dict[int, int] = defaultdict(int)
    relation_materials_by_point: dict[int, set[int]] = defaultdict(set)
    for rel in relations:
        relation_degree[rel.source_id] += 1
        relation_degree[rel.target_id] += 1
        if rel.material_id:
            relation_materials_by_point[rel.source_id].add(rel.material_id)
            relation_materials_by_point[rel.target_id].add(rel.material_id)

    point_by_norm = {_normalize_text(point.name): point for point in points if point.name}
    approved_candidates = (
        db.query(models.KnowledgeCandidate)
        .filter(
            models.KnowledgeCandidate.course_id == course_id,
            models.KnowledgeCandidate.status == "approved",
        )
        .all()
    )
    candidate_materials_by_point: dict[int, set[int]] = defaultdict(set)
    for candidate in approved_candidates:
        norm = candidate.term_norm or _normalize_text(candidate.term or "")
        point = point_by_norm.get(norm)
        if not point:
            continue
        candidate_materials_by_point[point.id].add(candidate.material_id)

    profiles: list[dict] = []
    for point in points:
        point_name = (point.name or "").strip()
        point_norm = _normalize_text(point_name)
        if not point_name or not point_norm:
            continue

        qa_hits = 0
        qa_recent_hits = 0
        for row in qa_records:
            question = row.question or ""
            if not _term_hit(point_norm, question):
                continue
            qa_hits += 1
            if row.created_at and row.created_at >= recent_threshold:
                qa_recent_hits += 1

        event_hits = sum(1 for item in event_corpus if _term_hit(point_norm, item))
        assignment_hits = 0
        low_score_hits = 0
        high_score_hits = 0
        recent_low_score_hits = 0
        for row in submission_rows:
            joined_text = "\n".join(
                [
                    row.assignment_title or "",
                    row.assignment_keywords or "",
                    row.content or "",
                    row.code or "",
                    row.feedback or "",
                ]
            )
            if not _term_hit(point_norm, joined_text):
                continue
            assignment_hits += 1
            score = _safe_score(row.score)
            if score is None:
                continue
            if score < 80:
                low_score_hits += 1
                if row.created_at and row.created_at >= recent_threshold:
                    recent_low_score_hits += 1
            if score >= 85:
                high_score_hits += 1

        repeat_question_hits = max(0, qa_recent_hits - 1)
        kg_degree = relation_degree.get(point.id, 0)
        linked_material_ids = set()
        linked_material_ids.update(candidate_materials_by_point.get(point.id, set()))
        linked_material_ids.update(relation_materials_by_point.get(point.id, set()))
        material_coverage = len(linked_material_ids)

        weakness_score = (
            low_score_hits * 2.4
            + recent_low_score_hits * 1.2
            + repeat_question_hits * 1.5
            + qa_recent_hits * 0.8
            + max(0, assignment_hits - high_score_hits) * 0.6
        )
        mastery_score = (
            high_score_hits * 2.1
            + assignment_hits * 0.8
            + max(0, qa_hits - qa_recent_hits) * 0.4
            + material_coverage * 0.3
        )
        expansion_score = mastery_score * 0.55 + kg_degree * 0.75 + material_coverage * 0.55
        confidence_base = qa_hits + assignment_hits + event_hits + material_coverage + (1 if kg_degree else 0)
        confidence = min(1.0, confidence_base / 8.0)

        evidence: list[dict] = []
        if low_score_hits:
            evidence.append(
                {
                    "source": "assignment",
                    "value": f"作业中相关低分提交 {low_score_hits} 次",
                    "weight": round(low_score_hits * 2.4, 4),
                }
            )
        if high_score_hits:
            evidence.append(
                {
                    "source": "assignment",
                    "value": f"作业中相关高分提交 {high_score_hits} 次",
                    "weight": round(high_score_hits * 2.1, 4),
                }
            )
        if qa_recent_hits:
            evidence.append(
                {
                    "source": "qa",
                    "value": f"近{window_days}天相关问答 {qa_recent_hits} 次",
                    "weight": round(qa_recent_hits * 0.8, 4),
                }
            )
        if repeat_question_hits:
            evidence.append(
                {
                    "source": "qa",
                    "value": f"存在重复追问（{repeat_question_hits} 次）",
                    "weight": round(repeat_question_hits * 1.5, 4),
                }
            )
        if kg_degree:
            evidence.append(
                {
                    "source": "kg",
                    "value": f"图谱关联度 {kg_degree}",
                    "weight": round(kg_degree * 0.75, 4),
                }
            )
        if material_coverage:
            evidence.append(
                {
                    "source": "material",
                    "value": f"可关联课程资料 {material_coverage} 份",
                    "weight": round(material_coverage * 0.55, 4),
                }
            )
        if event_hits:
            evidence.append(
                {
                    "source": "event",
                    "value": f"学习行为命中 {event_hits} 次",
                    "weight": round(event_hits * 0.3, 4),
                }
            )
        if not evidence:
            evidence.append(
                {
                    "source": "profile",
                    "value": "当前行为证据较少，建议增加练习与问答记录",
                    "weight": 0.2,
                }
            )

        profiles.append(
            {
                "point_name": point_name,
                "weakness_score": weakness_score,
                "mastery_score": mastery_score,
                "expansion_score": expansion_score,
                "confidence": confidence,
                "evidence": evidence,
                "material_ids": linked_material_ids,
            }
        )

    must_candidates = sorted(
        [
            item
            for item in profiles
            if item["weakness_score"] >= max(1.8, item["mastery_score"] + 0.8)
        ],
        key=lambda row: (row["weakness_score"], row["confidence"]),
        reverse=True,
    )
    explore_candidates = sorted(
        [
            item
            for item in profiles
            if item["mastery_score"] >= 1.6
            and item["expansion_score"] >= item["weakness_score"] + 0.2
        ],
        key=lambda row: (row["expansion_score"], row["mastery_score"], row["confidence"]),
        reverse=True,
    )
    consolidate_candidates = sorted(
        [
            item
            for item in profiles
            if item not in must_candidates and item not in explore_candidates
        ],
        key=lambda row: (
            -abs(row["weakness_score"] - row["mastery_score"]),
            row["confidence"],
            row["expansion_score"],
        ),
        reverse=True,
    )

    must_review = [
        _build_report_item(
            knowledge_point=item["point_name"],
            priority="high",
            weakness_score=item["weakness_score"],
            mastery_score=item["mastery_score"],
            expansion_score=item["expansion_score"],
            confidence=item["confidence"],
            evidence=item["evidence"],
            material_ids=item["material_ids"],
            material_map=material_map,
        )
        for item in must_candidates[:bucket_limit]
    ]
    need_explore = [
        _build_report_item(
            knowledge_point=item["point_name"],
            priority="medium",
            weakness_score=item["weakness_score"],
            mastery_score=item["mastery_score"],
            expansion_score=item["expansion_score"],
            confidence=item["confidence"],
            evidence=item["evidence"],
            material_ids=item["material_ids"],
            material_map=material_map,
        )
        for item in explore_candidates[:bucket_limit]
    ]
    need_consolidate = [
        _build_report_item(
            knowledge_point=item["point_name"],
            priority="medium",
            weakness_score=item["weakness_score"],
            mastery_score=item["mastery_score"],
            expansion_score=item["expansion_score"],
            confidence=item["confidence"],
            evidence=item["evidence"],
            material_ids=item["material_ids"],
            material_map=material_map,
        )
        for item in consolidate_candidates[:bucket_limit]
    ]

    if not must_review and profiles:
        fallback_item = max(profiles, key=lambda row: row["weakness_score"])
        must_review.append(
            _build_report_item(
                knowledge_point=fallback_item["point_name"],
                priority="high",
                weakness_score=fallback_item["weakness_score"],
                mastery_score=fallback_item["mastery_score"],
                expansion_score=fallback_item["expansion_score"],
                confidence=fallback_item["confidence"],
                evidence=fallback_item["evidence"],
                material_ids=fallback_item["material_ids"],
                material_map=material_map,
            )
        )

    if not need_explore and profiles:
        fallback_item = max(profiles, key=lambda row: row["expansion_score"])
        if fallback_item["point_name"] not in {item["knowledge_point"] for item in must_review}:
            need_explore.append(
                _build_report_item(
                    knowledge_point=fallback_item["point_name"],
                    priority="medium",
                    weakness_score=fallback_item["weakness_score"],
                    mastery_score=fallback_item["mastery_score"],
                    expansion_score=fallback_item["expansion_score"],
                    confidence=fallback_item["confidence"],
                    evidence=fallback_item["evidence"],
                    material_ids=fallback_item["material_ids"],
                    material_map=material_map,
                )
            )

    avg_weakness = sum(item["weakness_score"] for item in profiles) / len(profiles) if profiles else 0.0
    risk_level = _compute_risk_level(len(must_review), avg_weakness)
    main_weak_points = [item["knowledge_point"] for item in must_review[:2]]
    note = None
    if all(item["confidence"] < 0.35 for item in must_review + need_explore + need_consolidate):
        note = "当前行为数据较少，建议先完成更多作业并进行课程问答。"

    return {
        "summary": {
            "risk_level": risk_level,
            "activity_level": activity_level,
            "main_weak_points": main_weak_points,
            "generated_at": now,
            "window_days": window_days,
            "note": note,
        },
        "must_review": must_review[:bucket_limit],
        "need_consolidate": need_consolidate[:bucket_limit],
        "need_explore": need_explore[:bucket_limit],
    }


def recommend_for_user(db: Session, user_id: int, course_id: int, limit: int = 5) -> list[dict]:
    report = build_learning_report(
        db,
        user_id,
        course_id,
        bucket_limit=max(DEFAULT_BUCKET_LIMIT, min(limit, 5)),
        window_days=DEFAULT_WINDOW_DAYS,
    )

    must_review = report.get("must_review", [])
    need_explore = report.get("need_explore", [])
    need_consolidate = report.get("need_consolidate", [])

    ranked_blocks = []
    max_pair_count = max(len(must_review), len(need_explore))
    for idx in range(max_pair_count):
        if idx < len(must_review):
            ranked_blocks.append(("must_review", must_review[idx]))
        if idx < len(need_explore):
            ranked_blocks.append(("need_explore", need_explore[idx]))
    ranked_blocks.extend([("need_consolidate", item) for item in need_consolidate])

    items: list[dict] = []
    for source, row in ranked_blocks:
        if len(items) >= max(1, limit):
            break
        if source == "must_review":
            score = row["weakness_score"] + row["confidence"] * 0.8
        elif source == "need_explore":
            score = row["expansion_score"] + row["confidence"] * 0.6
        else:
            score = (
                abs(row["weakness_score"] - row["mastery_score"]) * -0.4
                + row["mastery_score"]
                + row["confidence"] * 0.5
            )
        items.append(
            {
                "knowledge_point": row["knowledge_point"],
                "reason": row["reason"],
                "score": round(max(0.0, score), 4),
            }
        )

    if items:
        return items[: max(1, limit)]

    return [
        {
            "knowledge_point": point,
            "reason": "推荐依据：课程通用学习路径兜底",
            "score": float(max(1, limit - idx)),
        }
        for idx, point in enumerate(_GENERIC_FALLBACK_POINTS[: max(1, limit)])
    ]
