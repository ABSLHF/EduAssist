from datetime import datetime, timedelta
import re
from collections import Counter
from sqlalchemy.orm import Session

from app.models import models


_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,8}|[A-Za-z]{3,20}")
_STOPWORDS = {
    "什么",
    "这个",
    "那个",
    "一个",
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
}


def _extract_fallback_topics(text: str, limit: int = 5) -> list[str]:
    counter: Counter[str] = Counter()
    for token in _TOKEN_RE.findall(text):
        t = token.strip().lower()
        if len(t) < 2 or t in _STOPWORDS or t.isdigit():
            continue
        counter[t] += 1
    return [k for k, _ in counter.most_common(limit)]


def _fallback_recommendations(qa_records: list[models.QARecord], learning_events: list[models.LearningEvent], limit: int) -> list[dict]:
    corpus: list[str] = []
    corpus.extend([q.question for q in qa_records if q.question])
    corpus.extend([e.content for e in learning_events if e.content])
    joined = "\n".join(corpus)
    topics = _extract_fallback_topics(joined, limit=limit)
    if not topics:
        return []

    ranked: list[dict] = []
    for topic in topics:
        score = float(joined.lower().count(topic.lower()))
        ranked.append(
            {
                "knowledge_point": topic,
                "reason": "课程尚未构建知识图谱，基于最近问答与学习记录的高频主题推荐",
                "score": score,
            }
        )
    return ranked[:limit]


def recommend_for_user(db: Session, user_id: int, course_id: int, limit: int = 5) -> list[dict]:
    qa_records = (
        db.query(models.QARecord)
        .filter_by(user_id=user_id, course_id=course_id)
        .order_by(models.QARecord.created_at.desc())
        .limit(80)
        .all()
    )
    learning_events = (
        db.query(models.LearningEvent)
        .filter_by(user_id=user_id, course_id=course_id)
        .order_by(models.LearningEvent.created_at.desc())
        .limit(200)
        .all()
    )
    text = "\n".join([q.question for q in qa_records])
    recent_threshold = datetime.utcnow() - timedelta(days=7)

    points = db.query(models.KnowledgePoint).filter_by(course_id=course_id).all()
    if not points:
        return _fallback_recommendations(qa_records, learning_events, limit=limit)

    ranked: list[dict] = []
    for point in points:
        qa_hits = text.count(point.name) if text else 0
        recent_hits = 0
        for event in learning_events:
            if event.content and point.name in event.content and event.created_at >= recent_threshold:
                recent_hits += 1
        score = float(qa_hits * 1.5 + recent_hits * 1.0)

        reason_parts = []
        if qa_hits:
            reason_parts.append(f"问答中出现 {qa_hits} 次")
        if recent_hits:
            reason_parts.append(f"近7天学习事件 {recent_hits} 次")
        reason = "，".join(reason_parts) if reason_parts else "建议系统性复习该知识点"

        ranked.append(
            {
                "knowledge_point": point.name,
                "reason": reason,
                "score": score,
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:limit]
