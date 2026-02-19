from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.api.deps import get_current_user, require_teacher
from app.db import get_db
from app.models import models
from app.schemas.schemas import KGCandidatesResponse, KGResponse
from app.services.keywords import extract_keywords
from app.services.materials import extract_text

router = APIRouter()


@router.get("/{course_id}", response_model=KGResponse, summary="获取知识图谱")
def get_kg(course_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    course = db.query(models.Course).get(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    nodes = db.query(models.KnowledgePoint).filter_by(course_id=course_id).all()
    edges = db.query(models.KnowledgeRelation).filter_by(course_id=course_id).all()
    if not edges:
        # Compatibility fallback for legacy table name knowledge_edges.
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
                    {"source": row.source_id, "target": row.target_id, "relation": row.relation}
                    for row in legacy_rows
                ],
            }
    return {
        "nodes": [{"id": n.id, "label": n.name, "description": n.description} for n in nodes],
        "edges": [{"source": e.source_id, "target": e.target_id, "relation": e.relation} for e in edges],
    }


@router.post("/{course_id}/points", summary="新增知识点")
def create_point(course_id: int, name: str, db: Session = Depends(get_db), teacher=Depends(require_teacher)):
    if not db.query(models.Course).get(course_id):
        raise HTTPException(status_code=404, detail="Course not found")
    exists = db.query(models.KnowledgePoint).filter_by(course_id=course_id, name=name).first()
    if exists:
        return {"id": exists.id, "name": exists.name}
    kp = models.KnowledgePoint(course_id=course_id, name=name)
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
    if not db.query(models.Course).get(course_id):
        raise HTTPException(status_code=404, detail="Course not found")
    edge = models.KnowledgeRelation(course_id=course_id, source_id=source_id, target_id=target_id, relation=relation)
    db.add(edge)
    db.commit()
    db.refresh(edge)
    return {"id": edge.id}


@router.post("/{course_id}/extract", summary="自动抽取知识点")
def extract_points(course_id: int, material_id: int, top_k: int = 10, db: Session = Depends(get_db), teacher=Depends(require_teacher)):
    material = db.query(models.Material).get(material_id)
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    text = extract_text(material.file_path)
    keywords = extract_keywords(text, top_k=top_k)
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
    db: Session = Depends(get_db),
    teacher=Depends(require_teacher),
):
    material = db.query(models.Material).get(material_id)
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    text = extract_text(material.file_path)
    candidates = extract_keywords(text, top_k=top_k)

    created: list[str] = []
    if auto_create:
        for kw in candidates:
            if not db.query(models.KnowledgePoint).filter_by(course_id=course_id, name=kw).first():
                db.add(models.KnowledgePoint(course_id=course_id, name=kw))
                created.append(kw)
        db.commit()
    return {"candidates": candidates, "created": created}
