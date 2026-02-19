from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import require_teacher
from app.db import get_db
from app.models import models
from app.rag.pipeline import EmbeddingModelUnavailable, upsert_material
from app.schemas.schemas import MaterialUploadOut
from app.services.materials import extract_text, save_upload

router = APIRouter()


@router.post("/", response_model=MaterialUploadOut, summary="上传课程资料")
def upload_material(
    course_id: int,
    title: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    teacher=Depends(require_teacher),
):
    if db.query(models.Course).get(course_id) is None:
        raise HTTPException(status_code=404, detail="Course not found")

    path = save_upload(course_id, file, base_dir="storage")
    material = models.Material(
        course_id=course_id,
        title=title,
        file_path=path,
        file_type=file.content_type or "unknown",
        parse_status="pending",
    )
    db.add(material)
    db.commit()
    db.refresh(material)

    try:
        text = extract_text(path)
        extracted_chars = len(text.strip())
        if extracted_chars == 0:
            material.parse_status = "failed"
            material.parse_error = "No valid text extracted from file"
            material.extracted_chars = 0
            material.parsed_at = datetime.utcnow()
            db.commit()
            raise HTTPException(status_code=400, detail=material.parse_error)

        inserted = upsert_material(course_id, material.id, text)
        material.parse_status = "success"
        material.parse_error = None
        material.extracted_chars = extracted_chars
        material.parsed_at = datetime.utcnow()
        db.commit()
    except EmbeddingModelUnavailable as exc:
        material.parse_status = "failed"
        material.parse_error = f"Embedding model unavailable: {exc}"
        material.extracted_chars = 0
        material.parsed_at = datetime.utcnow()
        db.commit()
        raise HTTPException(status_code=503, detail=material.parse_error)
    except HTTPException:
        raise
    except Exception as exc:
        material.parse_status = "failed"
        material.parse_error = str(exc)
        material.extracted_chars = 0
        material.parsed_at = datetime.utcnow()
        db.commit()
        raise HTTPException(status_code=500, detail=f"Material parsing failed: {exc}")

    return {
        "material_id": material.id,
        "chunks": inserted,
        "parse_status": material.parse_status,
        "extracted_chars": material.extracted_chars,
        "parse_error": material.parse_error,
    }
