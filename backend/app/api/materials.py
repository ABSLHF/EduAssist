from datetime import datetime
from mimetypes import guess_type
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_teacher
from app.db import get_db
from app.models import models
from app.rag.pipeline import EmbeddingModelUnavailable, upsert_material
from app.schemas.schemas import MaterialOut, MaterialUploadOut
from app.services.materials import extract_text, read_file_bytes, resolve_material_filename, save_upload

router = APIRouter()


def _can_access_course_material(db: Session, user, course_id: int) -> bool:
    course = db.get(models.Course, course_id)
    if not course:
        return False
    if course.teacher_id == user.id:
        return True
    enrollment = (
        db.query(models.UserCourse.id)
        .filter(models.UserCourse.user_id == user.id, models.UserCourse.course_id == course_id)
        .first()
    )
    return enrollment is not None


@router.get('/course/{course_id}', response_model=list[MaterialOut], summary='List course materials')
def list_course_materials(
    course_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    course = db.get(models.Course, course_id)
    if not course:
        raise HTTPException(status_code=404, detail='Course not found')

    return (
        db.query(models.Material)
        .filter(models.Material.course_id == course_id)
        .order_by(models.Material.uploaded_at.desc())
        .all()
    )


@router.get('/{material_id}/file', summary='Get material file')
def get_material_file(
    material_id: int,
    disposition: str = Query(default='attachment'),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    if disposition not in {'attachment', 'inline'}:
        raise HTTPException(status_code=422, detail='disposition must be "attachment" or "inline"')

    material = db.get(models.Material, material_id)
    if not material:
        raise HTTPException(status_code=404, detail='Material not found')
    if not _can_access_course_material(db, user, material.course_id):
        raise HTTPException(status_code=403, detail='No permission to access this material')

    try:
        file_bytes = read_file_bytes(material.file_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail='Material file not found')
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Failed to read material file: {exc}')

    filename = resolve_material_filename(material.file_path, fallback_title=material.title)
    media_type = guess_type(filename)[0] or 'application/octet-stream'
    encoded_name = quote(filename)
    headers = {
        'Content-Disposition': f"{disposition}; filename*=UTF-8''{encoded_name}",
    }
    return Response(content=file_bytes, media_type=media_type, headers=headers)


@router.post('/', response_model=MaterialUploadOut, summary='Upload course material')
def upload_material(
    course_id: int | None = Query(default=None),
    course_id_form: int | None = Form(default=None, alias='course_id'),
    title: str | None = Query(default=None),
    title_form: str | None = Form(default=None, alias='title'),
    file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
    teacher=Depends(require_teacher),
):
    resolved_course_id = course_id if course_id is not None else course_id_form
    resolved_title = title if title is not None else title_form

    if resolved_course_id is None:
        raise HTTPException(status_code=422, detail='course_id is required (query or form field)')
    if not resolved_title:
        raise HTTPException(status_code=422, detail='title is required (query or form field)')
    if file is None:
        raise HTTPException(
            status_code=422,
            detail='file is required in multipart/form-data with field name "file"',
        )

    if db.get(models.Course, resolved_course_id) is None:
        raise HTTPException(status_code=404, detail='Course not found')

    path = save_upload(resolved_course_id, file, base_dir='storage')
    material = models.Material(
        course_id=resolved_course_id,
        title=resolved_title,
        file_path=path,
        # DB column is VARCHAR(20); long MIME types like docx/pptx would fail insert.
        file_type=(file.content_type or 'unknown')[:20],
        parse_status='pending',
    )
    db.add(material)
    db.commit()
    db.refresh(material)

    try:
        text = extract_text(path)
        extracted_chars = len(text.strip())
        if extracted_chars == 0:
            material.parse_status = 'failed'
            material.parse_error = 'No valid text extracted from file'
            material.extracted_chars = 0
            material.parsed_at = datetime.utcnow()
            db.commit()
            raise HTTPException(status_code=400, detail=material.parse_error)

        inserted = upsert_material(resolved_course_id, material.id, text)
        from app.api.qa import clear_qa_cache

        clear_qa_cache(resolved_course_id)
        material.parse_status = 'success'
        material.parse_error = None
        material.extracted_chars = extracted_chars
        material.parsed_at = datetime.utcnow()
        db.commit()
    except EmbeddingModelUnavailable as exc:
        material.parse_status = 'failed'
        material.parse_error = f'Embedding model unavailable: {exc}'
        material.extracted_chars = 0
        material.parsed_at = datetime.utcnow()
        db.commit()
        raise HTTPException(status_code=503, detail=material.parse_error)
    except HTTPException:
        raise
    except Exception as exc:
        material.parse_status = 'failed'
        material.parse_error = str(exc)
        material.extracted_chars = 0
        material.parsed_at = datetime.utcnow()
        db.commit()
        raise HTTPException(status_code=500, detail=f'Material parsing failed: {exc}')

    return {
        'material_id': material.id,
        'chunks': inserted,
        'parse_status': material.parse_status,
        'extracted_chars': material.extracted_chars,
        'parse_error': material.parse_error,
    }
