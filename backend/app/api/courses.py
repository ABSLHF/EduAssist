from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_teacher
from app.db import get_db
from app.models import models
from app.schemas.schemas import CourseCreate, CourseDashboardOut, CourseOut

router = APIRouter()


def _build_course_out(db: Session, course: models.Course, user_id: int) -> dict:
    joined = (
        db.query(models.UserCourse.id)
        .filter(
            models.UserCourse.user_id == user_id,
            models.UserCourse.course_id == course.id,
        )
        .first()
        is not None
    )
    if course.teacher_id == user_id:
        joined = True

    student_count = (
        db.query(func.count(models.UserCourse.id))
        .filter(models.UserCourse.course_id == course.id)
        .scalar()
        or 0
    )
    material_count = (
        db.query(func.count(models.Material.id))
        .filter(models.Material.course_id == course.id)
        .scalar()
        or 0
    )

    return {
        'id': course.id,
        'name': course.name,
        'description': course.description,
        'teacher_id': course.teacher_id,
        'created_at': course.created_at,
        'joined': joined,
        'student_count': int(student_count),
        'material_count': int(material_count),
    }


@router.post('/', response_model=CourseOut, summary='Create course')
def create_course(
    payload: CourseCreate,
    db: Session = Depends(get_db),
    teacher=Depends(require_teacher),
):
    course = models.Course(
        name=payload.name,
        description=payload.description,
        teacher_id=teacher.id,
    )
    db.add(course)
    db.commit()
    db.refresh(course)
    return _build_course_out(db, course, teacher.id)


@router.get('/', response_model=list[CourseOut], summary='List courses')
def list_courses(db: Session = Depends(get_db), user=Depends(get_current_user)):
    courses = db.query(models.Course).order_by(models.Course.created_at.desc()).all()
    return [_build_course_out(db, course, user.id) for course in courses]


@router.get('/{course_id}', response_model=CourseOut, summary='Course detail')
def get_course_detail(course_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    course = db.get(models.Course, course_id)
    if not course:
        raise HTTPException(status_code=404, detail='Course not found')
    return _build_course_out(db, course, user.id)


@router.post('/{course_id}/join', summary='Join course')
def join_course(course_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    course = db.get(models.Course, course_id)
    if not course:
        raise HTTPException(status_code=404, detail='Course not found')
    exists = db.query(models.UserCourse).filter_by(user_id=user.id, course_id=course_id).first()
    if exists:
        return {'message': 'already joined'}
    link = models.UserCourse(user_id=user.id, course_id=course_id)
    db.add(link)
    db.commit()
    return {'message': 'joined'}


@router.get('/{course_id}/dashboard', response_model=CourseDashboardOut, summary='Course dashboard')
def get_course_dashboard(course_id: int, db: Session = Depends(get_db), teacher=Depends(require_teacher)):
    course = db.get(models.Course, course_id)
    if not course:
        raise HTTPException(status_code=404, detail='Course not found')
    if course.teacher_id != teacher.id:
        raise HTTPException(status_code=403, detail='Permission denied')

    student_count = (
        db.query(func.count(models.UserCourse.id))
        .filter(models.UserCourse.course_id == course_id)
        .scalar()
        or 0
    )
    material_count = (
        db.query(func.count(models.Material.id))
        .filter(models.Material.course_id == course_id)
        .scalar()
        or 0
    )
    assignment_count = (
        db.query(func.count(models.Assignment.id))
        .filter(models.Assignment.course_id == course_id)
        .scalar()
        or 0
    )

    avg_score = (
        db.query(func.avg(models.Submission.score))
        .join(models.Assignment, models.Assignment.id == models.Submission.assignment_id)
        .filter(
            models.Assignment.course_id == course_id,
            models.Submission.score.isnot(None),
        )
        .scalar()
    )

    submission_count = (
        db.query(func.count(models.Submission.id))
        .join(models.Assignment, models.Assignment.id == models.Submission.assignment_id)
        .filter(models.Assignment.course_id == course_id)
        .scalar()
        or 0
    )
    expected_total = int(student_count) * int(assignment_count)
    avg_submission_rate = (submission_count / expected_total * 100) if expected_total > 0 else 0.0

    recent_material_rows = (
        db.query(models.Material)
        .filter(models.Material.course_id == course_id)
        .order_by(models.Material.uploaded_at.desc())
        .limit(8)
        .all()
    )
    recent_materials = [
        {
            'id': row.id,
            'title': row.title,
            'parse_status': row.parse_status,
            'extracted_chars': row.extracted_chars,
            'uploaded_at': row.uploaded_at,
        }
        for row in recent_material_rows
    ]

    latest_submission_rows = (
        db.query(
            models.Submission.id.label('submission_id'),
            models.Submission.assignment_id,
            models.Submission.user_id,
            models.Submission.score,
            models.Submission.feedback,
            models.Submission.created_at,
            models.Assignment.title.label('assignment_title'),
            models.User.real_name.label('real_name'),
            models.User.username.label('username'),
        )
        .join(models.Assignment, models.Assignment.id == models.Submission.assignment_id)
        .join(models.User, models.User.id == models.Submission.user_id)
        .filter(models.Assignment.course_id == course_id)
        .order_by(models.Submission.created_at.desc())
        .limit(8)
        .all()
    )
    latest_submissions = [
        {
            'submission_id': row.submission_id,
            'assignment_id': row.assignment_id,
            'assignment_title': row.assignment_title,
            'user_id': row.user_id,
            'user_name': row.real_name or row.username,
            'score': row.score,
            'feedback_preview': (row.feedback or '')[:120] or None,
            'created_at': row.created_at,
        }
        for row in latest_submission_rows
    ]

    return {
        'course_id': course_id,
        'student_count': int(student_count),
        'material_count': int(material_count),
        'assignment_count': int(assignment_count),
        'average_score': round(float(avg_score), 2) if avg_score is not None else None,
        'average_submission_rate': round(float(avg_submission_rate), 2),
        'recent_materials': recent_materials,
        'latest_submissions': latest_submissions,
    }
