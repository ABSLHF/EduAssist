from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_teacher
from app.db import get_db
from app.llm.client import call_llm
from app.models import models
from app.schemas.schemas import (
    AssignmentSubmissionOut,
    AssignmentCreate,
    AssignmentOut,
    SubmissionCreate,
    SubmissionOut,
)
from app.services.assignment_feedback import generate_text_assignment_feedback

router = APIRouter()


def _is_course_member(db: Session, user_id: int, course_id: int) -> bool:
    exists = (
        db.query(models.UserCourse.id)
        .filter(models.UserCourse.user_id == user_id, models.UserCourse.course_id == course_id)
        .first()
    )
    return exists is not None


@router.post("/", response_model=AssignmentOut, summary="发布作业")
def create_assignment(payload: AssignmentCreate, db: Session = Depends(get_db), teacher=Depends(require_teacher)):
    course = db.query(models.Course).get(payload.course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    if course.teacher_id != teacher.id:
        raise HTTPException(status_code=403, detail="Permission denied")

    keywords = ",".join(payload.keywords) if payload.keywords else None
    assignment = models.Assignment(
        course_id=payload.course_id,
        title=payload.title,
        description=payload.description,
        type=payload.type,
        keywords=keywords,
    )
    db.add(assignment)
    db.commit()
    db.refresh(assignment)
    return assignment


@router.get("/{course_id}", response_model=list[AssignmentOut], summary="作业列表")
def list_assignments(course_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    course = db.query(models.Course).get(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    if user.role != 1 and not _is_course_member(db, user.id, course_id):
        raise HTTPException(status_code=403, detail="Please join the course first")
    if user.role == 1 and course.teacher_id != user.id:
        raise HTTPException(status_code=403, detail="Permission denied")
    return db.query(models.Assignment).filter_by(course_id=course_id).all()


@router.post("/{assignment_id}/submit", response_model=SubmissionOut, summary="提交作业")
async def submit_assignment(
    assignment_id: int,
    payload: SubmissionCreate,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    assignment = db.query(models.Assignment).get(assignment_id)
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")

    course = db.query(models.Course).get(assignment.course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    if user.role != 1 and not _is_course_member(db, user.id, assignment.course_id):
        raise HTTPException(status_code=403, detail="Please join the course first")
    if user.role == 1 and course.teacher_id != user.id:
        raise HTTPException(status_code=403, detail="Permission denied")

    if assignment.type != "text":
        raise HTTPException(status_code=400, detail="当前版本仅支持文本作业智能批改")

    content = (payload.content or "").strip()
    if not content:
        raise HTTPException(status_code=422, detail="文本作业需要提交 content")

    feedback = await generate_text_assignment_feedback(
        assignment=assignment,
        content=content,
        llm_call=call_llm,
        db=db,
    )

    submission = models.Submission(
        assignment_id=assignment_id,
        user_id=user.id,
        content=content,
        code=None,
        feedback=feedback,
        score=None,
    )
    db.add(submission)
    db.flush()
    db.add(
        models.LearningEvent(
            user_id=user.id,
            course_id=assignment.course_id,
            event_type="assignment",
            content=assignment.title,
        )
    )
    db.commit()
    return submission


@router.get("/{assignment_id}/submissions", response_model=list[AssignmentSubmissionOut], summary="查看提交")
def list_submissions(assignment_id: int, db: Session = Depends(get_db), teacher=Depends(require_teacher)):
    assignment = db.query(models.Assignment).get(assignment_id)
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")

    course = db.query(models.Course).get(assignment.course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    if course.teacher_id != teacher.id:
        raise HTTPException(status_code=403, detail="Permission denied")

    latest_ids_subquery = (
        db.query(func.max(models.Submission.id).label("latest_id"))
        .filter(models.Submission.assignment_id == assignment_id)
        .group_by(models.Submission.user_id)
        .subquery()
    )

    rows = (
        db.query(
            models.Submission.id,
            models.Submission.assignment_id,
            models.Submission.user_id,
            models.Submission.content,
            models.Submission.code,
            models.Submission.feedback,
            models.Submission.score,
            models.Submission.created_at,
            models.SubmissionFeedback.feedback.label("teacher_feedback"),
            models.SubmissionFeedback.score.label("teacher_score"),
            models.User.real_name,
            models.User.username,
        )
        .join(latest_ids_subquery, models.Submission.id == latest_ids_subquery.c.latest_id)
        .join(models.User, models.User.id == models.Submission.user_id)
        .outerjoin(
            models.SubmissionFeedback,
            models.SubmissionFeedback.submission_id == models.Submission.id,
        )
        .order_by(models.Submission.created_at.desc(), models.Submission.id.desc())
        .all()
    )

    return [
        {
            "id": row.id,
            "assignment_id": row.assignment_id,
            "assignment_title": assignment.title,
            "user_id": row.user_id,
            "user_name": row.real_name or row.username,
            "content": row.content,
            "code": row.code,
            "feedback": row.teacher_feedback,
            "score": row.teacher_score,
            "ai_feedback": row.feedback,
            "ai_score": row.score,
            "teacher_feedback": row.teacher_feedback,
            "teacher_score": row.teacher_score,
            "feedback_preview": (row.teacher_feedback or "")[:120] or None,
            "created_at": row.created_at,
        }
        for row in rows
    ]
