from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.assignments import submit_assignment
from app.api.deps import get_current_user, require_teacher
from app.db import get_db
from app.models import models
from app.schemas.schemas import (
    AssignmentSubmissionOut,
    SubmissionCreate,
    SubmissionFeedbackOut,
    SubmissionMineOut,
    SubmissionOut,
    SubmissionReviewUpdate,
)

router = APIRouter()


@router.get("/me", response_model=list[SubmissionMineOut], summary="查看我的作业提交记录")
def list_my_submissions(
    course_id: int | None = None,
    assignment_id: int | None = None,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    query = (
        db.query(
            models.Submission.id,
            models.Submission.assignment_id,
            models.Submission.content,
            models.Submission.code,
            models.Submission.feedback.label("ai_feedback"),
            models.Submission.score.label("ai_score"),
            models.Submission.created_at,
            models.SubmissionFeedback.feedback.label("teacher_feedback"),
            models.SubmissionFeedback.score.label("teacher_score"),
            models.Assignment.title.label("assignment_title"),
            models.Assignment.course_id.label("course_id"),
        )
        .join(models.Assignment, models.Assignment.id == models.Submission.assignment_id)
        .outerjoin(
            models.SubmissionFeedback,
            models.SubmissionFeedback.submission_id == models.Submission.id,
        )
        .filter(models.Submission.user_id == user.id)
    )
    if course_id is not None:
        query = query.filter(models.Assignment.course_id == course_id)
    if assignment_id is not None:
        query = query.filter(models.Submission.assignment_id == assignment_id)

    rows = query.order_by(models.Submission.created_at.asc(), models.Submission.id.asc()).all()

    attempt_counter: dict[int, int] = {}
    normalized: list[dict] = []
    for row in rows:
        current_attempt = attempt_counter.get(row.assignment_id, 0) + 1
        attempt_counter[row.assignment_id] = current_attempt
        teacher_feedback = row.teacher_feedback
        teacher_score = row.teacher_score
        ai_feedback = row.ai_feedback
        ai_score = row.ai_score
        normalized.append(
            {
                "id": row.id,
                "assignment_id": row.assignment_id,
                "assignment_title": row.assignment_title,
                "course_id": row.course_id,
                "content": row.content,
                "code": row.code,
                "ai_feedback": ai_feedback,
                "ai_score": ai_score,
                "teacher_feedback": teacher_feedback,
                "teacher_score": teacher_score,
                "final_feedback": teacher_feedback or ai_feedback,
                # 最终分数只认教师评分；AI 不再提供分数。
                "final_score": teacher_score,
                "attempt_no": current_attempt,
                "latest": False,
                "created_at": row.created_at,
            }
        )

    latest_seen: set[int] = set()
    output: list[dict] = []
    for item in reversed(normalized):
        aid = item["assignment_id"]
        if aid not in latest_seen:
            item["latest"] = True
            latest_seen.add(aid)
        output.append(item)
    return output


@router.post("/", response_model=SubmissionOut, summary="提交作业（通用入口）")
async def submit_assignment_v2(
    assignment_id: int,
    payload: SubmissionCreate,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    return await submit_assignment(assignment_id=assignment_id, payload=payload, db=db, user=user)


@router.get("/{submission_id}/feedback", response_model=SubmissionFeedbackOut, summary="获取作业反馈")
def get_submission_feedback(submission_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    submission = db.query(models.Submission).get(submission_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    if submission.user_id != user.id and user.role != 1:
        raise HTTPException(status_code=403, detail="Permission denied")

    detail = db.query(models.SubmissionFeedback).filter_by(submission_id=submission_id).first()
    if detail:
        return {
            "submission_id": detail.submission_id,
            "feedback": detail.feedback,
            "score": detail.score,
            "created_at": detail.created_at,
        }
    return {
        "submission_id": submission.id,
        "feedback": submission.feedback or "暂无反馈",
        # 未教师批改时不返回分数。
        "score": None,
        "created_at": submission.created_at,
    }


@router.put("/{submission_id}/review", response_model=AssignmentSubmissionOut, summary="教师审核作业提交")
def review_submission(
    submission_id: int,
    payload: SubmissionReviewUpdate,
    db: Session = Depends(get_db),
    teacher=Depends(require_teacher),
):
    submission = db.query(models.Submission).get(submission_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    assignment = db.query(models.Assignment).get(submission.assignment_id)
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")

    course = db.query(models.Course).get(assignment.course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    if course.teacher_id != teacher.id:
        raise HTTPException(status_code=403, detail="Permission denied")

    feedback_detail = (
        db.query(models.SubmissionFeedback)
        .filter(models.SubmissionFeedback.submission_id == submission_id)
        .first()
    )
    if not feedback_detail:
        feedback_detail = models.SubmissionFeedback(
            submission_id=submission_id,
            feedback="",
            score=None,
        )
        db.add(feedback_detail)
        db.flush()

    if payload.feedback is not None:
        feedback_detail.feedback = payload.feedback
    if payload.score is not None:
        feedback_detail.score = payload.score

    db.commit()
    db.refresh(feedback_detail)

    user = db.query(models.User).get(submission.user_id)
    user_name = ""
    if user:
        user_name = user.real_name or user.username

    return {
        "id": submission.id,
        "assignment_id": submission.assignment_id,
        "assignment_title": assignment.title,
        "user_id": submission.user_id,
        "user_name": user_name,
        "content": submission.content,
        "code": submission.code,
        "feedback": feedback_detail.feedback,
        "score": feedback_detail.score,
        "ai_feedback": submission.feedback,
        "ai_score": submission.score,
        "teacher_feedback": feedback_detail.feedback,
        "teacher_score": feedback_detail.score,
        "feedback_preview": (feedback_detail.feedback or "")[:120] or None,
        "created_at": submission.created_at,
    }
