from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.assignments import submit_assignment
from app.api.deps import get_current_user
from app.db import get_db
from app.models import models
from app.schemas.schemas import SubmissionCreate, SubmissionFeedbackOut, SubmissionOut

router = APIRouter()


@router.post("/", response_model=SubmissionOut, summary="提交作业（通用入口）")
def submit_assignment_v2(
    assignment_id: int,
    payload: SubmissionCreate,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    return submit_assignment(assignment_id=assignment_id, payload=payload, db=db, user=user)


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
        "score": submission.score,
        "created_at": submission.created_at,
    }
