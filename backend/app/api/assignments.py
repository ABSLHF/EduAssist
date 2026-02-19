import ast

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_teacher
from app.db import get_db
from app.models import models
from app.schemas.schemas import (
    AssignmentCreate,
    AssignmentOut,
    SubmissionCreate,
    SubmissionOut,
)
from app.services.grading import grade_text_submission

router = APIRouter()


def _safe_code_feedback(code: str) -> tuple[int, str]:
    if not code.strip():
        return 0, "未提交代码内容"
    try:
        ast.parse(code)
        lines = len([line for line in code.splitlines() if line.strip()])
        return 75, f"代码语法检查通过，非空代码行数 {lines}。当前阶段未执行代码，仅静态分析。"
    except SyntaxError as exc:
        return 40, f"代码存在语法错误: {exc.msg} (line {exc.lineno})。"


@router.post("/", response_model=AssignmentOut, summary="发布作业")
def create_assignment(payload: AssignmentCreate, db: Session = Depends(get_db), teacher=Depends(require_teacher)):
    course = db.query(models.Course).get(payload.course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
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
    return db.query(models.Assignment).filter_by(course_id=course_id).all()


@router.post("/{assignment_id}/submit", response_model=SubmissionOut, summary="提交作业")
def submit_assignment(assignment_id: int, payload: SubmissionCreate, db: Session = Depends(get_db), user=Depends(get_current_user)):
    assignment = db.query(models.Assignment).get(assignment_id)
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")

    if assignment.type == "text":
        keywords = assignment.keywords.split(",") if assignment.keywords else []
        score, feedback = grade_text_submission(payload.content or "", keywords)
    else:
        score, feedback = _safe_code_feedback(payload.code or "")

    submission = models.Submission(
        assignment_id=assignment_id,
        user_id=user.id,
        content=payload.content,
        code=payload.code,
        feedback=feedback,
        score=score,
    )
    db.add(submission)
    db.flush()
    db.add(
        models.SubmissionFeedback(
            submission_id=submission.id,
            feedback=feedback,
            score=score,
        )
    )
    db.add(
        models.LearningEvent(
            user_id=user.id,
            course_id=assignment.course_id,
            event_type="assignment",
            content=assignment.title,
        )
    )
    db.commit()
    db.refresh(submission)
    return submission


@router.get("/{assignment_id}/submissions", response_model=list[SubmissionOut], summary="查看提交")
def list_submissions(assignment_id: int, db: Session = Depends(get_db), teacher=Depends(require_teacher)):
    return db.query(models.Submission).filter_by(assignment_id=assignment_id).all()
