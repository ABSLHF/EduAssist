from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db import get_db
from app.models import models
from app.schemas.schemas import LearningReportOut, RecommendationOut
from app.services.recommender import build_learning_report, recommend_for_user

router = APIRouter()


def _check_course_access(course_id: int, db: Session, user):
    course = db.query(models.Course).get(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    if user.role == 1 and course.teacher_id != user.id:
        raise HTTPException(status_code=403, detail="Permission denied")
    if user.role != 1:
        joined = (
            db.query(models.UserCourse.id)
            .filter(models.UserCourse.user_id == user.id, models.UserCourse.course_id == course_id)
            .first()
        )
        if not joined:
            raise HTTPException(status_code=403, detail="Please join the course first")
    return course


@router.get("/{course_id}", response_model=RecommendationOut, summary="学习推荐")
def get_recommendations(course_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _check_course_access(course_id, db, user)
    items = recommend_for_user(db, user.id, course_id, limit=5)
    return {"items": items}


@router.get("/{course_id}/report", response_model=LearningReportOut, summary="学习推荐报告")
def get_recommendations_report(course_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _check_course_access(course_id, db, user)
    return build_learning_report(db, user.id, course_id)
