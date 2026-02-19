from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.api.deps import get_current_user, require_teacher
from app.schemas.schemas import CourseCreate, CourseOut
from app.models import models
from app.db import get_db

router = APIRouter()


@router.post("/", response_model=CourseOut, summary="创建课程")
def create_course(payload: CourseCreate, db: Session = Depends(get_db), teacher=Depends(require_teacher)):
    course = models.Course(name=payload.name, description=payload.description, teacher_id=teacher.id)
    db.add(course)
    db.commit()
    db.refresh(course)
    return course


@router.get("/", response_model=list[CourseOut], summary="课程列表")
def list_courses(db: Session = Depends(get_db), user=Depends(get_current_user)):
    return db.query(models.Course).all()


@router.post("/{course_id}/join", summary="加入课程")
def join_course(course_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    course = db.query(models.Course).get(course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    exists = db.query(models.UserCourse).filter_by(user_id=user.id, course_id=course_id).first()
    if exists:
        return {"message": "already joined"}
    link = models.UserCourse(user_id=user.id, course_id=course_id)
    db.add(link)
    db.commit()
    return {"message": "joined"}
