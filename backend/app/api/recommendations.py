from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.db import get_db
from app.schemas.schemas import RecommendationOut
from app.services.recommender import recommend_for_user

router = APIRouter()


@router.get("/{course_id}", response_model=RecommendationOut, summary="学习推荐")
def get_recommendations(course_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    items = recommend_for_user(db, user.id, course_id, limit=5)
    return {"items": items}
