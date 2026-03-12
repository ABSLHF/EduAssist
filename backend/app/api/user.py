from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import models
from app.services.auth import decode_token

router = APIRouter()


def _extract_token(authorization: Optional[str]) -> Optional[str]:
    if authorization and authorization.startswith('Bearer '):
        return authorization.split(' ', 1)[1]
    return None


@router.get('/info', summary='User info')
def user_info(authorization: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    token_value = _extract_token(authorization)
    payload = decode_token(token_value) if token_value else None
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid token')
    user = db.query(models.User).filter(models.User.id == int(payload.get('sub', 0))).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='User not found')
    is_teacher = user.role == 1
    role_name = 'teacher' if is_teacher else 'student'
    home_path = '/eduassist/teacher/courses' if is_teacher else '/eduassist/student/courses'
    return {
        'userId': str(user.id),
        'username': user.username,
        'realName': user.real_name or user.username,
        'avatar': '',
        'roles': [role_name],
        'desc': role_name,
        'homePath': home_path,
        'token': token_value or '',
    }
