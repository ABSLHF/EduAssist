from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import models
from app.schemas.schemas import UserCreate, UserLogin, UserOut
from app.services.auth import (
    create_access_token,
    decode_token,
    get_password_hash,
    verify_password,
)

router = APIRouter()


@router.post('/register', response_model=UserOut, summary='Register')
def register(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.username == payload.username).first():
        raise HTTPException(status_code=400, detail='Username already exists')
    user = models.User(
        username=payload.username,
        password_hash=get_password_hash(payload.password),
        real_name=payload.real_name,
        role=payload.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post('/login', summary='Login')
def login(payload: UserLogin, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == payload.username).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid credentials')
    token = create_access_token({'sub': str(user.id), 'role': user.role})
    return {'access_token': token, 'accessToken': token, 'token_type': 'bearer'}


@router.post('/token', summary='Get access token')
def token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid credentials')
    token_value = create_access_token({'sub': str(user.id), 'role': user.role})
    return {'access_token': token_value, 'accessToken': token_value, 'token_type': 'bearer'}


@router.post('/refresh', summary='Refresh token')
def refresh(authorization: Optional[str] = Header(default=None), db: Session = Depends(get_db)):
    token_value = None
    if authorization and authorization.startswith('Bearer '):
        token_value = authorization.split(' ', 1)[1]
    payload = decode_token(token_value) if token_value else None
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid token')
    user = db.query(models.User).filter(models.User.id == int(payload.get('sub', 0))).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='User not found')
    new_token = create_access_token({'sub': str(user.id), 'role': user.role})
    return {'data': new_token, 'accessToken': new_token, 'status': 0}


@router.post('/logout', summary='Logout')
def logout():
    return {'ok': True}


@router.get('/codes', summary='Get access codes')
def access_codes():
    return []
