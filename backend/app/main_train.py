from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import auth, model
from app.config import settings

app = FastAPI(title="EduAssist Train API")

cors_origins = [
    item.strip()
    for item in (settings.cors_allow_origins or "").split(",")
    if item.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=settings.cors_allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["Auth"])
app.include_router(model.router, prefix="/model", tags=["Model"])


@app.get("/")
def read_root():
    return {"status": "ok", "mode": "train"}
