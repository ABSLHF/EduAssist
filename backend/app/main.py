from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware
from app.api import auth, courses, materials, qa, kg, assignments, recommendations, model, submissions

app = FastAPI(title="EduAssist API", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["认证"])
app.include_router(courses.router, prefix="/courses", tags=["课程"])
app.include_router(materials.router, prefix="/materials", tags=["资料"])
app.include_router(qa.router, prefix="/qa", tags=["问答"])
app.include_router(kg.router, prefix="/kg", tags=["知识图谱"])
app.include_router(assignments.router, prefix="/assignments", tags=["作业"])
app.include_router(submissions.router, prefix="/submissions", tags=["提交"])
app.include_router(recommendations.router, prefix="/recommendations", tags=["推荐"])
app.include_router(model.router, prefix="/model", tags=["训练模型"])

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/docs", include_in_schema=False)
def custom_swagger_ui_html() -> HTMLResponse:
    base = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="EduAssist API",
    )
    html = base.body.decode("utf-8").replace(
        "</body>",
        "<script src='/static/swagger-i18n.js'></script></body>",
    )
    return HTMLResponse(content=html, status_code=base.status_code)

@app.get("/")
def read_root():
    return {"status": "ok"}
