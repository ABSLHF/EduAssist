from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api import (
    assignments,
    auth,
    courses,
    kg,
    materials,
    model,
    qa,
    recommendations,
    submissions,
    user,
)
from app.config import settings

app = FastAPI(title='EduAssist API', docs_url=None, redoc_url=None)

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
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(auth.router, prefix='/auth', tags=['Auth'])
app.include_router(courses.router, prefix='/courses', tags=['Courses'])
app.include_router(materials.router, prefix='/materials', tags=['Materials'])
app.include_router(qa.router, prefix='/qa', tags=['QA'])
app.include_router(kg.router, prefix='/kg', tags=['KnowledgeGraph'])
app.include_router(assignments.router, prefix='/assignments', tags=['Assignments'])
app.include_router(submissions.router, prefix='/submissions', tags=['Submissions'])
app.include_router(recommendations.router, prefix='/recommendations', tags=['Recommendations'])
app.include_router(model.router, prefix='/model', tags=['Model'])
app.include_router(user.router, prefix='/user', tags=['User'])

app.mount('/static', StaticFiles(directory='app/static'), name='static')


@app.get('/docs', include_in_schema=False)
def custom_swagger_ui_html() -> HTMLResponse:
    base = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title='EduAssist API',
    )
    html = base.body.decode('utf-8').replace(
        '</body>',
        "<script src='/static/swagger-i18n.js'></script></body>",
    )
    return HTMLResponse(content=html, status_code=base.status_code)


@app.get('/')
def read_root():
    return {'status': 'ok'}
