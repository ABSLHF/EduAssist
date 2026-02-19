from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    username: str
    password: str
    real_name: Optional[str] = None
    role: int = 0


class UserLogin(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    real_name: Optional[str]
    role: int
    created_at: datetime

    model_config = {"from_attributes": True}


class CourseCreate(BaseModel):
    name: str
    description: Optional[str] = None


class CourseOut(BaseModel):
    id: int
    name: str
    description: Optional[str]
    teacher_id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class MaterialUploadOut(BaseModel):
    material_id: int
    chunks: int
    parse_status: str
    extracted_chars: int
    parse_error: Optional[str] = None


class QARequest(BaseModel):
    course_id: int
    question: str


class QAResponse(BaseModel):
    answer: str
    source_type: int
    mode: str  # rag_llm | rag_fallback
    references: list[str] = Field(default_factory=list)


class KGNode(BaseModel):
    id: int
    label: str
    description: Optional[str] = None


class KGEdge(BaseModel):
    source: int
    target: int
    relation: str = "relates_to"


class KGResponse(BaseModel):
    nodes: list[KGNode]
    edges: list[KGEdge]


class KGCandidatesResponse(BaseModel):
    candidates: list[str]
    created: list[str]


class AssignmentCreate(BaseModel):
    course_id: int
    title: str
    description: Optional[str] = None
    type: str = "text"
    keywords: Optional[list[str]] = None


class AssignmentOut(BaseModel):
    id: int
    course_id: int
    title: str
    description: Optional[str]
    type: str
    keywords: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class SubmissionCreate(BaseModel):
    content: Optional[str] = None
    code: Optional[str] = None


class SubmissionOut(BaseModel):
    id: int
    assignment_id: int
    user_id: int
    content: Optional[str]
    code: Optional[str]
    feedback: Optional[str]
    score: Optional[int]
    created_at: datetime

    model_config = {"from_attributes": True}


class SubmissionFeedbackOut(BaseModel):
    submission_id: int
    feedback: str
    score: Optional[int]
    created_at: datetime


class RecommendationItem(BaseModel):
    knowledge_point: str
    reason: str
    score: float


class RecommendationOut(BaseModel):
    items: list[RecommendationItem]


class ModelTrainRequest(BaseModel):
    dataset_name: str = "sample_local"
    task_type: str = "text_classification"
    dataset_config: Optional[str] = None
    model_name: Optional[str] = None
    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 2e-5
    max_samples: int = 1200


class ModelTrainResponse(BaseModel):
    job_id: int
    status: str


class ModelRunOut(BaseModel):
    id: int
    task_type: str
    dataset_name: Optional[str]
    status: str
    metrics: Optional[str]
    model_path: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class QAPredictRequest(BaseModel):
    context: str
    question: str
    model_path: Optional[str] = None


class QAPredictResponse(BaseModel):
    answer: str
    confidence: Optional[float] = None
