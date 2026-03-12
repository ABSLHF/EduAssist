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
    joined: Optional[bool] = None
    student_count: Optional[int] = None
    material_count: Optional[int] = None

    model_config = {"from_attributes": True}


class DashboardMaterialOut(BaseModel):
    id: int
    title: str
    parse_status: str
    extracted_chars: int
    uploaded_at: datetime


class DashboardSubmissionOut(BaseModel):
    submission_id: int
    assignment_id: int
    assignment_title: str
    user_id: int
    user_name: str
    score: Optional[int] = None
    feedback_preview: Optional[str] = None
    created_at: datetime


class CourseDashboardOut(BaseModel):
    course_id: int
    student_count: int
    material_count: int
    assignment_count: int
    average_score: Optional[float] = None
    average_submission_rate: Optional[float] = None
    recent_materials: list[DashboardMaterialOut] = Field(default_factory=list)
    latest_submissions: list[DashboardSubmissionOut] = Field(default_factory=list)


class MaterialUploadOut(BaseModel):
    material_id: int
    chunks: int
    parse_status: str
    extracted_chars: int
    parse_error: Optional[str] = None


class MaterialOut(BaseModel):
    id: int
    course_id: int
    title: str
    file_type: str
    parse_status: str
    parse_error: Optional[str]
    extracted_chars: int
    parsed_at: Optional[datetime]
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class ChatMessage(BaseModel):
    role: str
    content: str


class QARequest(BaseModel):
    course_id: Optional[int] = None
    question: str
    history: Optional[list[ChatMessage]] = None


class QAResponse(BaseModel):
    answer: str
    source_type: int
    mode: str  # rag_llm | rag_llm_ft | rag_extension | rag_fallback | rag_fallback_ft | rag_small_qa
    references: list[str] = Field(default_factory=list)


class KGNode(BaseModel):
    id: int
    label: str
    description: Optional[str] = None


class KGEdge(BaseModel):
    source: int
    target: int
    relation: str = "relates_to"
    material_id: Optional[int] = None
    cooccur_score: Optional[float] = None
    evidence_sentence: Optional[str] = None
    is_weak: bool = False
    extractor: Optional[str] = None


class KGResponse(BaseModel):
    nodes: list[KGNode]
    edges: list[KGEdge]


class KGNodeBriefOut(BaseModel):
    node_id: int
    label: str
    description: str
    generated: bool = False
    from_cache: bool = False


class KGCandidatesResponse(BaseModel):
    candidates: list[str]
    created: list[str]
    created_nodes: int = 0
    created_edges: int = 0
    filtered_noise: int = 0
    cooccur_pairs: int = 0
    extractor: str = "hybrid"
    fallback_used: bool = False
    candidate_ids: list[int] = Field(default_factory=list)
    status_stats: dict[str, int] = Field(default_factory=dict)
    edge_ready_pairs: int = 0
    edge_blocked_by_threshold: int = 0


class KGCandidateOut(BaseModel):
    id: int
    course_id: int
    material_id: int
    term: str
    source_sentence: Optional[str] = None
    status: str
    extractor: str
    fallback_used: bool = False
    score: Optional[float] = None
    created_at: datetime
    updated_at: datetime


class KGCandidateListOut(BaseModel):
    items: list[KGCandidateOut] = Field(default_factory=list)
    total: int
    page: int
    page_size: int
    status_stats: dict[str, int] = Field(default_factory=dict)


class KGCandidateBatchActionRequest(BaseModel):
    candidate_ids: list[int] = Field(default_factory=list)
    material_id: Optional[int] = None


class KGCandidateBatchActionOut(BaseModel):
    affected: int
    created_nodes: int = 0
    created_edges: int = 0
    approved_terms_in_material: int = 0
    edge_ready_pairs: int = 0
    edge_blocked_by_threshold: int = 0
    weak_edges_created: int = 0
    extractor_used: str = "hybrid"
    fallback_used: bool = False


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


class AssignmentSubmissionOut(BaseModel):
    id: int
    assignment_id: int
    assignment_title: str
    user_id: int
    user_name: str
    content: Optional[str]
    code: Optional[str]
    feedback: Optional[str]
    score: Optional[int]
    ai_feedback: Optional[str] = None
    ai_score: Optional[int] = None
    teacher_feedback: Optional[str] = None
    teacher_score: Optional[int] = None
    feedback_preview: Optional[str] = None
    created_at: datetime


class SubmissionFeedbackOut(BaseModel):
    submission_id: int
    feedback: str
    score: Optional[int]
    created_at: datetime


class SubmissionReviewUpdate(BaseModel):
    feedback: Optional[str] = None
    score: Optional[int] = None


class SubmissionMineOut(BaseModel):
    id: int
    assignment_id: int
    assignment_title: str
    course_id: int
    content: Optional[str]
    code: Optional[str]
    ai_feedback: Optional[str] = None
    ai_score: Optional[int] = None
    teacher_feedback: Optional[str] = None
    teacher_score: Optional[int] = None
    final_feedback: Optional[str] = None
    final_score: Optional[int] = None
    attempt_no: int
    latest: bool = False
    created_at: datetime


class RecommendationItem(BaseModel):
    knowledge_point: str
    reason: str
    score: float


class RecommendationOut(BaseModel):
    items: list[RecommendationItem]


class RecommendedMaterialRef(BaseModel):
    id: int
    title: str
    parse_status: str


class RecommendationEvidenceItem(BaseModel):
    source: str
    value: str
    weight: Optional[float] = None


class RecommendationBucketItem(BaseModel):
    knowledge_point: str
    reason: str
    priority: str
    weakness_score: float
    mastery_score: float
    expansion_score: float
    confidence: float
    evidence: list[RecommendationEvidenceItem] = Field(default_factory=list)
    recommended_materials: list[RecommendedMaterialRef] = Field(default_factory=list)


class LearningReportSummary(BaseModel):
    risk_level: str
    activity_level: str
    main_weak_points: list[str] = Field(default_factory=list)
    generated_at: datetime
    window_days: int = 14
    note: Optional[str] = None


class LearningReportOut(BaseModel):
    summary: LearningReportSummary
    must_review: list[RecommendationBucketItem] = Field(default_factory=list)
    need_consolidate: list[RecommendationBucketItem] = Field(default_factory=list)
    need_explore: list[RecommendationBucketItem] = Field(default_factory=list)


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
