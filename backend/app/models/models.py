from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, UniqueConstraint, Float
from sqlalchemy.orm import relationship
from app.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    real_name = Column(String(50), nullable=True)
    role = Column(Integer, nullable=False, default=0)  # 0 student, 1 teacher
    created_at = Column(DateTime, default=datetime.utcnow)

    courses = relationship("Course", back_populates="teacher")
    enrollments = relationship("UserCourse", back_populates="user")


class Course(Base):
    __tablename__ = "courses"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    teacher_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    teacher = relationship("User", back_populates="courses")
    materials = relationship("Material", back_populates="course")
    enrollments = relationship("UserCourse", back_populates="course")


class UserCourse(Base):
    __tablename__ = "user_course"
    __table_args__ = (UniqueConstraint("user_id", "course_id", name="uq_user_course"),)

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    role = Column(Integer, default=0)  # 0 student, 1 teacher assistant
    joined_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="enrollments")
    course = relationship("Course", back_populates="enrollments")


class Material(Base):
    __tablename__ = "materials"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    title = Column(String(200), nullable=False)
    file_path = Column(String(255), nullable=False)
    file_type = Column(String(20), nullable=False)
    parse_status = Column(String(20), nullable=False, default="pending")
    parse_error = Column(Text, nullable=True)
    extracted_chars = Column(Integer, nullable=False, default=0)
    parsed_at = Column(DateTime, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    course = relationship("Course", back_populates="materials")


class QARecord(Base):
    __tablename__ = "qa_records"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    source_type = Column(Integer, nullable=False, default=0)  # 0 material, 1 extension
    created_at = Column(DateTime, default=datetime.utcnow)


class KnowledgePoint(Base):
    __tablename__ = "knowledge_points"

    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)


class KnowledgeRelation(Base):
    __tablename__ = "knowledge_relations"

    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    source_id = Column(Integer, ForeignKey("knowledge_points.id"), nullable=False)
    target_id = Column(Integer, ForeignKey("knowledge_points.id"), nullable=False)
    relation = Column(String(50), default="relates_to")
    material_id = Column(Integer, ForeignKey("materials.id"), nullable=True)
    cooccur_score = Column(Float, nullable=True)
    evidence_sentence = Column(Text, nullable=True)
    is_weak = Column(Integer, nullable=False, default=0)
    extractor = Column(String(30), nullable=True)


class KnowledgeEdge(Base):
    __tablename__ = "knowledge_edges"

    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    source_id = Column(Integer, ForeignKey("knowledge_points.id"), nullable=False)
    target_id = Column(Integer, ForeignKey("knowledge_points.id"), nullable=False)
    relation = Column(String(50), default="relates_to")


class KnowledgeCandidate(Base):
    __tablename__ = "knowledge_candidates"
    __table_args__ = (
        UniqueConstraint("course_id", "material_id", "term_norm", name="uq_kg_candidate_material_term"),
    )

    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    material_id = Column(Integer, ForeignKey("materials.id"), nullable=False)
    term = Column(String(120), nullable=False)
    term_norm = Column(String(120), nullable=False)
    source_sentence = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="pending")  # pending|approved|rejected
    extractor = Column(String(30), nullable=False, default="hybrid")
    fallback_used = Column(Integer, nullable=False, default=0)
    score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Assignment(Base):
    __tablename__ = "assignments"

    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    type = Column(String(20), nullable=False, default="text")  # text | code
    keywords = Column(Text, nullable=True)  # comma-separated keywords for text grading
    created_at = Column(DateTime, default=datetime.utcnow)


class Submission(Base):
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True)
    assignment_id = Column(Integer, ForeignKey("assignments.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=True)  # text submission
    code = Column(Text, nullable=True)  # code submission (not executed)
    feedback = Column(Text, nullable=True)  # kept for backward compatibility
    score = Column(Integer, nullable=True)  # kept for backward compatibility
    created_at = Column(DateTime, default=datetime.utcnow)


class SubmissionFeedback(Base):
    __tablename__ = "submission_feedback"

    id = Column(Integer, primary_key=True)
    submission_id = Column(Integer, ForeignKey("submissions.id"), nullable=False, unique=True)
    feedback = Column(Text, nullable=False)
    score = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class LearningEvent(Base):
    __tablename__ = "learning_events"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    event_type = Column(String(30), nullable=False)  # qa | material | assignment
    content = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True)
    task_type = Column(String(30), nullable=False, default="text_classification")
    dataset_name = Column(String(200), nullable=True)
    params = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="queued")  # queued | running | success | failed
    metrics = Column(Text, nullable=True)
    model_path = Column(String(255), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
