from pathlib import Path

from sqlalchemy.orm import Session

from app.config import settings
from app.models import models

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _normalize_existing_path(raw_path: str | None) -> str | None:
    if not raw_path:
        return None

    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return str(candidate) if candidate.exists() else None

    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return str(cwd_candidate.resolve())

    backend_candidate = BACKEND_ROOT / candidate
    if backend_candidate.exists():
        return str(backend_candidate.resolve())
    return None


def latest_model_run_path(db: Session, task_type: str) -> str | None:
    row = (
        db.query(models.ModelRun)
        .filter(models.ModelRun.task_type == task_type, models.ModelRun.status == "success")
        .order_by(models.ModelRun.updated_at.desc())
        .first()
    )
    if not row:
        return None
    return _normalize_existing_path(row.model_path)


def latest_qa_model_dir_from_fs() -> str | None:
    models_dir = BACKEND_ROOT / "models"
    if not models_dir.exists():
        return None

    candidates: list[Path] = []
    for child in models_dir.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("qa_"):
            continue
        if not (child / "config.json").exists():
            continue
        candidates.append(child)

    if not candidates:
        return None

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest.resolve())


def resolve_active_qa_model_path(db: Session) -> tuple[str | None, str]:
    env_path = _normalize_existing_path((settings.finetuned_qa_model_path or "").strip())
    if env_path:
        return env_path, "env"

    run_path = latest_model_run_path(db, "qa_extractive_hf")
    if run_path:
        return run_path, "model_run_latest"

    fs_path = latest_qa_model_dir_from_fs()
    if fs_path:
        return fs_path, "models_dir_latest"

    return None, "none"


def latest_assignment_relevance_model_dir_from_fs() -> str | None:
    models_dir = BACKEND_ROOT / "models"
    if not models_dir.exists():
        return None

    candidates: list[Path] = []
    for child in models_dir.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("assignment_rel_"):
            continue
        if not (child / "config.json").exists():
            continue
        candidates.append(child)

    if not candidates:
        return None

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest.resolve())


def resolve_active_assignment_relevance_model_path(db: Session | None) -> tuple[str | None, str]:
    env_path = _normalize_existing_path((settings.assignment_relevance_model_path or "").strip())
    if env_path:
        return env_path, "env"

    if db is not None:
        run_path = latest_model_run_path(db, "assignment_relevance_hf")
        if run_path:
            return run_path, "model_run_latest"

    fs_path = latest_assignment_relevance_model_dir_from_fs()
    if fs_path:
        return fs_path, "models_dir_latest"

    return None, "none"


def latest_assignment_feedback_model_dir_from_fs() -> str | None:
    models_dir = BACKEND_ROOT / "models"
    if not models_dir.exists():
        return None

    candidates: list[Path] = []
    for child in models_dir.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("assignment_feedback_"):
            continue
        if not (child / "config.json").exists():
            continue
        candidates.append(child)

    if not candidates:
        return None

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest.resolve())


def resolve_active_assignment_feedback_model_path(db: Session | None) -> tuple[str | None, str]:
    env_path = _normalize_existing_path((settings.assignment_feedback_model_path or "").strip())
    if env_path:
        return env_path, "env"

    if db is not None:
        run_path = latest_model_run_path(db, "assignment_feedback_hf")
        if run_path:
            return run_path, "model_run_latest"

    fs_path = latest_assignment_feedback_model_dir_from_fs()
    if fs_path:
        return fs_path, "models_dir_latest"

    return None, "none"


def latest_assignment_feedback_sft_model_dir_from_fs() -> str | None:
    models_dir = BACKEND_ROOT / "models"
    if not models_dir.exists():
        return None

    candidates: list[Path] = []
    for child in models_dir.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("assignment_feedback_sft_"):
            continue
        if not (child / "config.json").exists() and not (child / "adapter_config.json").exists():
            continue
        candidates.append(child)

    if not candidates:
        return None

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest.resolve())


def resolve_active_assignment_feedback_sft_model_path(db: Session | None) -> tuple[str | None, str]:
    env_path = _normalize_existing_path((settings.assignment_feedback_sft_model_path or "").strip())
    if env_path:
        return env_path, "env"

    if db is not None:
        run_path = latest_model_run_path(db, "assignment_feedback_sft_hf")
        if run_path:
            return run_path, "model_run_latest"

    fs_path = latest_assignment_feedback_sft_model_dir_from_fs()
    if fs_path:
        return fs_path, "models_dir_latest"

    return None, "none"
