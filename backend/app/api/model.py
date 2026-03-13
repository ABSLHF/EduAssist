import json
import threading
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_teacher
from app.config import settings
from app.db import get_db
from app.models import models
from app.schemas.schemas import (
    ModelRunOut,
    ModelTrainRequest,
    ModelTrainResponse,
    QAPredictRequest,
    QAPredictResponse,
)
from app.services.classifier import predict_label
from app.services.model_paths import (
    resolve_active_assignment_feedback_model_path,
    resolve_active_assignment_feedback_sft_model_path,
    resolve_active_assignment_relevance_model_path,
    resolve_active_qa_model_path,
)
from app.services.qa_small_model import predict_answer
from training.train import train as run_training_local
from training.train_assignment_feedback_hf import train as run_training_assignment_feedback_hf
from training.train_assignment_feedback_sft_hf import train as run_training_assignment_feedback_sft_hf
from training.train_assignment_relevance_hf import train as run_training_assignment_relevance_hf
from training.train_cls_hf import train as run_training_cls_hf
from training.train_qa_hf import train as run_training_qa_hf

router = APIRouter()


class PredictRequest(BaseModel):
    text: str
    model_path: str | None = None


class ActiveModelResponse(BaseModel):
    qa_model_path: str | None
    qa_model_source: str
    enable_finetuned_qa_in_rag: bool
    finetuned_qa_min_conf: float
    finetuned_qa_top_chunks: int
    finetuned_qa_max_evidence: int
    qa_train_stage: str | None = None
    qa_dataset_manifest: str | None = None
    assignment_relevance_model_path: str | None = None
    assignment_relevance_model_source: str = "none"
    enable_assignment_relevance_model: bool = False
    assignment_relevance_threshold_hi: float = 0.7
    assignment_relevance_threshold_lo: float = 0.25
    assignment_feedback_model_path: str | None = None
    assignment_feedback_model_source: str = "none"
    enable_assignment_feedback_model: bool = False
    assignment_feedback_sft_model_path: str | None = None
    assignment_feedback_sft_model_source: str = "none"
    enable_assignment_feedback_sft_model: bool = False
    assignment_feedback_pipeline_version: str = "single_v2"


def _timestamped_model_dir(prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"models/{prefix}_{stamp}"


def _latest_model_path_by_task(db: Session, task_type: str) -> str | None:
    row = (
        db.query(models.ModelRun)
        .filter(models.ModelRun.task_type == task_type, models.ModelRun.status == "success")
        .order_by(models.ModelRun.updated_at.desc())
        .first()
    )
    if not row:
        return None
    return row.model_path


def _latest_qa_run_metrics(db: Session) -> dict:
    row = (
        db.query(models.ModelRun)
        .filter(models.ModelRun.task_type == "qa_extractive_hf", models.ModelRun.status == "success")
        .order_by(models.ModelRun.updated_at.desc())
        .first()
    )
    if not row or not row.metrics:
        return {}
    try:
        payload = json.loads(row.metrics)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _latest_assignment_relevance_metrics(db: Session) -> dict:
    row = (
        db.query(models.ModelRun)
        .filter(models.ModelRun.task_type == "assignment_relevance_hf", models.ModelRun.status == "success")
        .order_by(models.ModelRun.updated_at.desc())
        .first()
    )
    if not row or not row.metrics:
        return {}
    try:
        payload = json.loads(row.metrics)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _execute_training(payload: ModelTrainRequest) -> tuple[str, dict]:
    task_type = payload.task_type

    if task_type == "text_classification":
        model_path = "models/keyword_clf.joblib"
        metrics = run_training_local(
            csv_path="training/data/sample_labels.csv",
            model_path=model_path,
            dataset_name=payload.dataset_name,
            max_samples=payload.max_samples,
        )
        return model_path, metrics

    if task_type == "text_classification_hf":
        model_name = payload.model_name or "bert-base-chinese"
        model_path = _timestamped_model_dir("cls")
        metrics = run_training_cls_hf(
            dataset_name=payload.dataset_name or "clue",
            dataset_config=payload.dataset_config,
            model_name=model_name,
            output_dir=model_path,
            max_samples=payload.max_samples,
            epochs=payload.epochs,
            batch_size=payload.batch_size,
            learning_rate=payload.learning_rate,
        )
        return model_path, metrics

    if task_type == "qa_extractive_hf":
        model_name = payload.model_name or "bert-base-chinese"
        dataset_name = payload.dataset_name or "cmrc2018"
        if dataset_name == "edu_mix_qa_local":
            stage_a_samples = payload.max_samples if payload.max_samples != 1200 else 8000
            stage_b_samples = min(max(stage_a_samples // 4, 1200), 2000)
            stage_batch = max(1, payload.batch_size // 4)
            stage_a_path = _timestamped_model_dir("qa_stage_a")
            stage_a_metrics = run_training_qa_hf(
                dataset_name=dataset_name,
                dataset_config=payload.dataset_config,
                model_name=model_name,
                output_dir=stage_a_path,
                max_samples=stage_a_samples,
                epochs=max(1, payload.epochs),
                batch_size=stage_batch,
                learning_rate=payload.learning_rate,
                train_stage="A",
            )
            stage_b_path = _timestamped_model_dir("qa_stage_b")
            stage_b_metrics = run_training_qa_hf(
                dataset_name=dataset_name,
                dataset_config=payload.dataset_config,
                model_name=stage_a_path,
                output_dir=stage_b_path,
                max_samples=stage_b_samples,
                epochs=1,
                batch_size=stage_batch,
                learning_rate=min(payload.learning_rate, 2e-5),
                train_stage="B",
            )
            merged_metrics = {
                **stage_b_metrics,
                "pipeline": "two_stage_A_to_B",
                "stage_a": stage_a_metrics,
                "stage_b": stage_b_metrics,
            }
            return stage_b_path, merged_metrics

        model_path = _timestamped_model_dir("qa")
        metrics = run_training_qa_hf(
            dataset_name=dataset_name,
            dataset_config=payload.dataset_config,
            model_name=model_name,
            output_dir=model_path,
            max_samples=payload.max_samples,
            epochs=payload.epochs,
            batch_size=max(1, payload.batch_size // 2),
            learning_rate=payload.learning_rate,
            train_stage="A",
        )
        return model_path, metrics

    if task_type == "assignment_relevance_hf":
        model_name = payload.model_name or "hfl/chinese-roberta-wwm-ext"
        dataset_name = payload.dataset_name or "assignment_relevance_mix_local"

        stage_a_samples = payload.max_samples if payload.max_samples != 1200 else 24000
        stage_b_samples = min(max(stage_a_samples // 2, 6000), stage_a_samples)
        stage_a_epochs = payload.epochs if payload.epochs > 1 else 3
        stage_b_epochs = max(1, payload.epochs - 1) if payload.epochs > 1 else 2
        stage_batch = max(8, payload.batch_size)

        stage_a_path = _timestamped_model_dir("assignment_rel_stage_a")
        stage_a_metrics = run_training_assignment_relevance_hf(
            dataset_name=dataset_name,
            dataset_config=payload.dataset_config,
            model_name=model_name,
            output_dir=stage_a_path,
            max_samples=stage_a_samples,
            epochs=stage_a_epochs,
            batch_size=stage_batch,
            learning_rate=payload.learning_rate,
            train_stage="A",
        )
        stage_b_path = _timestamped_model_dir("assignment_rel_stage_b")
        stage_b_metrics = run_training_assignment_relevance_hf(
            dataset_name=dataset_name,
            dataset_config=payload.dataset_config,
            model_name=stage_a_path,
            output_dir=stage_b_path,
            max_samples=stage_b_samples,
            epochs=stage_b_epochs,
            batch_size=stage_batch,
            learning_rate=min(payload.learning_rate, 2e-5),
            train_stage="B",
        )
        merged_metrics = {
            **stage_b_metrics,
            "pipeline": "two_stage_A_to_B",
            "stage_a": stage_a_metrics,
            "stage_b": stage_b_metrics,
        }
        return stage_b_path, merged_metrics

    if task_type == "assignment_feedback_hf":
        model_name = payload.model_name or "hfl/chinese-roberta-wwm-ext"
        dataset_name = payload.dataset_name or "assignment_feedback_mix_local"
        model_path = _timestamped_model_dir("assignment_feedback")
        metrics = run_training_assignment_feedback_hf(
            dataset_name=dataset_name,
            dataset_config=payload.dataset_config,
            model_name=model_name,
            output_dir=model_path,
            max_samples=payload.max_samples,
            epochs=max(1, payload.epochs),
            batch_size=max(4, payload.batch_size),
            learning_rate=payload.learning_rate,
        )
        return model_path, metrics

    if task_type == "assignment_feedback_sft_hf":
        model_name = payload.model_name or "Qwen/Qwen2.5-7B-Instruct"
        dataset_name = payload.dataset_name or "assignment_feedback_sft_mix"
        model_path = _timestamped_model_dir("assignment_feedback_sft")
        metrics = run_training_assignment_feedback_sft_hf(
            dataset_name=dataset_name,
            dataset_config=payload.dataset_config,
            model_name=model_name,
            output_dir=model_path,
            max_samples=payload.max_samples,
            epochs=max(1, payload.epochs),
            batch_size=max(1, payload.batch_size),
            learning_rate=payload.learning_rate,
        )
        return model_path, metrics

    raise ValueError(f"Unsupported task_type: {task_type}")


def _run_training_job(run_id: int, payload: ModelTrainRequest, session_factory):
    db = session_factory()
    try:
        run = db.query(models.ModelRun).get(run_id)
        if not run:
            return
        run.status = "running"
        db.commit()

        model_path, metrics = _execute_training(payload)
        run.status = "success"
        run.model_path = model_path
        run.metrics = json.dumps(metrics, ensure_ascii=False)
        run.error_message = None
        db.commit()
    except Exception as exc:
        run = db.query(models.ModelRun).get(run_id)
        if run:
            run.status = "failed"
            run.error_message = str(exc)
            db.commit()
    finally:
        db.close()


@router.post("/predict", summary="知识点分类预测")
def predict(payload: PredictRequest, db: Session = Depends(get_db), teacher=Depends(require_teacher)):
    model_path = payload.model_path or _latest_model_path_by_task(db, "text_classification_hf")
    if not model_path:
        model_path = "models/keyword_clf.joblib"
    try:
        label = predict_label(payload.text, path=model_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Model load or inference failed: {exc}")
    if label is None:
        raise HTTPException(status_code=400, detail="Model not trained. Run training first.")
    return {"label": label, "model_path": model_path}


@router.post("/qa_predict", response_model=QAPredictResponse, summary="抽取式问答预测")
def qa_predict(payload: QAPredictRequest, db: Session = Depends(get_db), teacher=Depends(require_teacher)):
    model_path = payload.model_path or _latest_model_path_by_task(db, "qa_extractive_hf")
    if not model_path:
        raise HTTPException(status_code=400, detail="QA model not trained. Run qa_extractive_hf training first.")
    answer, confidence = predict_answer(payload.question, payload.context, model_path=model_path)
    return {"answer": answer, "confidence": confidence}


@router.post("/train", response_model=ModelTrainResponse, summary="触发训练任务")
def start_training(payload: ModelTrainRequest, db: Session = Depends(get_db), teacher=Depends(require_teacher)):
    run = models.ModelRun(
        task_type=payload.task_type,
        dataset_name=payload.dataset_name,
        params=json.dumps(payload.model_dump(), ensure_ascii=False),
        status="queued",
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    from app.db import SessionLocal

    threading.Thread(
        target=_run_training_job,
        args=(run.id, payload, SessionLocal),
        daemon=True,
    ).start()
    return {"job_id": run.id, "status": run.status}


@router.get("/train/{job_id}", response_model=ModelRunOut, summary="查询训练任务状态")
def get_training_job(job_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    run = db.query(models.ModelRun).get(job_id)
    if not run:
        raise HTTPException(status_code=404, detail="Job not found")
    return run


@router.get("/active", response_model=ActiveModelResponse, summary="当前生效微调模型")
def get_active_model(db: Session = Depends(get_db), user=Depends(get_current_user)):
    qa_model_path, qa_model_source = resolve_active_qa_model_path(db)
    latest_metrics = _latest_qa_run_metrics(db)
    assignment_model_path, assignment_model_source = resolve_active_assignment_relevance_model_path(db)
    assignment_feedback_model_path, assignment_feedback_model_source = resolve_active_assignment_feedback_model_path(db)
    assignment_feedback_sft_model_path, assignment_feedback_sft_model_source = resolve_active_assignment_feedback_sft_model_path(
        db
    )
    latest_assignment_metrics = _latest_assignment_relevance_metrics(db)
    manifest_hint = None
    dataset_name = str(latest_metrics.get("dataset", ""))
    dataset_config = str(latest_metrics.get("dataset_config", "") or "")
    if dataset_name.startswith("local_jsonl:"):
        base_dir = Path(dataset_name.replace("local_jsonl:", "", 1).strip())
        manifest_hint = str((base_dir / "manifest.json"))
    elif dataset_config:
        cfg_path = Path(dataset_config).expanduser()
        manifest_hint = str((cfg_path / "manifest.json")) if cfg_path.suffix == "" else str(cfg_path)

    assignment_threshold_hi = float(
        latest_assignment_metrics.get("threshold_hi", settings.assignment_relevance_threshold_hi)
    )
    assignment_threshold_lo = float(
        latest_assignment_metrics.get("threshold_lo", settings.assignment_relevance_threshold_lo)
    )

    return {
        "qa_model_path": qa_model_path,
        "qa_model_source": qa_model_source,
        "enable_finetuned_qa_in_rag": settings.enable_finetuned_qa_in_rag,
        "finetuned_qa_min_conf": settings.finetuned_qa_min_conf,
        "finetuned_qa_top_chunks": settings.finetuned_qa_top_chunks,
        "finetuned_qa_max_evidence": settings.finetuned_qa_max_evidence,
        "qa_train_stage": latest_metrics.get("train_stage"),
        "qa_dataset_manifest": manifest_hint,
        "assignment_relevance_model_path": assignment_model_path,
        "assignment_relevance_model_source": assignment_model_source,
        "enable_assignment_relevance_model": settings.enable_assignment_relevance_model,
        "assignment_relevance_threshold_hi": assignment_threshold_hi,
        "assignment_relevance_threshold_lo": assignment_threshold_lo,
        "assignment_feedback_model_path": assignment_feedback_model_path,
        "assignment_feedback_model_source": assignment_feedback_model_source,
        "enable_assignment_feedback_model": settings.enable_assignment_feedback_model,
        "assignment_feedback_sft_model_path": assignment_feedback_sft_model_path,
        "assignment_feedback_sft_model_source": assignment_feedback_sft_model_source,
        "enable_assignment_feedback_sft_model": settings.enable_assignment_feedback_sft_model,
        "assignment_feedback_pipeline_version": settings.assignment_feedback_pipeline_version,
    }
