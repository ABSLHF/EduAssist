import json
import threading
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_teacher
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
from app.services.qa_small_model import predict_answer
from training.train import train as run_training_local
from training.train_cls_hf import train as run_training_cls_hf
from training.train_qa_hf import train as run_training_qa_hf

router = APIRouter()


class PredictRequest(BaseModel):
    text: str
    model_path: str | None = None


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
        model_path = _timestamped_model_dir("qa")
        metrics = run_training_qa_hf(
            dataset_name=payload.dataset_name or "cmrc2018",
            dataset_config=payload.dataset_config,
            model_name=model_name,
            output_dir=model_path,
            max_samples=payload.max_samples,
            epochs=payload.epochs,
            batch_size=max(1, payload.batch_size // 2),
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
