import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_MIX_DIR = ROOT_DIR / "training" / "data" / "assignment_relevance_mix"
STAGE_B_SOURCES = ("course_local", "teacher_review", "hard_negative", "eduassist_local")


def _normalize_binary_label(raw_label: Any) -> int | None:
    if raw_label is None:
        return None
    text = str(raw_label).strip().lower()
    if text in {"1", "true", "yes", "relevant", "entailment", "similar", "match"}:
        return 1
    if text in {"0", "false", "no", "off_topic", "not_relevant", "contradiction", "neutral", "mismatch"}:
        return 0

    try:
        value = int(float(text))
        return 1 if value > 0 else 0
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        payload_raw = line.strip()
        if not payload_raw:
            continue
        try:
            payload = json.loads(payload_raw)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        records.append(payload)
    return records


def _normalize_pair_record(item: dict[str, Any]) -> dict[str, Any] | None:
    question = str(item.get("question") or item.get("prompt") or item.get("title") or "").strip()
    answer = str(item.get("answer") or item.get("content") or item.get("response") or "").strip()
    label = _normalize_binary_label(item.get("label"))
    source = str(item.get("source") or "unknown").strip() or "unknown"
    if not question or not answer or label is None:
        return None
    return {
        "question": question,
        "answer": answer,
        "label": int(label),
        "source": source,
    }


def _source_mix(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in records:
        source = str(rec.get("source", "")).strip() or "unknown"
        counts[source] = counts.get(source, 0) + 1
    return counts


def _filter_stage_records(records: list[dict[str, Any]], train_stage: str | None) -> list[dict[str, Any]]:
    stage = (train_stage or "").strip().upper()
    if stage != "B":
        return records
    filtered = [rec for rec in records if str(rec.get("source", "")).lower().startswith(STAGE_B_SOURCES)]
    return filtered if filtered else records


def _prepare_local_mix_dataset(dataset_config: str | None, max_samples: int, train_stage: str | None):
    try:
        from datasets import Dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency 'datasets'. Install: pip install datasets") from exc

    base_dir = Path(dataset_config).expanduser() if dataset_config else DEFAULT_LOCAL_MIX_DIR
    if not base_dir.is_absolute():
        base_dir = ROOT_DIR / base_dir

    train_path = base_dir / "train.jsonl"
    val_path = base_dir / "validation.jsonl"
    if not train_path.exists():
        raise RuntimeError(f"Local dataset file not found: {train_path}")

    train_raw = _read_jsonl(train_path)
    val_raw = _read_jsonl(val_path)
    train_records = [row for row in (_normalize_pair_record(item) for item in train_raw) if row]
    val_records = [row for row in (_normalize_pair_record(item) for item in val_raw) if row]

    train_records = _filter_stage_records(train_records, train_stage=train_stage)
    if not train_records:
        raise RuntimeError("No usable train samples in assignment relevance dataset.")
    if not val_records:
        fallback_count = min(256, len(train_records))
        val_records = train_records[:fallback_count]

    train_records = train_records[: min(len(train_records), max_samples)]
    val_records = val_records[: min(len(val_records), max(64, max_samples // 5))]

    ds = {
        "train": Dataset.from_list(train_records),
        "validation": Dataset.from_list(val_records),
    }
    return ds, f"local_jsonl:{base_dir}", _source_mix(train_records)


def _load_ocnli(dataset_config: str | None, max_samples: int):
    try:
        from datasets import Dataset, load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency 'datasets'. Install: pip install datasets") from exc

    config = dataset_config or "ocnli"
    ds = load_dataset("clue", config)

    def _map_split(split):
        rows = []
        for item in split:
            premise = str(item.get("sentence1", "")).strip()
            hypothesis = str(item.get("sentence2", "")).strip()
            label_raw = item.get("label")
            label_text = str(item.get("label_des", "")).strip().lower()
            if not premise or not hypothesis:
                continue
            if label_text:
                label = 1 if label_text in {"entailment", "蕴含"} else 0
            else:
                label = _normalize_binary_label(label_raw)
                if label is None:
                    label = 0
            rows.append({"question": premise, "answer": hypothesis, "label": int(label), "source": "ocnli"})
        return rows

    train_rows = _map_split(ds["train"])[:max_samples]
    val_source = ds["validation"] if "validation" in ds else ds["train"]
    val_rows = _map_split(val_source)[: max(64, max_samples // 5)]

    return {"train": Dataset.from_list(train_rows), "validation": Dataset.from_list(val_rows)}, "hf:clue/ocnli", {
        "ocnli": len(train_rows)
    }


def _load_lcqmc(max_samples: int):
    try:
        from datasets import Dataset, load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency 'datasets'. Install: pip install datasets") from exc

    ds = load_dataset("lcqmc")

    def _map_split(split):
        rows = []
        for item in split:
            q1 = str(item.get("sentence1", "")).strip()
            q2 = str(item.get("sentence2", "")).strip()
            label = _normalize_binary_label(item.get("label"))
            if label is None or not q1 or not q2:
                continue
            rows.append({"question": q1, "answer": q2, "label": int(label), "source": "lcqmc"})
        return rows

    train_rows = _map_split(ds["train"])[:max_samples]
    val_source = ds["validation"] if "validation" in ds else ds["train"]
    val_rows = _map_split(val_source)[: max(64, max_samples // 5)]
    return {"train": Dataset.from_list(train_rows), "validation": Dataset.from_list(val_rows)}, "hf:lcqmc", {
        "lcqmc": len(train_rows)
    }


def _prepare_dataset(dataset_name: str, dataset_config: str | None, max_samples: int, train_stage: str | None):
    dataset_norm = (dataset_name or "").strip().lower()
    if dataset_norm in {"assignment_relevance_mix_local", "local_jsonl"}:
        return _prepare_local_mix_dataset(
            dataset_config=dataset_config,
            max_samples=max_samples,
            train_stage=train_stage,
        )
    if dataset_norm in {"ocnli", "clue_ocnli"}:
        return _load_ocnli(dataset_config=dataset_config, max_samples=max_samples)
    if dataset_norm == "lcqmc":
        return _load_lcqmc(max_samples=max_samples)
    raise ValueError(f"Unsupported assignment relevance dataset: {dataset_name}")


def _calibrate_thresholds(
    relevant_probs: np.ndarray,
    labels: np.ndarray,
    target_recall_relevant: float = 0.95,
    target_precision_off_topic: float = 0.80,
) -> dict[str, float]:
    probs = np.asarray(relevant_probs, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    if probs.size == 0 or y.size == 0:
        return {"threshold_hi": 0.70, "threshold_lo": 0.25}

    candidates = np.unique(np.clip(probs, 0.0, 1.0))
    if candidates.size < 16:
        candidates = np.linspace(0.0, 1.0, num=101)

    threshold_hi = 0.70
    for th in sorted(candidates):
        pred_rel = (probs >= th).astype(int)
        recall_rel = recall_score(y, pred_rel, pos_label=1, zero_division=0)
        if recall_rel >= target_recall_relevant:
            threshold_hi = float(th)

    threshold_lo = 0.25
    best_lo = threshold_lo
    for th in sorted(candidates):
        pred_off = (probs <= th).astype(int)
        true_off = (y == 0).astype(int)
        precision_off = precision_score(true_off, pred_off, pos_label=1, zero_division=0)
        if precision_off >= target_precision_off_topic:
            best_lo = float(th)
    threshold_lo = min(best_lo, threshold_hi)

    return {
        "threshold_hi": round(float(threshold_hi), 4),
        "threshold_lo": round(float(threshold_lo), 4),
    }


def train(
    dataset_name: str,
    model_name: str,
    output_dir: str,
    max_samples: int = 20000,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    dataset_config: str | None = None,
    train_stage: str | None = None,
) -> dict[str, Any]:
    try:
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency 'transformers/torch'. Install: pip install transformers torch") from exc

    start = time.time()
    ds, dataset_desc, source_mix = _prepare_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        max_samples=max_samples,
        train_stage=train_stage,
    )
    train_ds = ds["train"]
    eval_ds = ds["validation"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            batch["question"],
            batch["answer"],
            truncation=True,
            padding="max_length",
            max_length=384,
        )

    train_tok = train_ds.map(tokenize_fn, batched=True)
    eval_tok = eval_ds.map(tokenize_fn, batched=True)
    keep_cols = {"input_ids", "attention_mask", "token_type_ids", "label"}
    train_tok = train_tok.remove_columns([c for c in train_tok.column_names if c not in keep_cols])
    eval_tok = eval_tok.remove_columns([c for c in eval_tok.column_names if c not in keep_cols])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "off_topic", 1: "relevant"},
        label2id={"off_topic": 0, "relevant": 1},
    )

    raw_args = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": 2,
        "learning_rate": learning_rate,
        "warmup_ratio": 0.06,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_relevant",
        "greater_is_better": True,
        "report_to": [],
        "bf16": bool(torch.cuda.is_available()),
        "gradient_checkpointing": True,
    }
    valid_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    if "evaluation_strategy" not in valid_params and "eval_strategy" in valid_params:
        raw_args["eval_strategy"] = raw_args.pop("evaluation_strategy")
    elif "evaluation_strategy" not in valid_params:
        raw_args.pop("evaluation_strategy")
    args = TrainingArguments(**{k: v for k, v in raw_args.items() if k in valid_params})

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = float(accuracy_score(labels, preds))
        f1_macro = float(f1_score(labels, preds, average="macro"))
        f1_relevant = float(f1_score(labels, preds, pos_label=1))
        recall_relevant = float(recall_score(labels, preds, pos_label=1, zero_division=0))
        precision_off_topic = float(precision_score(labels, preds, pos_label=0, zero_division=0))
        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_relevant": f1_relevant,
            "recall_relevant": recall_relevant,
            "precision_off_topic": precision_off_topic,
        }

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_tok,
        "eval_dataset": eval_tok,
        "compute_metrics": compute_metrics,
    }
    trainer_params = set(inspect.signature(Trainer.__init__).parameters.keys())
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    eval_pred = trainer.predict(eval_tok)
    probs = np.exp(eval_pred.predictions - np.max(eval_pred.predictions, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    relevant_probs = probs[:, 1]
    labels = np.array(eval_tok["label"], dtype=int)
    calibration = _calibrate_thresholds(
        relevant_probs,
        labels,
        target_recall_relevant=0.95,
        target_precision_off_topic=0.80,
    )

    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "relevance_calibration.json"), "w", encoding="utf-8") as fp:
        json.dump(calibration, fp, ensure_ascii=False, indent=2)

    metrics = compute_metrics((eval_pred.predictions, labels))
    stage_label = (train_stage or "A").strip().upper()[:1] if train_stage else "A"
    return {
        "dataset": dataset_desc,
        "dataset_config": dataset_config,
        "task_type": "assignment_relevance_hf",
        "model_name": model_name,
        "sample_size": int(len(train_tok)),
        "eval_size": int(len(eval_tok)),
        "train_time_sec": round(time.time() - start, 2),
        "accuracy": round(metrics["accuracy"], 4),
        "f1_macro": round(metrics["f1_macro"], 4),
        "f1_relevant": round(metrics["f1_relevant"], 4),
        "relevant_recall": round(metrics["recall_relevant"], 4),
        "off_topic_precision": round(metrics["precision_off_topic"], 4),
        "threshold_hi": calibration["threshold_hi"],
        "threshold_lo": calibration["threshold_lo"],
        "source_mix": source_mix,
        "train_stage": stage_label,
        "model_path": output_dir,
    }
