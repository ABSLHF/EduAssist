import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_MIX_DIR = ROOT_DIR / "training" / "data" / "assignment_feedback_mix"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    output: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except Exception:
            continue
        if isinstance(item, dict):
            output.append(item)
    return output


def _normalize_label(raw: Any) -> int:
    text = str(raw or "").strip().lower()
    if text == "off_topic":
        return 0
    if text in {"partial", "unknown", "neutral", "ok"}:
        return 1
    if text in {"relevant", "good", "correct"}:
        return 2
    return 1


def _normalize_record(item: dict[str, Any]) -> dict[str, Any] | None:
    question = str(item.get("question") or item.get("prompt") or "").strip()
    reference_answer = str(item.get("reference_answer") or item.get("reference") or "").strip()
    student_answer = str(item.get("student_answer") or item.get("answer") or "").strip()
    rubric = item.get("rubric_labels") if isinstance(item.get("rubric_labels"), dict) else {}
    label = _normalize_label((rubric or {}).get("relevance") or item.get("label"))
    source = str(item.get("source") or "unknown").strip() or "unknown"
    if not question or not student_answer:
        return None
    return {
        "question": question,
        "reference_answer": reference_answer,
        "student_answer": student_answer,
        "label": label,
        "source": source,
    }


def _source_mix(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        source = str(row.get("source", "")).strip() or "unknown"
        counts[source] = counts.get(source, 0) + 1
    return counts


def _prepare_local_mix_dataset(dataset_config: str | None, max_samples: int):
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
    train_rows = [row for row in (_normalize_record(item) for item in train_raw) if row]
    val_rows = [row for row in (_normalize_record(item) for item in val_raw) if row]

    if not train_rows:
        raise RuntimeError("No usable train samples in assignment feedback dataset.")
    if not val_rows:
        fallback_count = min(512, len(train_rows))
        val_rows = train_rows[:fallback_count]

    train_rows = train_rows[: min(len(train_rows), max_samples)]
    val_rows = val_rows[: min(len(val_rows), max(128, max_samples // 5))]
    ds = {"train": Dataset.from_list(train_rows), "validation": Dataset.from_list(val_rows)}
    return ds, f"local_jsonl:{base_dir}", _source_mix(train_rows)


def train(
    dataset_name: str,
    model_name: str,
    output_dir: str,
    max_samples: int = 20000,
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    dataset_config: str | None = None,
) -> dict[str, Any]:
    _ = dataset_name
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
    ds, dataset_desc, source_mix = _prepare_local_mix_dataset(dataset_config=dataset_config, max_samples=max_samples)
    train_ds = ds["train"]
    eval_ds = ds["validation"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch: dict[str, Any]) -> dict[str, Any]:
        text_a = [f"题目：{q}\n参考答案：{r or '无'}" for q, r in zip(batch["question"], batch["reference_answer"])]
        text_b = [f"学生答案：{a}" for a in batch["student_answer"]]
        return tokenizer(text_a, text_b, truncation=True, padding="max_length", max_length=384)

    train_tok = train_ds.map(tokenize_fn, batched=True)
    eval_tok = eval_ds.map(tokenize_fn, batched=True)
    keep_cols = {"input_ids", "attention_mask", "token_type_ids", "label"}
    train_tok = train_tok.remove_columns([c for c in train_tok.column_names if c not in keep_cols])
    eval_tok = eval_tok.remove_columns([c for c in eval_tok.column_names if c not in keep_cols])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label={0: "weak", 1: "ok", 2: "good"},
        label2id={"weak": 0, "ok": 1, "good": 2},
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
        "metric_for_best_model": "f1_good",
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
        f1_good = float(f1_score(labels, preds, labels=[2], average="macro", zero_division=0))
        recall_good = float(recall_score(labels, preds, labels=[2], average="macro", zero_division=0))
        precision_weak = float(precision_score(labels, preds, labels=[0], average="macro", zero_division=0))
        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_good": f1_good,
            "recall_good": recall_good,
            "precision_weak": precision_weak,
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
    metrics = compute_metrics((eval_pred.predictions, np.array(eval_tok["label"], dtype=int)))

    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {
        "dataset": dataset_desc,
        "dataset_config": dataset_config,
        "task_type": "assignment_feedback_hf",
        "model_name": model_name,
        "sample_size": int(len(train_tok)),
        "eval_size": int(len(eval_tok)),
        "train_time_sec": round(time.time() - start, 2),
        "accuracy": round(metrics["accuracy"], 4),
        "f1_macro": round(metrics["f1_macro"], 4),
        "f1_good": round(metrics["f1_good"], 4),
        "recall_good": round(metrics["recall_good"], 4),
        "precision_weak": round(metrics["precision_weak"], 4),
        "source_mix": source_mix,
        "model_path": output_dir,
    }
