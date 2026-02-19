import os
import time
import inspect
import json
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _pick_text_column(columns: list[str]) -> str:
    candidates = ["text", "sentence", "content", "sentence1", "query"]
    for col in candidates:
        if col in columns:
            return col
    for col in columns:
        if col != "label":
            return col
    raise ValueError("No text column found in dataset.")


def _load_dataset(dataset_name: str, dataset_config: str | None):
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency 'datasets'. Install: pip install datasets") from exc

    if dataset_name == "clue":
        config = dataset_config or "iflytek"
        return load_dataset("clue", config)
    if dataset_config:
        return load_dataset(dataset_name, dataset_config)
    return load_dataset(dataset_name)


def _normalize_label_column(ds) -> tuple[object, dict[int, str], dict[str, int]]:
    from datasets import ClassLabel, DatasetDict  # type: ignore

    if not isinstance(ds, DatasetDict):
        raise ValueError("Expected a DatasetDict with train/validation splits.")

    train_split = ds["train"]
    if "label" not in train_split.column_names:
        raise ValueError("Dataset must contain a 'label' column for classification.")

    feature = train_split.features.get("label")
    if isinstance(feature, ClassLabel):
        id2label = {i: name for i, name in enumerate(feature.names)}
        label2id = {name: i for i, name in id2label.items()}
        return ds, id2label, label2id

    labels = sorted({str(v) for v in train_split["label"]})
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {v: k for k, v in label2id.items()}

    def map_label(ex: dict[str, Any]) -> dict[str, int]:
        return {"label": label2id[str(ex["label"])]}

    mapped = DatasetDict({name: split.map(map_label) for name, split in ds.items()})
    return mapped, id2label, label2id


def _ensure_eval_split(ds, max_samples: int):
    train_ds = ds["train"]
    max_train = min(len(train_ds), max_samples)
    train_ds = train_ds.select(range(max_train))

    if "validation" in ds:
        eval_ds = ds["validation"]
        eval_ds = eval_ds.select(range(min(len(eval_ds), max(32, max_samples // 5))))
        return train_ds, eval_ds

    split = train_ds.train_test_split(test_size=0.2, seed=42)
    return split["train"], split["test"]


def _build_readable_label_map(train_ds, id2label: dict[int, str]) -> dict[str, str]:
    mapping = {str(k): str(v) for k, v in id2label.items()}
    if "label_des" not in train_ds.column_names:
        return mapping

    try:
        labels = train_ds["label"]
        label_descs = train_ds["label_des"]
    except Exception:
        return mapping

    for lid, desc in zip(labels, label_descs):
        if desc is None:
            continue
        desc_text = str(desc).strip()
        if not desc_text:
            continue
        mapping[str(int(lid))] = desc_text
    return mapping


def train(
    dataset_name: str,
    model_name: str,
    output_dir: str,
    max_samples: int = 1000,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    dataset_config: str | None = None,
) -> dict[str, Any]:
    try:
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'transformers' (and torch). Install: pip install transformers torch"
        ) from exc

    start = time.time()
    ds = _load_dataset(dataset_name, dataset_config)
    ds, id2label, label2id = _normalize_label_column(ds)
    label_map = _build_readable_label_map(ds["train"], id2label)
    train_ds, eval_ds = _ensure_eval_split(ds, max_samples=max_samples)

    text_col = _pick_text_column(train_ds.column_names)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(batch[text_col], truncation=True, padding="max_length", max_length=256)

    train_tok = train_ds.map(tokenize_fn, batched=True)
    eval_tok = eval_ds.map(tokenize_fn, batched=True)
    keep_cols = {"input_ids", "attention_mask", "label", "token_type_ids"}
    train_tok = train_tok.remove_columns([c for c in train_tok.column_names if c not in keep_cols])
    eval_tok = eval_tok.remove_columns([c for c in eval_tok.column_names if c not in keep_cols])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    raw_args = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "evaluation_strategy": "epoch",
        "save_strategy": "no",
        "logging_strategy": "epoch",
        "report_to": [],
    }
    valid_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    if "evaluation_strategy" not in valid_params and "eval_strategy" in valid_params:
        raw_args["eval_strategy"] = raw_args.pop("evaluation_strategy")
    elif "evaluation_strategy" not in valid_params:
        raw_args.pop("evaluation_strategy")
    args = TrainingArguments(**{k: v for k, v in raw_args.items() if k in valid_params})

    trainer = Trainer(model=model, args=args, train_dataset=train_tok, eval_dataset=eval_tok)
    trainer.train()

    pred = trainer.predict(eval_tok)
    y_true = np.array(eval_tok["label"])
    y_pred = np.argmax(pred.predictions, axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))

    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "label_map.json"), "w", encoding="utf-8") as fp:
        json.dump(label_map, fp, ensure_ascii=False, indent=2)

    return {
        "dataset": f"hf:{dataset_name}",
        "dataset_config": dataset_config,
        "task_type": "text_classification_hf",
        "model_name": model_name,
        "sample_size": int(len(train_tok)),
        "eval_size": int(len(eval_tok)),
        "train_time_sec": round(time.time() - start, 2),
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "model_path": output_dir,
    }
