import inspect
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_MIX_DIR = ROOT_DIR / "training" / "data" / "edu_mix_qa"
COURSE_LOCAL_SOURCE_PREFIX = "course_local"


def _normalize_text(text: str) -> str:
    return "".join(text.strip().lower().split())


def _char_f1(pred: str, truth: str) -> float:
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    pred_chars = list(_normalize_text(pred))
    truth_chars = list(_normalize_text(truth))
    common = Counter(pred_chars) & Counter(truth_chars)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(len(pred_chars), 1)
    recall = overlap / max(len(truth_chars), 1)
    return 2 * precision * recall / max(precision + recall, 1e-8)


def _first_answer(example: dict[str, Any]) -> str:
    answers = example.get("answers")
    if isinstance(answers, dict):
        texts = answers.get("text", [])
        if texts:
            return str(texts[0])
    if isinstance(answers, list) and answers:
        first = answers[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict) and "text" in first:
            return str(first["text"])
    answer = example.get("answer")
    if isinstance(answer, str):
        return answer
    return ""


def _to_qa_fields(example: dict[str, Any]) -> dict[str, Any]:
    question = str(example.get("question", "")).strip()
    context = str(example.get("context", "")).strip()
    answer_text = _first_answer(example).strip()
    answer_start = context.find(answer_text) if answer_text else -1
    if answer_start < 0:
        answer_start = 0
    return {
        "question": question,
        "context": context,
        "answers": {
            "text": [answer_text],
            "answer_start": [answer_start],
        },
        "source": str(example.get("source", "")).strip(),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _course_like_source(source_name: str) -> bool:
    return source_name.lower().startswith(COURSE_LOCAL_SOURCE_PREFIX)


def _filter_stage_records(records: list[dict[str, Any]], train_stage: str | None) -> list[dict[str, Any]]:
    stage = (train_stage or "").strip().upper()
    if stage == "B":
        filtered = [
            rec
            for rec in records
            if _course_like_source(str(rec.get("source", "")).strip())
        ]
        if filtered:
            return filtered
    return records


def _source_mix(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in records:
        source = str(rec.get("source", "")).strip() or "unknown"
        counts[source] = counts.get(source, 0) + 1
    return counts


def _prepare_local_mix_dataset(
    dataset_config: str | None,
    max_samples: int,
    train_stage: str | None,
):
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

    train_records = _read_jsonl(train_path)
    val_records = _read_jsonl(val_path)
    train_records = _filter_stage_records(train_records, train_stage=train_stage)

    if not train_records:
        raise RuntimeError("No usable train samples in local QA mix dataset")
    if not val_records:
        fallback_count = min(64, len(train_records))
        val_records = train_records[:fallback_count]

    train_records = train_records[: min(len(train_records), max_samples)]
    val_records = val_records[: min(len(val_records), max(32, max_samples // 5))]

    in_domain_eval = [
        rec
        for rec in val_records
        if _course_like_source(str(rec.get("source", "")).strip())
    ]
    if not in_domain_eval:
        in_domain_eval = val_records[: min(64, len(val_records))]

    ds = {
        "train": Dataset.from_list(train_records),
        "validation": Dataset.from_list(val_records),
    }
    return ds, f"local_jsonl:{base_dir}", _source_mix(train_records), in_domain_eval


def _prepare_hf_dataset(
    dataset_name: str,
    dataset_config: str | None,
    max_samples: int,
):
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency 'datasets'. Install: pip install datasets") from exc

    if dataset_name == "cmrc2018":
        ds = load_dataset("cmrc2018")
    elif dataset_config:
        ds = load_dataset(dataset_name, dataset_config)
    else:
        ds = load_dataset(dataset_name)

    train_raw = ds["train"].select(range(min(len(ds["train"]), max_samples)))
    val_source = ds["validation"] if "validation" in ds else ds["train"]
    eval_raw = val_source.select(range(min(len(val_source), max(32, max_samples // 5))))
    in_domain_eval = [eval_raw[i] for i in range(min(64, len(eval_raw)))]
    out = {"train": train_raw, "validation": eval_raw}
    return out, f"hf:{dataset_name}", {dataset_name: int(len(train_raw))}, in_domain_eval


def _prepare_dataset(
    dataset_name: str,
    dataset_config: str | None,
    max_samples: int,
    train_stage: str | None,
):
    if dataset_name in {"edu_mix_qa_local", "local_jsonl"}:
        return _prepare_local_mix_dataset(
            dataset_config=dataset_config,
            max_samples=max_samples,
            train_stage=train_stage,
        )
    return _prepare_hf_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        max_samples=max_samples,
    )


def _evaluate_pipeline(qa_pipe, eval_norm, limit: int = 64) -> tuple[float, float, int]:
    eval_count = min(limit, len(eval_norm))
    em_total = 0.0
    f1_total = 0.0
    for idx in range(eval_count):
        ex = eval_norm[idx]
        pred = qa_pipe(question=ex["question"], context=ex["context"])
        pred_text = str(pred.get("answer", ""))
        truth = ex["answers"]["text"][0] if ex["answers"]["text"] else ""
        if _normalize_text(pred_text) == _normalize_text(truth):
            em_total += 1.0
        f1_total += _char_f1(pred_text, truth)

    em = em_total / max(eval_count, 1)
    f1 = f1_total / max(eval_count, 1)
    return em, f1, eval_count


def train(
    dataset_name: str,
    model_name: str,
    output_dir: str,
    max_samples: int = 600,
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 3e-5,
    dataset_config: str | None = None,
    train_stage: str | None = None,
) -> dict[str, Any]:
    try:
        from transformers import (  # type: ignore
            AutoModelForQuestionAnswering,
            AutoTokenizer,
            DefaultDataCollator,
            Trainer,
            TrainingArguments,
            pipeline,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency 'transformers' (and torch). Install: pip install transformers torch"
        ) from exc

    start = time.time()
    ds, dataset_desc, source_mix, in_domain_eval_rows = _prepare_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        max_samples=max_samples,
        train_stage=train_stage,
    )
    train_raw = ds["train"]
    eval_raw = ds["validation"]

    train_norm = train_raw.map(_to_qa_fields)
    eval_norm = eval_raw.map(_to_qa_fields)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(examples: dict[str, list[Any]]) -> dict[str, Any]:
        inputs = tokenizer(
            examples["question"],
            examples["context"],
            max_length=384,
            truncation="only_second",
            padding="max_length",
            return_offsets_mapping=True,
        )

        offsets = inputs.pop("offset_mapping")
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offsets):
            answer = examples["answers"][i]
            answer_starts = answer["answer_start"]
            answer_texts = answer["text"]
            if not answer_starts or not answer_texts:
                start_positions.append(0)
                end_positions.append(0)
                continue

            start_char = answer_starts[0]
            end_char = start_char + len(answer_texts[0])
            sequence_ids = inputs.sequence_ids(i)

            context_start = 0
            while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
                context_start += 1

            context_end = len(sequence_ids) - 1
            while context_end >= 0 and sequence_ids[context_end] != 1:
                context_end -= 1

            if context_start >= len(offset) or context_end < 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
                continue

            token_start = context_start
            while token_start <= context_end and offset[token_start][0] <= start_char:
                token_start += 1
            start_positions.append(max(context_start, token_start - 1))

            token_end = context_end
            while token_end >= context_start and offset[token_end][1] >= end_char:
                token_end -= 1
            end_positions.append(min(context_end, token_end + 1))

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    train_tok = train_norm.map(preprocess, batched=True, remove_columns=train_norm.column_names)
    eval_tok = eval_norm.map(preprocess, batched=True, remove_columns=eval_norm.column_names)

    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    raw_args = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "evaluation_strategy": "no",
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
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_tok,
        "eval_dataset": eval_tok,
        "data_collator": DefaultDataCollator(),
    }
    trainer_params = set(inspect.signature(Trainer.__init__).parameters.keys())
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    qa_pipe = pipeline("question-answering", model=output_dir, tokenizer=output_dir)
    em, f1, eval_count = _evaluate_pipeline(qa_pipe, eval_norm, limit=64)

    in_domain_norm = []
    for row in in_domain_eval_rows:
        if isinstance(row, dict):
            in_domain_norm.append(_to_qa_fields(row))
    if not in_domain_norm:
        in_domain_norm = [eval_norm[i] for i in range(min(32, len(eval_norm)))]
    in_domain_em, in_domain_f1, in_domain_eval_count = _evaluate_pipeline(qa_pipe, in_domain_norm, limit=64)

    stage_label = (train_stage or "A").strip().upper()[:1] if train_stage else "A"
    return {
        "dataset": dataset_desc,
        "dataset_config": dataset_config,
        "task_type": "qa_extractive_hf",
        "model_name": model_name,
        "sample_size": int(len(train_tok)),
        "eval_size": int(eval_count),
        "train_time_sec": round(time.time() - start, 2),
        "em": round(float(em), 4),
        "f1": round(float(f1), 4),
        "in_domain_eval_em": round(float(in_domain_em), 4),
        "in_domain_eval_f1": round(float(in_domain_f1), 4),
        "in_domain_eval_size": int(in_domain_eval_count),
        "source_mix": source_mix,
        "train_stage": stage_label,
        "model_path": output_dir,
    }
