import os
import time
import inspect
from collections import Counter
from typing import Any


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
    }


def _prepare_dataset(dataset_name: str, dataset_config: str | None):
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
    return ds


def train(
    dataset_name: str,
    model_name: str,
    output_dir: str,
    max_samples: int = 600,
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 3e-5,
    dataset_config: str | None = None,
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
    ds = _prepare_dataset(dataset_name=dataset_name, dataset_config=dataset_config)
    train_raw = ds["train"].select(range(min(len(ds["train"]), max_samples)))
    val_source = ds["validation"] if "validation" in ds else ds["train"]
    eval_raw = val_source.select(range(min(len(val_source), max(32, max_samples // 5))))

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
    eval_count = min(64, len(eval_norm))
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

    return {
        "dataset": f"hf:{dataset_name}",
        "dataset_config": dataset_config,
        "task_type": "qa_extractive_hf",
        "model_name": model_name,
        "sample_size": int(len(train_tok)),
        "eval_size": int(eval_count),
        "train_time_sec": round(time.time() - start, 2),
        "em": round(float(em), 4),
        "f1": round(float(f1), 4),
        "model_path": output_dir,
    }
