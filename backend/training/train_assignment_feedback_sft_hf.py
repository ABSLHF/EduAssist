import inspect
import json
import math
import os
import time
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_MIX_DIR = ROOT_DIR / "training" / "data" / "assignment_feedback_sft_mix"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _normalize_record(item: dict[str, Any]) -> dict[str, Any] | None:
    instruction = str(item.get("instruction") or "").strip()
    model_input = str(item.get("input") or "").strip()
    output = str(item.get("output") or "").strip()
    tier = str(item.get("tier") or "partial").strip()
    source = str(item.get("source") or "unknown").strip() or "unknown"
    if not instruction or not model_input or not output:
        return None
    prompt = (
        "你是高校课程助教。请基于题目与学生答案生成评语。\n"
        f"{instruction}\n"
        "### 输入\n"
        f"{model_input}\n"
        "### 输出\n"
    )
    return {"prompt": prompt, "output": output, "tier": tier, "source": source}


def _source_mix(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        src = str(row.get("source", "")).strip() or "unknown"
        counts[src] = counts.get(src, 0) + 1
    return counts


def _prepare_local_dataset(dataset_config: str | None, max_samples: int):
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

    train_rows = [r for r in (_normalize_record(x) for x in _read_jsonl(train_path)) if r]
    val_rows = [r for r in (_normalize_record(x) for x in _read_jsonl(val_path)) if r]
    if not train_rows:
        raise RuntimeError("No usable train samples in assignment feedback SFT dataset.")
    if not val_rows:
        val_rows = train_rows[: min(512, len(train_rows))]

    train_rows = train_rows[: min(len(train_rows), max_samples)]
    val_rows = val_rows[: min(len(val_rows), max(64, max_samples // 8))]
    return {"train": Dataset.from_list(train_rows), "validation": Dataset.from_list(val_rows)}, f"local_jsonl:{base_dir}", _source_mix(train_rows)


def train(
    dataset_name: str,
    model_name: str,
    output_dir: str,
    max_samples: int = 16000,
    epochs: int = 2,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    dataset_config: str | None = None,
) -> dict[str, Any]:
    _ = dataset_name
    try:
        import torch  # type: ignore
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependencies for SFT. Install: peft bitsandbytes transformers torch") from exc

    start = time.time()
    ds, dataset_desc, source_mix = _prepare_local_dataset(dataset_config=dataset_config, max_samples=max_samples)
    train_ds = ds["train"]
    eval_ds = ds["validation"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    max_length = 2048

    def tokenize_fn(batch: dict[str, Any]) -> dict[str, Any]:
        prompts = batch["prompt"]
        outputs = batch["output"]
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        for prompt, target in zip(prompts, outputs):
            prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(target, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]
            input_ids = (prompt_ids + target_ids + [tokenizer.eos_token_id])[:max_length]
            labels = ([-100] * len(prompt_ids) + target_ids + [tokenizer.eos_token_id])[:max_length]
            attention_mask = [1] * len(input_ids)

            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len
                attention_mask = attention_mask + [0] * pad_len

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)

        return {"input_ids": all_input_ids, "attention_mask": all_attention_mask, "labels": all_labels}

    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(tokenize_fn, batched=True, remove_columns=eval_ds.column_names)

    raw_args = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": 16,
        "learning_rate": learning_rate,
        "warmup_ratio": 0.03,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 10,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
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
    trainer = Trainer(model=model, args=args, train_dataset=train_tok, eval_dataset=eval_tok, tokenizer=tokenizer)
    trainer.train()

    eval_metrics = trainer.evaluate()
    eval_loss = float(eval_metrics.get("eval_loss", 0.0) or 0.0)
    ppl = math.exp(eval_loss) if eval_loss > 0 else 0.0

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {
        "dataset": dataset_desc,
        "dataset_config": dataset_config,
        "task_type": "assignment_feedback_sft_hf",
        "model_name": model_name,
        "sample_size": int(len(train_tok)),
        "eval_size": int(len(eval_tok)),
        "train_time_sec": round(time.time() - start, 2),
        "eval_loss": round(eval_loss, 4),
        "perplexity": round(float(ppl), 4),
        "source_mix": source_mix,
        "model_path": output_dir,
    }

