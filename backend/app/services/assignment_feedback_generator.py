from __future__ import annotations

from pathlib import Path
from typing import Any


_GEN_MODEL_CACHE: dict[str, tuple[Any, Any, str]] = {}


def _load_generator(model_path: str):
    try:
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependencies for feedback generator. Install: transformers torch bitsandbytes") from exc

    normalized = str(Path(model_path).expanduser().resolve())
    if normalized in _GEN_MODEL_CACHE:
        return _GEN_MODEL_CACHE[normalized]

    path_obj = Path(normalized)
    adapter_cfg_path = path_obj / "adapter_config.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_4bit = torch.cuda.is_available()
    model = None
    tokenizer = None

    if adapter_cfg_path.exists():
        try:
            from peft import PeftConfig, PeftModel  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Missing peft for LoRA adapter loading. Install: peft") from exc

        peft_cfg = PeftConfig.from_pretrained(normalized)
        base_model_name = str(peft_cfg.base_model_name_or_path or "").strip()
        if not base_model_name:
            raise RuntimeError(f"Invalid adapter config without base model path: {normalized}")

        quant_cfg = None
        if use_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            quantization_config=quant_cfg,
        )
        model = PeftModel.from_pretrained(base_model, normalized)
    else:
        tokenizer = AutoTokenizer.from_pretrained(normalized, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            normalized,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    if device == "cpu":
        model.to("cpu")

    _GEN_MODEL_CACHE[normalized] = (tokenizer, model, device)
    return tokenizer, model, device


def generate_feedback_text(
    prompt: str,
    model_path: str,
    max_new_tokens: int = 220,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str:
    import torch  # type: ignore

    tokenizer, model, device = _load_generator(model_path)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    if device == "cuda":
        encoded = {k: v.to("cuda") for k, v in encoded.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=max(0.05, float(temperature)),
            top_p=min(max(float(top_p), 0.1), 1.0),
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][encoded["input_ids"].shape[1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return (text or "").strip()

