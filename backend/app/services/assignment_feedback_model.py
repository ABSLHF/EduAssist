from __future__ import annotations

from pathlib import Path
from typing import Any


_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def _normalize_quality_label(raw_label: str, idx: int, num_labels: int) -> str:
    text = (raw_label or "").strip().lower()
    if any(token in text for token in ("off_topic", "weak", "bad", "incorrect")):
        return "weak"
    if any(token in text for token in ("partial", "ok", "neutral", "medium")):
        return "ok"
    if any(token in text for token in ("relevant", "good", "correct", "strong")):
        return "good"

    if num_labels >= 3:
        if idx <= 0:
            return "weak"
        if idx == 1:
            return "ok"
        return "good"

    return "good" if idx > 0 else "weak"


def _load_model(model_path: str):
    try:
        import torch  # type: ignore
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependencies for assignment feedback model. Install: pip install transformers torch") from exc

    normalized_path = str(Path(model_path).expanduser().resolve())
    if normalized_path in _MODEL_CACHE:
        return _MODEL_CACHE[normalized_path]

    tokenizer = AutoTokenizer.from_pretrained(normalized_path)
    model = AutoModelForSequenceClassification.from_pretrained(normalized_path)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    _MODEL_CACHE[normalized_path] = (tokenizer, model)
    return tokenizer, model


def predict_feedback_quality(
    question: str,
    reference_answer: str,
    student_answer: str,
    model_path: str,
    max_length: int = 384,
) -> tuple[str, float]:
    import torch  # type: ignore

    tokenizer, model = _load_model(model_path)
    text_a = f"题目：{question}\n参考答案：{reference_answer or '无'}"
    text_b = f"学生答案：{student_answer}"
    encoded = tokenizer(
        text_a,
        text_b,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    if torch.cuda.is_available():
        encoded = {k: v.to("cuda") for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits

    num_labels = int(getattr(model.config, "num_labels", 2) or 2)
    if logits.shape[-1] == 1:
        score = float(torch.sigmoid(logits.reshape(-1)[0]).item())
        if score >= 0.66:
            return "good", score
        if score >= 0.33:
            return "ok", score
        return "weak", score

    probs = torch.softmax(logits.reshape(1, -1), dim=-1)[0]
    idx = int(torch.argmax(probs).item())
    confidence = float(probs[idx].item())
    id2label = getattr(model.config, "id2label", {}) or {}
    raw_label = str(id2label.get(idx, f"LABEL_{idx}"))
    quality = _normalize_quality_label(raw_label, idx=idx, num_labels=num_labels)
    return quality, min(max(confidence, 0.0), 1.0)
