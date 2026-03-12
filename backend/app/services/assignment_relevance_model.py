from __future__ import annotations

from functools import lru_cache


def _resolve_relevant_index(model) -> int:
    label2id = getattr(model.config, "label2id", {}) or {}
    id2label = getattr(model.config, "id2label", {}) or {}
    num_labels = int(getattr(model.config, "num_labels", 2) or 2)

    normalized = {str(k).strip().lower(): int(v) for k, v in label2id.items() if isinstance(v, int)}
    for key in ("relevant", "entailment", "similar", "match", "label_1", "1"):
        if key in normalized:
            return normalized[key]

    for idx, label in id2label.items():
        label_norm = str(label).strip().lower()
        if label_norm in {"relevant", "entailment", "similar", "match", "label_1", "1"}:
            try:
                return int(idx)
            except Exception:
                continue

    if num_labels == 2:
        return 1
    return max(0, num_labels - 1)


@lru_cache(maxsize=2)
def _load_classifier(model_path: str):
    try:
        import torch  # type: ignore
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependencies for assignment relevance model. Install: pip install transformers torch"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    relevant_index = _resolve_relevant_index(model)
    return tokenizer, model, device, relevant_index


def predict_relevance_probability(question: str, answer: str, model_path: str, max_length: int = 384) -> float:
    tokenizer, model, device, relevant_index = _load_classifier(model_path)

    import torch  # type: ignore

    encoded = tokenizer(
        question or "",
        answer or "",
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits

    if logits.ndim == 2 and logits.shape[-1] > 1:
        probs = torch.softmax(logits, dim=-1)[0]
        idx = min(max(int(relevant_index), 0), int(probs.shape[0]) - 1)
        score = float(probs[idx].item())
    else:
        flat = logits.reshape(-1)[0]
        score = float(torch.sigmoid(flat).item())

    return min(max(score, 0.0), 1.0)


@lru_cache(maxsize=1)
def _load_reranker(model_name: str):
    try:
        import torch  # type: ignore
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependencies for reranker scoring. Install: pip install transformers torch") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device


def predict_reranker_probability(question: str, answer: str, model_name: str, max_length: int = 512) -> float:
    tokenizer, model, device = _load_reranker(model_name)

    import torch  # type: ignore

    encoded = tokenizer(
        question or "",
        answer or "",
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits

    if logits.ndim == 2 and logits.shape[-1] > 1:
        probs = torch.softmax(logits, dim=-1)[0]
        score = float(probs[-1].item())
    else:
        score = float(torch.sigmoid(logits.reshape(-1)[0]).item())
    return min(max(score, 0.0), 1.0)
