import os
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

DEFAULT_MODEL_PATH = "models/keyword_clf.joblib"


def _is_hf_model_dir(path: str) -> bool:
    p = Path(path)
    return p.is_dir() and (p / "config.json").exists()


@lru_cache(maxsize=2)
def _load_hf_pipeline(path: str):
    try:
        from transformers import pipeline  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency 'transformers'. Install: pip install transformers torch") from exc
    return pipeline("text-classification", model=path, tokenizer=path)


def _predict_hf(text: str, path: str) -> str:
    clf = _load_hf_pipeline(path)
    result = clf(text, truncation=True)
    if isinstance(result, list) and result:
        label = result[0].get("label", "")
        return str(label)
    return ""


def _predict_joblib(text: str, path: str) -> str:
    if not os.path.exists(path):
        return ""
    model: Any = joblib.load(path)
    pred = model.predict([text])
    return str(pred[0]) if len(pred) else ""


def _load_hf_label_map(path: str) -> dict[str, str]:
    model_dir = Path(path)
    mapping: dict[str, str] = {}

    map_file = model_dir / "label_map.json"
    if map_file.exists():
        try:
            raw = json.loads(map_file.read_text(encoding="utf-8"))
            mapping.update({str(k): str(v) for k, v in raw.items()})
        except Exception:
            pass

    cfg_file = model_dir / "config.json"
    if cfg_file.exists():
        try:
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            id2label = cfg.get("id2label")
            if isinstance(id2label, dict):
                mapping.update({str(k): str(v) for k, v in id2label.items()})
        except Exception:
            pass
    return mapping


def _normalize_label(raw_label: str, path: str) -> str:
    label = raw_label.strip()
    if not label:
        return label

    mapping = _load_hf_label_map(path)
    if label in mapping:
        mapped = mapping[label].strip()
        if re.fullmatch(r"\d+", mapped):
            return f"label_{mapped}"
        return mapped

    match = re.fullmatch(r"(?:LABEL_)?(\d+)", label, flags=re.IGNORECASE)
    if match:
        idx = match.group(1)
        if idx in mapping:
            mapped = mapping[idx].strip()
            if re.fullmatch(r"\d+", mapped):
                return f"label_{mapped}"
            return mapped
        return f"label_{idx}"
    return label


def predict_label(text: str, path: str = DEFAULT_MODEL_PATH) -> str | None:
    if not path:
        return None

    if _is_hf_model_dir(path):
        label = _normalize_label(_predict_hf(text, path), path)
        return label or None

    label = _predict_joblib(text, path)
    return label or None
