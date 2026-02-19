import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def _load_sample_csv(csv_path: str) -> tuple[list[str], list[str], str]:
    df = pd.read_csv(csv_path)
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must contain columns: label,text")
    x = df["text"].astype(str).tolist()
    y = df["label"].astype(str).tolist()
    return x, y, f"local_csv:{csv_path}"


def _load_hf_dataset(dataset_name: str, max_samples: int) -> tuple[list[str], list[str], str]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError("HuggingFace datasets is not installed. Run: pip install datasets") from exc

    ds = load_dataset(dataset_name, split="train")
    rows = ds.select(range(min(len(ds), max_samples)))
    text_col = None
    for candidate in ("text", "sentence", "content"):
        if candidate in rows.column_names:
            text_col = candidate
            break
    if text_col is None or "label" not in rows.column_names:
        raise ValueError("Dataset must contain text-like column and label column")

    x = [str(v) for v in rows[text_col]]
    y = [str(v) for v in rows["label"]]
    return x, y, f"hf:{dataset_name}"


def train(
    csv_path: str,
    model_path: str,
    dataset_name: str = "sample_local",
    max_samples: int = 1200,
) -> dict:
    if dataset_name == "sample_local":
        x, y, source = _load_sample_csv(csv_path)
    else:
        x, y, source = _load_hf_dataset(dataset_name, max_samples=max_samples)

    label_counts = pd.Series(y).value_counts()
    min_count = int(label_counts.min()) if not label_counts.empty else 0
    can_stratify = len(label_counts) > 1 and min_count >= 2
    stratify_target = y if can_stratify else None

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )
    clf = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("svm", LinearSVC()),
        ]
    )
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_val)
    accuracy = float(accuracy_score(y_val, y_pred))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)

    metrics = {
        "dataset": source,
        "stratified_split": can_stratify,
        "min_class_count": min_count,
        "train_size": len(x_train),
        "val_size": len(x_val),
        "accuracy": round(accuracy, 4),
        "model_path": model_path,
    }
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="training/data/sample_labels.csv")
    parser.add_argument("--out", default="models/keyword_clf.joblib")
    parser.add_argument("--dataset", default="sample_local")
    parser.add_argument("--max-samples", type=int, default=1200)
    parser.add_argument("--metrics-out", default="")
    args = parser.parse_args()

    result = train(
        csv_path=args.csv,
        model_path=args.out,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
    )
    if args.metrics_out:
        Path(args.metrics_out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
