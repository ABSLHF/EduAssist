import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any


def _normalize_binary_label(raw_label: Any) -> int | None:
    if raw_label is None:
        return None
    text = str(raw_label).strip().lower()
    if text in {"1", "true", "yes", "relevant", "entailment", "similar", "match"}:
        return 1
    if text in {"0", "false", "no", "off_topic", "not_relevant", "contradiction", "neutral", "mismatch"}:
        return 0
    try:
        value = int(float(text))
        return 1 if value > 0 else 0
    except Exception:
        return None


def _normalize_pair_record(item: dict[str, Any], source_name: str) -> dict[str, Any] | None:
    question = str(item.get("question") or item.get("prompt") or item.get("title") or "").strip()
    answer = str(item.get("answer") or item.get("content") or item.get("response") or "").strip()
    label = _normalize_binary_label(item.get("label"))
    if not question or not answer or label is None:
        return None
    return {"question": question, "answer": answer, "label": int(label), "source": source_name}


def _read_jsonl(path: Path, source_name: str) -> list[dict[str, Any]]:
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
        if not isinstance(payload, dict):
            continue
        normalized = _normalize_pair_record(payload, source_name=source_name)
        if normalized:
            records.append(normalized)
    return records


def _load_ocnli(max_train: int, max_val: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("clue", "ocnli")

    def _map_split(split) -> list[dict[str, Any]]:
        rows = []
        for item in split:
            s1 = str(item.get("sentence1", "")).strip()
            s2 = str(item.get("sentence2", "")).strip()
            if not s1 or not s2:
                continue
            label_text = str(item.get("label_des", "")).strip().lower()
            if label_text:
                label = 1 if label_text in {"entailment", "蕴含"} else 0
            else:
                label = _normalize_binary_label(item.get("label"))
                if label is None:
                    continue
            rows.append({"question": s1, "answer": s2, "label": int(label), "source": "ocnli"})
        return rows

    train_rows = _map_split(ds["train"])[:max_train]
    val_source = ds["validation"] if "validation" in ds else ds["train"]
    val_rows = _map_split(val_source)[:max_val]
    return train_rows, val_rows


def _load_lcqmc(max_train: int, max_val: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("lcqmc")

    def _map_split(split) -> list[dict[str, Any]]:
        rows = []
        for item in split:
            q1 = str(item.get("sentence1", "")).strip()
            q2 = str(item.get("sentence2", "")).strip()
            label = _normalize_binary_label(item.get("label"))
            if label is None or not q1 or not q2:
                continue
            rows.append({"question": q1, "answer": q2, "label": int(label), "source": "lcqmc"})
        return rows

    train_rows = _map_split(ds["train"])[:max_train]
    val_source = ds["validation"] if "validation" in ds else ds["train"]
    val_rows = _map_split(val_source)[:max_val]
    return train_rows, val_rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]):
    lines = [json.dumps(item, ensure_ascii=False) for item in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _source_stats(rows: list[dict[str, Any]]) -> dict[str, int]:
    stats: dict[str, int] = {}
    for item in rows:
        source = str(item.get("source", "")).strip() or "unknown"
        stats[source] = stats.get(source, 0) + 1
    return stats


def main():
    parser = argparse.ArgumentParser(description="Build assignment relevance train/validation jsonl mix.")
    parser.add_argument("--out-dir", default="training/data/assignment_relevance_mix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--include-ocnli", action="store_true")
    parser.add_argument("--include-lcqmc", action="store_true")
    parser.add_argument("--ocnli-max-train", type=int, default=30000)
    parser.add_argument("--ocnli-max-val", type=int, default=3000)
    parser.add_argument("--lcqmc-max-train", type=int, default=30000)
    parser.add_argument("--lcqmc-max-val", type=int, default=3000)
    parser.add_argument("--local-train", default="")
    parser.add_argument("--local-validation", default="")
    parser.add_argument("--sas-train", default="")
    parser.add_argument("--sas-validation", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    source_counts: dict[str, dict[str, int]] = {}

    if args.include_ocnli:
        ocnli_train, ocnli_val = _load_ocnli(max_train=args.ocnli_max_train, max_val=args.ocnli_max_val)
        train_rows.extend(ocnli_train)
        val_rows.extend(ocnli_val)
        source_counts["ocnli"] = {"train": len(ocnli_train), "validation": len(ocnli_val)}

    if args.include_lcqmc:
        lcqmc_train, lcqmc_val = _load_lcqmc(max_train=args.lcqmc_max_train, max_val=args.lcqmc_max_val)
        train_rows.extend(lcqmc_train)
        val_rows.extend(lcqmc_val)
        source_counts["lcqmc"] = {"train": len(lcqmc_train), "validation": len(lcqmc_val)}

    local_train_path = Path(args.local_train).expanduser() if args.local_train else None
    local_val_path = Path(args.local_validation).expanduser() if args.local_validation else None
    if local_train_path and local_train_path.exists():
        local_train_rows = _read_jsonl(local_train_path, source_name="course_local")
        train_rows.extend(local_train_rows)
        source_counts["course_local_train"] = {"train": len(local_train_rows), "validation": 0}
    if local_val_path and local_val_path.exists():
        local_val_rows = _read_jsonl(local_val_path, source_name="course_local")
        val_rows.extend(local_val_rows)
        source_counts["course_local_validation"] = {"train": 0, "validation": len(local_val_rows)}

    sas_train_path = Path(args.sas_train).expanduser() if args.sas_train else None
    sas_val_path = Path(args.sas_validation).expanduser() if args.sas_validation else None
    if sas_train_path and sas_train_path.exists():
        sas_train_rows = _read_jsonl(sas_train_path, source_name="sas_bench")
        train_rows.extend(sas_train_rows)
        source_counts["sas_bench_train"] = {"train": len(sas_train_rows), "validation": 0}
    if sas_val_path and sas_val_path.exists():
        sas_val_rows = _read_jsonl(sas_val_path, source_name="sas_bench")
        val_rows.extend(sas_val_rows)
        source_counts["sas_bench_validation"] = {"train": 0, "validation": len(sas_val_rows)}

    random.shuffle(train_rows)
    random.shuffle(val_rows)
    if not val_rows and train_rows:
        split_idx = max(1, int(len(train_rows) * (1.0 - args.val_ratio)))
        val_rows = train_rows[split_idx:]
        train_rows = train_rows[:split_idx]

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "validation.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": "assignment_relevance_mix_local",
        "train_path": str(train_path.resolve()),
        "validation_path": str(val_path.resolve()),
        "totals": {
            "train": len(train_rows),
            "validation": len(val_rows),
            "all": len(train_rows) + len(val_rows),
        },
        "sources": source_counts,
        "source_mix_train": _source_stats(train_rows),
        "source_mix_validation": _source_stats(val_rows),
        "label_distribution_train": {
            "relevant": sum(1 for x in train_rows if int(x["label"]) == 1),
            "off_topic": sum(1 for x in train_rows if int(x["label"]) == 0),
        },
        "label_distribution_validation": {
            "relevant": sum(1 for x in val_rows if int(x["label"]) == 1),
            "off_topic": sum(1 for x in val_rows if int(x["label"]) == 0),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "out_dir": str(out_dir.resolve()),
                "train": len(train_rows),
                "validation": len(val_rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
