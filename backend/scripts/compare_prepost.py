import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pre/post evaluation JSON reports.")
    parser.add_argument("--before", required=True, help="Path to pre report json")
    parser.add_argument("--after", required=True, help="Path to post report json")
    parser.add_argument("--out-dir", default="docs/eval")
    parser.add_argument("--name", default=None, help="Optional output prefix")
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"report not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def get_overall_metrics(payload: dict[str, Any]) -> dict[str, float]:
    summary = payload.get("summary", {})
    rag = summary.get("rag", {})
    qa_predict = summary.get("qa_predict", {})
    return {
        "rag_hit_rate": float(rag.get("hit_rate", 0.0)),
        "rag_avg_score": float(rag.get("avg_score", 0.0)),
        "qa_hit_rate": float(qa_predict.get("hit_rate", 0.0)),
        "qa_avg_score": float(qa_predict.get("avg_score", 0.0)),
        "fallback_ratio": float(summary.get("fallback_ratio", 0.0)),
        "reference_coverage": float(summary.get("reference_coverage", 0.0)),
        "off_topic_rate": float(summary.get("off_topic_rate", 0.0)),
        "insufficient_misfire_rate": float(summary.get("insufficient_misfire_rate", 0.0)),
    }


def get_set_metrics(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for item in payload.get("sets", []):
        course_name = str(item.get("course_name", "")).strip()
        if not course_name:
            continue
        rag = item.get("rag", {})
        qa_predict = item.get("qa_predict", {})
        out[course_name] = {
            "rag_hit_rate": float(rag.get("hit_rate", 0.0)),
            "rag_avg_score": float(rag.get("avg_score", 0.0)),
            "qa_hit_rate": float(qa_predict.get("hit_rate", 0.0)),
            "qa_avg_score": float(qa_predict.get("avg_score", 0.0)),
            "fallback_ratio": float(item.get("fallback_ratio", 0.0)),
            "reference_coverage": float(item.get("reference_coverage", 0.0)),
            "off_topic_rate": float(item.get("off_topic_rate", 0.0)),
            "insufficient_misfire_rate": float(item.get("insufficient_misfire_rate", 0.0)),
        }
    return out


def diff_metrics(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    keys = [
        "rag_hit_rate",
        "rag_avg_score",
        "qa_hit_rate",
        "qa_avg_score",
        "fallback_ratio",
        "reference_coverage",
        "off_topic_rate",
        "insufficient_misfire_rate",
    ]
    return {key: round(after.get(key, 0.0) - before.get(key, 0.0), 4) for key in keys}


def main() -> int:
    args = parse_args()
    before_payload = load_json(args.before)
    after_payload = load_json(args.after)

    before_overall = get_overall_metrics(before_payload)
    after_overall = get_overall_metrics(after_payload)
    overall_delta = diff_metrics(before_overall, after_overall)

    before_sets = get_set_metrics(before_payload)
    after_sets = get_set_metrics(after_payload)
    set_names = sorted(set(before_sets.keys()) | set(after_sets.keys()))
    per_set = []
    for name in set_names:
        b = before_sets.get(name, {})
        a = after_sets.get(name, {})
        per_set.append(
            {
                "course_name": name,
                "before": b,
                "after": a,
                "delta": diff_metrics(b, a),
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.name or f"delta_{stamp}"
    out_json = out_dir / f"{prefix}.json"
    out_md = out_dir / f"{prefix}.md"

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "before": str(Path(args.before)),
        "after": str(Path(args.after)),
        "overall": {
            "before": before_overall,
            "after": after_overall,
            "delta": overall_delta,
        },
        "per_set": per_set,
    }
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 微调前后对比报告",
        "",
        f"- 时间: {result['timestamp']}",
        f"- before: `{result['before']}`",
        f"- after: `{result['after']}`",
        "",
        "## 总体指标变化",
        "",
        "| 指标 | before | after | delta |",
        "|---|---:|---:|---:|",
    ]
    metric_names = [
        ("rag_hit_rate", "RAG命中率"),
        ("rag_avg_score", "RAG平均分"),
        ("qa_hit_rate", "QA抽取命中率"),
        ("qa_avg_score", "QA抽取平均分"),
        ("fallback_ratio", "Fallback占比"),
        ("reference_coverage", "引用覆盖率"),
        ("off_topic_rate", "答非所问率"),
        ("insufficient_misfire_rate", "资料不足误判率"),
    ]
    for key, label in metric_names:
        lines.append(
            f"| {label} | {before_overall.get(key, 0.0):.4f} | "
            f"{after_overall.get(key, 0.0):.4f} | {overall_delta.get(key, 0.0):+.4f} |"
        )

    lines.extend(
        [
            "",
            "## 分课程变化",
            "",
            "| 课程 | RAG命中率Δ | QA命中率Δ | Fallback占比Δ | 引用覆盖率Δ | 答非所问率Δ | 资料不足误判率Δ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in per_set:
        d = item["delta"]
        lines.append(
            f"| {item['course_name']} | {d.get('rag_hit_rate', 0.0):+.4f} | "
            f"{d.get('qa_hit_rate', 0.0):+.4f} | {d.get('fallback_ratio', 0.0):+.4f} | "
            f"{d.get('reference_coverage', 0.0):+.4f} | {d.get('off_topic_rate', 0.0):+.4f} | "
            f"{d.get('insufficient_misfire_rate', 0.0):+.4f} |"
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] JSON delta report: {out_json}")
    print(f"[OK] Markdown delta report: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
