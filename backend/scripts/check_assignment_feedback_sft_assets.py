from __future__ import annotations

import argparse
import json
from pathlib import Path


def _exists(path_str: str) -> dict:
    path = Path(path_str).expanduser()
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "is_file": path.is_file(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check assignment feedback SFT assets on AutoDL.")
    parser.add_argument(
        "--model-dir",
        default="/root/autodl-tmp/hf_models/Qwen2.5-7B-Instruct",
        help="Local base model directory.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="training/data/assignment_feedback_sft_mix",
        help="SFT dataset directory (relative to backend by default).",
    )
    parser.add_argument(
        "--backend-dir",
        default=".",
        help="Backend project directory.",
    )
    args = parser.parse_args()

    backend_dir = Path(args.backend_dir).expanduser().resolve()
    dataset_dir = (backend_dir / args.dataset_dir).resolve()
    train_script = (backend_dir / "training" / "train_assignment_feedback_sft_hf.py").resolve()
    run_script = (backend_dir / "scripts" / "run_assignment_feedback_sft_autodl.py").resolve()
    manifest = (dataset_dir / "manifest.json").resolve()

    payload = {
        "backend_dir": str(backend_dir),
        "checks": {
            "base_model_dir": _exists(args.model_dir),
            "dataset_dir": _exists(str(dataset_dir)),
            "dataset_manifest": _exists(str(manifest)),
            "train_script": _exists(str(train_script)),
            "run_script": _exists(str(run_script)),
        },
    }

    ok = all(item["exists"] for item in payload["checks"].values())
    payload["ok"] = ok
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

