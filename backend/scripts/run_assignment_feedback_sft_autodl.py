"""AutoDL helper for assignment feedback SFT model training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def build_dataset(args: argparse.Namespace, backend_dir: Path) -> None:
    cmd = [
        sys.executable,
        "training/build_assignment_feedback_sft_mix.py",
        "--out-dir",
        args.out_dir,
        "--from-feedback-mix",
        args.from_feedback_mix,
    ]
    if args.include_scientsbank:
        cmd.append("--include-scientsbank")
    if args.include_beetle:
        cmd.append("--include-beetle")
    if args.local_train:
        cmd.extend(["--local-train", args.local_train])
    if args.local_validation:
        cmd.extend(["--local-validation", args.local_validation])

    run_cmd(cmd, cwd=backend_dir)
    manifest_path = backend_dir / args.out_dir / "manifest.json"
    if manifest_path.exists():
        print(f"\n[OK] Dataset manifest: {manifest_path}")
        print(manifest_path.read_text(encoding="utf-8"))


def login(base_url: str, username: str, password: str, timeout: float) -> str:
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            f"{base_url.rstrip('/')}/auth/login",
            json={"username": username, "password": password},
        )
        resp.raise_for_status()
        payload = resp.json()
    token = payload.get("access_token")
    if not token:
        raise RuntimeError(f"Login succeeded but no access_token in response: {payload}")
    return token


def trigger_training(base_url: str, token: str, args: argparse.Namespace, timeout: float) -> int:
    payload = {
        "task_type": "assignment_feedback_sft_hf",
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_samples": args.max_samples,
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            f"{base_url.rstrip('/')}/model/train",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    job_id = data.get("job_id")
    if not isinstance(job_id, int):
        raise RuntimeError(f"Unexpected train response: {data}")
    print(f"\n[OK] Training job submitted: job_id={job_id}, status={data.get('status')}")
    return job_id


def poll_job(
    base_url: str,
    token: str,
    job_id: int,
    timeout: float,
    poll_seconds: int,
    max_wait_seconds: int,
) -> dict:
    begin = time.time()
    last = None
    with httpx.Client(timeout=timeout) as client:
        while True:
            resp = client.get(
                f"{base_url.rstrip('/')}/model/train/{job_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
            data = resp.json()
            status = str(data.get("status", "unknown"))
            if status != last:
                print(f"[INFO] job_id={job_id}, status={status}")
                last = status
            if status in {"success", "failed", "error", "canceled", "cancelled"}:
                return data
            if time.time() - begin > max_wait_seconds:
                raise TimeoutError(f"Job polling timeout after {max_wait_seconds}s")
            time.sleep(max(1, poll_seconds))


def query_active(base_url: str, token: str, timeout: float) -> dict:
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(
            f"{base_url.rstrip('/')}/model/active",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return resp.json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoDL one-shot helper for assignment feedback SFT training.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--http-timeout", type=float, default=30.0)

    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--out-dir", default="training/data/assignment_feedback_sft_mix")
    parser.add_argument("--from-feedback-mix", default="training/data/assignment_feedback_mix")
    parser.add_argument("--include-scientsbank", action="store_true")
    parser.add_argument("--include-beetle", action="store_true")
    parser.add_argument("--local-train")
    parser.add_argument("--local-validation")

    parser.add_argument("--dataset-name", default="assignment_feedback_sft_mix")
    parser.add_argument("--dataset-config", default="training/data/assignment_feedback_sft_mix")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-samples", type=int, default=16000)

    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=8 * 3600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend_dir = Path(__file__).resolve().parents[1]
    try:
        if not args.skip_build:
            build_dataset(args, backend_dir)
        token = login(args.base_url, args.username, args.password, args.http_timeout)
        print("[OK] Login succeeded.")
        job_id = trigger_training(args.base_url, token, args, args.http_timeout)
        final = poll_job(
            args.base_url,
            token,
            job_id,
            args.http_timeout,
            args.poll_seconds,
            args.max_wait_seconds,
        )
        print("\n[RESULT] job:")
        print(json.dumps(final, ensure_ascii=False, indent=2))
        active = query_active(args.base_url, token, args.http_timeout)
        print("\n[RESULT] active model:")
        print(
            json.dumps(
                {
                    "assignment_feedback_sft_model_path": active.get("assignment_feedback_sft_model_path"),
                    "assignment_feedback_sft_model_source": active.get("assignment_feedback_sft_model_source"),
                    "enable_assignment_feedback_sft_model": active.get("enable_assignment_feedback_sft_model"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if str(final.get("status")) == "success" else 1
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

