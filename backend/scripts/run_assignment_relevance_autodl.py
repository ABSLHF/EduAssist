"""AutoDL helper for assignment relevance model training.

This script can:
1) Build mixed dataset for assignment relevance.
2) Login to backend as teacher.
3) Trigger /model/train with task_type=assignment_relevance_hf.
4) Poll training status until completion.
"""

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
        "training/build_assignment_relevance_mix.py",
        "--out-dir",
        args.out_dir,
    ]
    if args.include_ocnli:
        cmd.append("--include-ocnli")
    if args.include_lcqmc:
        cmd.append("--include-lcqmc")
    if args.local_train:
        cmd.extend(["--local-train", args.local_train])
    if args.local_validation:
        cmd.extend(["--local-validation", args.local_validation])
    if args.sas_train:
        cmd.extend(["--sas-train", args.sas_train])
    if args.sas_validation:
        cmd.extend(["--sas-validation", args.sas_validation])

    run_cmd(cmd, cwd=backend_dir)

    manifest_path = backend_dir / args.out_dir / "manifest.json"
    if manifest_path.exists():
        print(f"\n[OK] Dataset manifest: {manifest_path}")
        print(manifest_path.read_text(encoding="utf-8"))
    else:
        print(f"\n[WARN] Manifest not found: {manifest_path}")


def login(base_url: str, username: str, password: str, timeout: float) -> str:
    url = f"{base_url.rstrip('/')}/auth/login"
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json={"username": username, "password": password})
        resp.raise_for_status()
        payload = resp.json()
    token = payload.get("access_token")
    if not token:
        raise RuntimeError(f"Login succeeded but no access_token in response: {payload}")
    return token


def trigger_training(base_url: str, token: str, args: argparse.Namespace, timeout: float) -> int:
    url = f"{base_url.rstrip('/')}/model/train"
    payload = {
        "task_type": "assignment_relevance_hf",
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_samples": args.max_samples,
    }
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    job_id = data.get("job_id")
    if not isinstance(job_id, int):
        raise RuntimeError(f"Unexpected train response: {data}")
    print(f"\n[OK] Training job submitted: job_id={job_id}, status={data.get('status')}")
    return job_id


def poll_training(
    base_url: str,
    token: str,
    job_id: int,
    timeout: float,
    poll_seconds: int,
    max_wait_seconds: int,
) -> dict:
    url = f"{base_url.rstrip('/')}/model/train/{job_id}"
    headers = {"Authorization": f"Bearer {token}"}
    begin = time.time()
    last_status = None
    with httpx.Client(timeout=timeout) as client:
        while True:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            status = str(data.get("status", "unknown"))
            if status != last_status:
                print(f"[INFO] job_id={job_id}, status={status}")
                last_status = status
            if status in {"success", "failed", "error", "canceled", "cancelled"}:
                return data
            if time.time() - begin > max_wait_seconds:
                raise TimeoutError(f"Job polling timeout after {max_wait_seconds}s")
            time.sleep(max(1, poll_seconds))


def query_active(base_url: str, token: str, timeout: float) -> dict:
    url = f"{base_url.rstrip('/')}/model/active"
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoDL one-shot helper for assignment relevance training.")

    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Backend API base URL.")
    parser.add_argument("--username", required=True, help="Teacher username.")
    parser.add_argument("--password", required=True, help="Teacher password.")
    parser.add_argument("--http-timeout", type=float, default=30.0, help="HTTP timeout in seconds.")

    parser.add_argument("--skip-build", action="store_true", help="Skip dataset build step.")
    parser.add_argument("--out-dir", default="training/data/assignment_relevance_mix", help="Mixed dataset output dir.")
    parser.add_argument("--include-ocnli", action="store_true", help="Include OCNLI source.")
    parser.add_argument("--include-lcqmc", action="store_true", help="Include LCQMC source.")
    parser.add_argument("--local-train", help="Local train jsonl path.")
    parser.add_argument("--local-validation", help="Local validation jsonl path.")
    parser.add_argument("--sas-train", help="SAS train jsonl path.")
    parser.add_argument("--sas-validation", help="SAS validation jsonl path.")

    parser.add_argument("--dataset-name", default="assignment_relevance_mix_local")
    parser.add_argument("--dataset-config", default="training/data/assignment_relevance_mix")
    parser.add_argument("--model-name", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-samples", type=int, default=24000)

    parser.add_argument("--poll-seconds", type=int, default=20, help="Polling interval in seconds.")
    parser.add_argument("--max-wait-seconds", type=int, default=6 * 3600, help="Max wait before timeout.")
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
        final_state = poll_training(
            args.base_url,
            token,
            job_id,
            args.http_timeout,
            args.poll_seconds,
            args.max_wait_seconds,
        )
        print("\n[RESULT] job:")
        print(json.dumps(final_state, ensure_ascii=False, indent=2))

        active = query_active(args.base_url, token, args.http_timeout)
        print("\n[RESULT] active model:")
        print(
            json.dumps(
                {
                    "assignment_relevance_model_path": active.get("assignment_relevance_model_path"),
                    "assignment_relevance_model_source": active.get("assignment_relevance_model_source"),
                    "assignment_relevance_threshold_hi": active.get("assignment_relevance_threshold_hi"),
                    "assignment_relevance_threshold_lo": active.get("assignment_relevance_threshold_lo"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if str(final_state.get("status")) == "success" else 1
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
