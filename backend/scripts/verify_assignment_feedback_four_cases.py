from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Any

import httpx


MARKDOWN_NOISE_PATTERNS = [
    re.compile(r"^\s*#{1,6}\s*", re.MULTILINE),
    re.compile(r"\*\*"),
    re.compile(r"`{1,3}"),
]


@dataclass
class Case:
    name: str
    content: str
    should_contain_advantage: bool


CASES = [
    Case(name="invalid", content="啊啊啊？？？123###", should_contain_advantage=False),
    Case(name="off_topic", content="线程是 CPU 调度的最小单位，与进程不同。", should_contain_advantage=False),
    Case(name="partial", content="进程是程序的一次执行过程，是系统进行资源分配和调度的单位。", should_contain_advantage=True),
    Case(
        name="good",
        content="进程是程序在操作系统中的一次执行实例，拥有独立地址空间和资源，是资源分配与调度的基本单位。",
        should_contain_advantage=True,
    ),
]


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


def submit(base_url: str, token: str, assignment_id: int, content: str, timeout: float) -> dict[str, Any]:
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            f"{base_url.rstrip('/')}/assignments/{assignment_id}/submit",
            headers={"Authorization": f"Bearer {token}"},
            json={"content": content},
        )
        resp.raise_for_status()
        return resp.json()


def has_markdown_noise(text: str) -> bool:
    for pattern in MARKDOWN_NOISE_PATTERNS:
        if pattern.search(text or ""):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify assignment feedback with four canonical cases.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--assignment-id", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    token = login(args.base_url, args.username, args.password, args.timeout)
    results: list[dict[str, Any]] = []
    all_ok = True

    for case in CASES:
        row = submit(args.base_url, token, args.assignment_id, case.content, args.timeout)
        feedback = str(row.get("feedback") or "")
        score = row.get("score")

        checks = {
            "feedback_non_empty": bool(feedback.strip()),
            "score_is_null": score is None,
            "no_markdown_noise": not has_markdown_noise(feedback),
        }
        if case.should_contain_advantage:
            checks["contains_advantage_or_positive"] = ("优点" in feedback) or ("较好" in feedback) or ("清晰" in feedback)
        else:
            checks["invalid_offtopic_without_advantage"] = "优点" not in feedback

        case_ok = all(checks.values())
        all_ok = all_ok and case_ok
        results.append(
            {
                "case": case.name,
                "request_content": case.content,
                "response_id": row.get("id"),
                "feedback": feedback,
                "score": score,
                "checks": checks,
                "ok": case_ok,
            }
        )

    report = {"ok": all_ok, "assignment_id": args.assignment_id, "results": results}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        print(json.dumps({"ok": False, "error": f"http_status_error: {detail}"}, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(2)

