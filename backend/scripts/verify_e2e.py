import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


@dataclass
class CaseResult:
    case_id: str
    name: str
    endpoint: str
    method: str
    status: str  # pass | warn | fail
    severity: str  # info | warn | blocker
    request: dict[str, Any] | None
    status_code: int | None
    response_body: Any
    exception_log: str | None = None


class Verifier:
    def __init__(self, base_url: str, timeout_sec: int, poll_timeout_sec: int, trust_env: bool):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.poll_timeout_sec = poll_timeout_sec
        self.results: list[CaseResult] = []
        # trust_env=False avoids system proxy interception for local 127.0.0.1 calls.
        self.client = httpx.Client(timeout=self.timeout_sec, trust_env=trust_env)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.teacher = {
            "username": f"verify_teacher_{stamp}",
            "password": "123456",
            "real_name": "验收教师",
            "role": 1,
        }
        self.student = {
            "username": f"verify_student_{stamp}",
            "password": "123456",
            "real_name": "验收学生",
            "role": 0,
        }
        self.teacher_token = ""
        self.student_token = ""
        self.course_id = 0
        self.material_id = 0
        self.assignment_id = 0
        self.submission_id = 0
        self.cls_job_id = 0
        self.qa_job_id = 0
        self.cls_run: dict[str, Any] = {}
        self.qa_run: dict[str, Any] = {}

    def _headers(self, token: str | None = None) -> dict[str, str]:
        headers = {"accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _record(
        self,
        case_id: str,
        name: str,
        endpoint: str,
        method: str,
        ok: bool,
        request: dict[str, Any] | None,
        status_code: int | None,
        response_body: Any,
        severity_if_fail: str = "blocker",
        exception_log: str | None = None,
        warn: bool = False,
    ) -> None:
        if ok and not warn:
            status = "pass"
            severity = "info"
        elif ok and warn:
            status = "warn"
            severity = "warn"
        else:
            status = "fail"
            severity = severity_if_fail
        self.results.append(
            CaseResult(
                case_id=case_id,
                name=name,
                endpoint=endpoint,
                method=method,
                status=status,
                severity=severity,
                request=request,
                status_code=status_code,
                response_body=response_body,
                exception_log=exception_log,
            )
        )

    def _post_json(self, path: str, payload: dict[str, Any], token: str | None = None) -> httpx.Response:
        return self.client.post(
            f"{self.base_url}{path}",
            headers={**self._headers(token), "Content-Type": "application/json"},
            json=payload,
        )

    def _get(self, path: str, token: str | None = None) -> httpx.Response:
        return self.client.get(f"{self.base_url}{path}", headers=self._headers(token))

    def run(self) -> int:
        try:
            self.verify_auth()
            self.verify_course_and_material()
            self.verify_qa()
            self.verify_assignment_flow()
            self.verify_recommendation_and_kg()
            self.verify_hf_cls()
            self.verify_hf_qa()
        except Exception as exc:  # guardrail, keep report generated
            self._record(
                case_id="X-0",
                name="执行异常",
                endpoint="internal",
                method="SCRIPT",
                ok=False,
                request=None,
                status_code=None,
                response_body={},
                exception_log=str(exc),
            )

        return self.write_reports()

    def verify_auth(self) -> None:
        # teacher register
        res = self._post_json("/auth/register", self.teacher)
        body = self._safe_json(res)
        self._record(
            "A-1",
            "注册教师",
            "/auth/register",
            "POST",
            res.status_code == 200 and body.get("username") == self.teacher["username"],
            self.teacher,
            res.status_code,
            body,
        )

        # student register
        res = self._post_json("/auth/register", self.student)
        body = self._safe_json(res)
        self._record(
            "A-2",
            "注册学生",
            "/auth/register",
            "POST",
            res.status_code == 200 and body.get("username") == self.student["username"],
            self.student,
            res.status_code,
            body,
        )

        # teacher token
        res = self.client.post(
            f"{self.base_url}/auth/token",
            data={"username": self.teacher["username"], "password": self.teacher["password"]},
            headers={"accept": "application/json"},
        )
        body = self._safe_json(res)
        ok = res.status_code == 200 and "access_token" in body
        if ok:
            self.teacher_token = body["access_token"]
        self._record("A-3", "教师令牌", "/auth/token", "POST", ok, {"username": self.teacher["username"]}, res.status_code, body)

        # student token
        res = self.client.post(
            f"{self.base_url}/auth/token",
            data={"username": self.student["username"], "password": self.student["password"]},
            headers={"accept": "application/json"},
        )
        body = self._safe_json(res)
        ok = res.status_code == 200 and "access_token" in body
        if ok:
            self.student_token = body["access_token"]
        self._record("A-4", "学生令牌", "/auth/token", "POST", ok, {"username": self.student["username"]}, res.status_code, body)

    def verify_course_and_material(self) -> None:
        payload = {"name": f"自动验收课程_{int(time.time())}", "description": "自动化验收创建"}
        res = self._post_json("/courses/", payload, token=self.teacher_token)
        body = self._safe_json(res)
        ok = res.status_code == 200 and "id" in body
        if ok:
            self.course_id = int(body["id"])
        self._record("B-1", "教师创建课程", "/courses/", "POST", ok, payload, res.status_code, body)

        material_text = (
            "数据结构是计算机存储、组织数据的方式。\n"
            "顺序存储使用连续内存，链式存储使用指针连接结点。\n"
            "栈是后进先出，队列是先进先出。"
        )
        docs_dir = Path(__file__).resolve().parents[1] / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        test_file = docs_dir / "verify_material.txt"
        test_file.write_text(material_text, encoding="utf-8")

        with test_file.open("rb") as fp:
            files = {"file": (test_file.name, fp, "text/plain")}
            params = {"course_id": str(self.course_id), "title": "自动验收资料"}
            res = self.client.post(
                f"{self.base_url}/materials/",
                headers=self._headers(self.teacher_token),
                params=params,
                files=files,
            )
        body = self._safe_json(res)
        ok = (
            res.status_code == 200
            and body.get("parse_status") == "success"
            and int(body.get("extracted_chars", 0)) > 0
            and "material_id" in body
        )
        if ok:
            self.material_id = int(body["material_id"])
        self._record("B-2", "上传资料并解析", "/materials/", "POST", ok, {"course_id": self.course_id}, res.status_code, body)

    def verify_qa(self) -> None:
        join_res = self._post_json(f"/courses/{self.course_id}/join", {}, token=self.student_token)
        self._record(
            "C-1",
            "学生加入课程",
            f"/courses/{self.course_id}/join",
            "POST",
            join_res.status_code == 200,
            {"course_id": self.course_id},
            join_res.status_code,
            self._safe_json(join_res),
        )

        payload = {"course_id": self.course_id, "question": "什么是数据结构"}
        res = self._post_json("/qa/", payload, token=self.student_token)
        body = self._safe_json(res)
        ok = (
            res.status_code == 200
            and all(k in body for k in ["answer", "source_type", "mode", "references"])
            and isinstance(body.get("references"), list)
        )
        self._record("C-2", "学生问答", "/qa/", "POST", ok, payload, res.status_code, body)

    def verify_assignment_flow(self) -> None:
        payload = {
            "course_id": self.course_id,
            "title": "自动验收作业",
            "description": "说明顺序存储与链式存储",
            "type": "text",
            "keywords": ["顺序存储", "链式存储"],
        }
        res = self._post_json("/assignments/", payload, token=self.teacher_token)
        body = self._safe_json(res)
        ok = res.status_code == 200 and "id" in body
        if ok:
            self.assignment_id = int(body["id"])
        self._record("D-1", "教师发布作业", "/assignments/", "POST", ok, payload, res.status_code, body)

        sub_payload = {"content": "顺序存储使用连续空间，链式存储通过指针连接。", "code": None}
        res = self.client.post(
            f"{self.base_url}/submissions/",
            headers={**self._headers(self.student_token), "Content-Type": "application/json"},
            params={"assignment_id": self.assignment_id},
            json=sub_payload,
        )
        body = self._safe_json(res)
        ok = res.status_code == 200 and "id" in body
        if ok:
            self.submission_id = int(body["id"])
        self._record("D-2", "学生提交作业", "/submissions/", "POST", ok, sub_payload, res.status_code, body)

        res = self._get(f"/submissions/{self.submission_id}/feedback", token=self.student_token)
        body = self._safe_json(res)
        ok = res.status_code == 200 and "feedback" in body and "score" in body
        self._record("D-3", "获取作业反馈", f"/submissions/{self.submission_id}/feedback", "GET", ok, None, res.status_code, body)

    def verify_recommendation_and_kg(self) -> None:
        res = self._get(f"/recommendations/{self.course_id}", token=self.student_token)
        body = self._safe_json(res)
        ok = res.status_code == 200 and "items" in body
        warn = ok and len(body.get("items", [])) == 0
        self._record(
            "E-1",
            "学习推荐",
            f"/recommendations/{self.course_id}",
            "GET",
            ok,
            None,
            res.status_code,
            body,
            warn=warn,
            severity_if_fail="blocker",
        )

        res = self.client.post(
            f"{self.base_url}/kg/{self.course_id}/candidates",
            headers=self._headers(self.teacher_token),
            params={"material_id": self.material_id, "top_k": 5, "auto_create": True},
        )
        body = self._safe_json(res)
        ok = res.status_code == 200 and "candidates" in body
        self._record("E-2", f"/kg/{self.course_id}/candidates", f"/kg/{self.course_id}/candidates", "POST", ok, None, res.status_code, body)

        res = self._get(f"/kg/{self.course_id}", token=self.teacher_token)
        body = self._safe_json(res)
        ok = res.status_code == 200 and "nodes" in body and "edges" in body
        self._record("E-3", "获取知识图谱", f"/kg/{self.course_id}", "GET", ok, None, res.status_code, body)

    def verify_hf_cls(self) -> None:
        payload = {
            "task_type": "text_classification_hf",
            "dataset_name": "clue",
            "dataset_config": "iflytek",
            "model_name": "bert-base-chinese",
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 2e-5,
            "max_samples": 120,
        }
        res = self._post_json("/model/train", payload, token=self.teacher_token)
        body = self._safe_json(res)
        ok = res.status_code == 200 and "job_id" in body
        if ok:
            self.cls_job_id = int(body["job_id"])
        self._record("F-1", "触发分类微调任务", "/model/train", "POST", ok, payload, res.status_code, body)
        if not ok:
            return

        run = self._poll_job(self.cls_job_id)
        self.cls_run = run
        ok = run.get("status") == "success"
        self._record("F-2", "分类微调任务完成", f"/model/train/{self.cls_job_id}", "GET", ok, None, 200, run)
        if not ok:
            return

        metric = self._parse_metrics(run.get("metrics"))
        warn = isinstance(metric.get("f1"), (float, int)) and metric.get("f1", 0.0) < 0.05

        res = self._post_json("/model/predict", {"text": "顺序存储和链式存储区别"}, token=self.teacher_token)
        body = self._safe_json(res)
        ok = res.status_code == 200 and "label" in body
        self._record("F-3", "分类预测", "/model/predict", "POST", ok, {"text": "顺序存储和链式存储区别"}, res.status_code, body, warn=warn)

    def verify_hf_qa(self) -> None:
        payload = {
            "task_type": "qa_extractive_hf",
            "dataset_name": "cmrc2018",
            "model_name": "bert-base-chinese",
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 3e-5,
            "max_samples": 80,
        }
        res = self._post_json("/model/train", payload, token=self.teacher_token)
        body = self._safe_json(res)
        ok = res.status_code == 200 and "job_id" in body
        if ok:
            self.qa_job_id = int(body["job_id"])
        self._record("G-1", "触发QA微调任务", "/model/train", "POST", ok, payload, res.status_code, body)
        if not ok:
            return

        run = self._poll_job(self.qa_job_id)
        self.qa_run = run
        ok = run.get("status") == "success"
        self._record("G-2", "QA微调任务完成", f"/model/train/{self.qa_job_id}", "GET", ok, None, 200, run)
        if not ok:
            return

        payload = {
            "question": "什么是数据结构",
            "context": "数据结构是计算机存储、组织数据的方式。顺序存储使用连续空间。",
        }
        res = self._post_json("/model/qa_predict", payload, token=self.teacher_token)
        body = self._safe_json(res)
        ok = res.status_code == 200 and "answer" in body
        self._record("G-3", "QA抽取预测", "/model/qa_predict", "POST", ok, payload, res.status_code, body)

    def _poll_job(self, job_id: int) -> dict[str, Any]:
        start = time.time()
        last = {}
        while time.time() - start < self.poll_timeout_sec:
            res = self._get(f"/model/train/{job_id}", token=self.teacher_token)
            body = self._safe_json(res)
            last = body
            status = body.get("status")
            if status in {"success", "failed"}:
                return body
            time.sleep(3)
        last["status"] = "failed"
        last["error_message"] = f"Timeout: job {job_id} still running after {self.poll_timeout_sec}s"
        return last

    @staticmethod
    def _safe_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except Exception:
            return {"raw": response.text}

    @staticmethod
    def _parse_metrics(metrics: Any) -> dict[str, Any]:
        if isinstance(metrics, dict):
            return metrics
        if isinstance(metrics, str):
            try:
                return json.loads(metrics)
            except Exception:
                return {}
        return {}

    def write_reports(self) -> int:
        docs_dir = Path(__file__).resolve().parents[1] / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        json_path = docs_dir / "verification_report.json"
        md_path = docs_dir / "verification_report.md"

        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "base_url": self.base_url,
            "total": len(self.results),
            "pass": sum(1 for r in self.results if r.status == "pass"),
            "warn": sum(1 for r in self.results if r.status == "warn"),
            "fail": sum(1 for r in self.results if r.status == "fail"),
        }
        payload = {
            "summary": summary,
            "results": [asdict(r) for r in self.results],
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = [
            "# 自动化验收报告",
            "",
            f"- 时间: {summary['timestamp']}",
            f"- 服务: `{self.base_url}`",
            f"- 结果: pass={summary['pass']}, warn={summary['warn']}, fail={summary['fail']}",
            "",
            "## 用例结果",
            "",
            "| case_id | name | status | severity | endpoint | code |",
            "|---|---|---|---|---|---|",
        ]
        for r in self.results:
            code = r.status_code if r.status_code is not None else "-"
            lines.append(f"| {r.case_id} | {r.name} | {r.status} | {r.severity} | `{r.endpoint}` | {code} |")

        lines += ["", "## 失败/警告详情", ""]
        for r in self.results:
            if r.status in {"fail", "warn"}:
                lines.append(f"### {r.case_id} - {r.name} ({r.status})")
                lines.append(f"- endpoint: `{r.endpoint}`")
                lines.append(f"- status_code: `{r.status_code}`")
                lines.append(f"- response: `{json.dumps(r.response_body, ensure_ascii=False)}`")
                if r.exception_log:
                    lines.append(f"- exception: `{r.exception_log}`")
                lines.append("")

        md_path.write_text("\n".join(lines), encoding="utf-8")
        self._append_docs_summary(summary)

        has_blocker = any(r.status == "fail" and r.severity == "blocker" for r in self.results)
        return 1 if has_blocker else 0

    def _append_docs_summary(self, summary: dict[str, Any]) -> None:
        docs_dir = Path(__file__).resolve().parents[1] / "docs"
        worklog = docs_dir / "WORKLOG.md"
        progress = docs_dir / "PROGRESS.md"
        stamp = summary["timestamp"]
        note = (
            f"\n- 自动化验收执行（{stamp}）：pass={summary['pass']}, warn={summary['warn']}, fail={summary['fail']}。"
            " 详情见 `docs/verification_report.md`。\n"
        )

        for path in (worklog, progress):
            if path.exists():
                content = path.read_text(encoding="utf-8", errors="ignore")
                path.write_text(content + note, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run end-to-end automated verification against online EduAssist service.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--poll-timeout", type=int, default=1200)
    parser.add_argument("--trust-env", action="store_true", help="Use system proxy/env settings for HTTP requests.")
    args = parser.parse_args()

    verifier = Verifier(
        base_url=args.base_url,
        timeout_sec=args.timeout,
        poll_timeout_sec=args.poll_timeout,
        trust_env=args.trust_env,
    )
    return verifier.run()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())
