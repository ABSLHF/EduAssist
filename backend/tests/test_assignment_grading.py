from __future__ import annotations

from dataclasses import dataclass
import json
import tempfile
import unittest
from unittest.mock import patch

from app.services.assignment_feedback import (
    evaluate_relevance,
    generate_text_assignment_feedback,
    parse_llm_feedback,
)


@dataclass
class DummyAssignment:
    title: str
    description: str | None = None
    keywords: str | None = None
    course_id: int | None = None


class AssignmentFeedbackTests(unittest.IsolatedAsyncioTestCase):
    def test_relevant_for_process_definition(self):
        assignment = DummyAssignment(
            title="作业1",
            description="什么是进程？",
            keywords=None,
        )
        content = "进程是程序的一次执行过程，是操作系统进行资源分配和调度的基本单位。"
        result = evaluate_relevance(assignment, content)
        self.assertEqual(result.label, "relevant")
        self.assertIn("进程", result.matched_core_terms)

    def test_off_topic_for_noise_text(self):
        assignment = DummyAssignment(
            title="作业1",
            description="什么是进程？",
            keywords=None,
        )
        content = "今天天气很好，我准备去操场跑步。"
        result = evaluate_relevance(assignment, content)
        self.assertEqual(result.label, "off_topic")

    def test_title_index_tokens_should_not_be_focus_terms(self):
        assignment = DummyAssignment(
            title="数据结构test1，作业1",
            description="数据结构有哪些？",
            keywords=None,
        )
        content = "数据结构包括数组、链表、栈、队列、树和图。"
        result = evaluate_relevance(assignment, content)
        self.assertNotIn("test1", result.focus_terms)
        self.assertNotIn("作业1", result.focus_terms)
        self.assertNotEqual(result.label, "off_topic")

    def test_parse_llm_feedback_strips_score(self):
        raw = "评分: 85\n评语: 回答基本正确，但建议补充进程与线程区别。"
        cleaned = parse_llm_feedback(raw)
        self.assertNotIn("评分", cleaned)
        self.assertIn("回答基本正确", cleaned)

    async def test_v2_fallback_has_three_sections_and_no_score(self):
        assignment = DummyAssignment(
            title="作业1",
            description="什么是进程？",
            keywords="进程,调度",
            course_id=1,
        )

        async def fake_llm(_prompt: str) -> str:
            return "[GLM接口错误]timeout after 70s"

        with patch("app.services.assignment_feedback.settings.assignment_feedback_mode", "v2"), patch(
            "app.services.assignment_feedback.retrieve",
            return_value=[],
        ):
            feedback = await generate_text_assignment_feedback(assignment, "线程是CPU调度单位。", fake_llm)

        self.assertIn("优点：", feedback)
        self.assertIn("问题：", feedback)
        self.assertIn("改进建议：", feedback)
        self.assertNotIn("评分", feedback)
        self.assertNotIn("得分", feedback)

    async def test_shadow_mode_returns_legacy_and_writes_log(self):
        assignment = DummyAssignment(
            title="作业1",
            description="什么是进程？",
            keywords="进程,调度",
            course_id=1,
        )

        async def fake_llm(prompt: str) -> str:
            if "仅输出JSON" in prompt:
                return '{"label":"off_topic","reason":"偏题"}'
            return "优点：结构清晰\n问题：覆盖不全\n改进建议：补充核心术语。"

        with tempfile.TemporaryDirectory() as tmp:
            log_path = f"{tmp}/shadow.jsonl"
            with patch("app.services.assignment_feedback.settings.assignment_feedback_mode", "shadow"), patch(
                "app.services.assignment_feedback.settings.assignment_feedback_shadow_log_path",
                log_path,
            ), patch("app.services.assignment_feedback.retrieve", return_value=[]):
                feedback = await generate_text_assignment_feedback(assignment, "线程是CPU调度单位。", fake_llm)

            self.assertTrue(feedback)
            with open(log_path, "r", encoding="utf-8") as fp:
                lines = [line.strip() for line in fp if line.strip()]
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload.get("mode"), "shadow")
            self.assertIn("legacy_feedback", payload)
            self.assertIn("v2_feedback", payload)


if __name__ == "__main__":
    unittest.main()
