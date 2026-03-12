from __future__ import annotations

from dataclasses import dataclass
import unittest
from unittest.mock import patch

from app.services.assignment_grading import (
    evaluate_relevance,
    generate_text_assignment_feedback,
    parse_llm_feedback,
)


@dataclass
class DummyAssignment:
    title: str
    description: str | None = None
    keywords: str | None = None


class AssignmentGradingTests(unittest.IsolatedAsyncioTestCase):
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

    def test_off_topic_for_thread_definition(self):
        assignment = DummyAssignment(
            title="作业1",
            description="什么是进程？",
            keywords=None,
        )
        content = "线程是操作系统调度的最小单位，共享进程资源。"

        result = evaluate_relevance(assignment, content)

        self.assertEqual(result.label, "off_topic")

    def test_generic_title_description_priority(self):
        assignment = DummyAssignment(
            title="作业1",
            description="请解释死锁的四个必要条件。",
            keywords=None,
        )
        content = "死锁发生必须同时满足互斥、请求与保持、不可剥夺、循环等待四个必要条件。"

        result = evaluate_relevance(assignment, content)

        self.assertEqual(result.label, "relevant")

    async def test_ambiguous_llm_unavailable_fallback_to_relevant_feedback(self):
        assignment = DummyAssignment(
            title="作业1",
            description=None,
            keywords=None,
        )
        content = "这是一次简短回答。"

        async def fake_llm(_prompt: str) -> str:
            return "[GLM接口错误]timeout after 70s"

        feedback = await generate_text_assignment_feedback(assignment, content, fake_llm)

        self.assertTrue(feedback)
        self.assertNotIn("要求不一致", feedback)

    def test_parse_llm_feedback_strips_score(self):
        raw = "评分: 85\n评语: 回答基本正确，建议补充进程状态转换和调度示例。"
        cleaned = parse_llm_feedback(raw)

        self.assertNotIn("评分", cleaned)
        self.assertIn("回答基本正确", cleaned)

    def test_parse_llm_feedback_strips_single_line_score(self):
        raw = "评分:85评语:你的定义基本正确，但建议补充进程与线程区别。"
        cleaned = parse_llm_feedback(raw)

        self.assertNotIn("评分", cleaned)
        self.assertIn("你的定义基本正确", cleaned)

    def test_noise_text_off_topic(self):
        assignment = DummyAssignment(
            title="作业1",
            description="什么是进程？",
            keywords=None,
        )
        content = "今天天气不错，我们准备去操场打球。"

        result = evaluate_relevance(assignment, content)

        self.assertEqual(result.label, "off_topic")

    def test_model_conflict_kept_ambiguous(self):
        assignment = DummyAssignment(
            title="作业1",
            description="什么是进程？",
            keywords=None,
        )
        content = "线程是操作系统调度的最小单位，共享进程资源。"

        with patch("app.services.assignment_grading.settings.enable_assignment_relevance_model", True), patch(
            "app.services.assignment_grading.resolve_active_assignment_relevance_model_path",
            return_value=("models/assignment_rel_dummy", "model_run_latest"),
        ), patch(
            "app.services.assignment_grading.predict_relevance_probability",
            return_value=0.91,
        ):
            result = evaluate_relevance(assignment, content)

        self.assertEqual(result.label, "ambiguous")


if __name__ == "__main__":
    unittest.main()
