from __future__ import annotations

from dataclasses import dataclass
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
        assignment = DummyAssignment(title="作业1", description="什么是进程？", keywords=None)
        content = "进程是程序的一次执行过程，是操作系统进行资源分配和调度的基本单位。"
        result = evaluate_relevance(assignment, content)
        self.assertEqual(result.label, "relevant")
        self.assertIn("进程", result.matched_core_terms)

    def test_off_topic_for_noise_text(self):
        assignment = DummyAssignment(title="作业1", description="什么是进程？", keywords=None)
        content = "今天天气很好，我准备去操场跑步。"
        result = evaluate_relevance(assignment, content)
        self.assertEqual(result.label, "off_topic")

    def test_parse_llm_feedback_cleanup(self):
        raw = "### 作业批改评语\n**评分: 85**\n**问题：**内容偏题\n**改进建议：**重写答案"
        cleaned = parse_llm_feedback(raw)
        self.assertNotIn("###", cleaned)
        self.assertNotIn("**", cleaned)
        self.assertNotIn("评分", cleaned)
        self.assertIn("问题：", cleaned)

    async def test_invalid_answer_should_not_output_pros(self):
        assignment = DummyAssignment(
            title="数据结构test1，作业1",
            description="数据结构有哪些？",
            keywords="数组,链表,栈,队列",
            course_id=1,
        )

        async def fake_llm(_prompt: str) -> str:
            return "优点：有尝试作答。\n问题：内容无意义。\n改进建议：请重写。"

        with patch("app.services.assignment_feedback.settings.enable_assignment_feedback_sft_model", False), patch(
            "app.services.assignment_feedback.settings.assignment_feedback_external_fallback", True
        ):
            feedback = await generate_text_assignment_feedback(assignment, "流口水噶包括v那种", fake_llm)

        self.assertNotIn("优点：", feedback)
        self.assertIn("问题：", feedback)
        self.assertIn("改进建议：", feedback)

    async def test_offtopic_answer_should_not_output_pros(self):
        assignment = DummyAssignment(
            title="作业1",
            description="什么是进程？",
            keywords="进程,调度",
            course_id=1,
        )

        async def fake_llm(_prompt: str) -> str:
            return "优点：表达流畅。\n问题：回答了线程定义。\n改进建议：改答进程。"

        with patch("app.services.assignment_feedback.settings.enable_assignment_feedback_sft_model", False), patch(
            "app.services.assignment_feedback.settings.assignment_feedback_external_fallback", True
        ):
            feedback = await generate_text_assignment_feedback(assignment, "线程是CPU调度的最小单位。", fake_llm)

        self.assertNotIn("优点：", feedback)
        self.assertIn("问题：", feedback)

    async def test_scope_guard_should_remove_unasked_dimensions(self):
        assignment = DummyAssignment(
            title="数据结构test1，作业1",
            description="数据结构有哪些？",
            keywords="数组,链表,栈,队列",
            course_id=1,
        )

        async def fake_llm(_prompt: str) -> str:
            return (
                "优点：列举了数组、链表、栈、队列。\n"
                "问题：缺少时间复杂度、空间复杂度和应用场景分析。\n"
                "改进建议：补充优缺点比较与工程实践案例。"
            )

        with patch("app.services.assignment_feedback.settings.enable_assignment_feedback_sft_model", False), patch(
            "app.services.assignment_feedback.settings.assignment_feedback_external_fallback", True
        ):
            feedback = await generate_text_assignment_feedback(assignment, "数组、链表、栈、队列", fake_llm)

        self.assertNotIn("时间复杂度", feedback)
        self.assertNotIn("空间复杂度", feedback)
        self.assertNotIn("应用场景", feedback)
        self.assertNotIn("优缺点", feedback)
        self.assertNotIn("工程实践", feedback)

    async def test_good_answer_should_keep_pros(self):
        assignment = DummyAssignment(
            title="作业2",
            description="什么是哈希表？",
            keywords="哈希表,键值映射",
            course_id=1,
        )

        async def fake_llm(_prompt: str) -> str:
            return "优点：定义清晰，术语准确。\n改进建议：可再补充一个简短示例。"

        with patch("app.services.assignment_feedback.settings.enable_assignment_feedback_sft_model", False), patch(
            "app.services.assignment_feedback.settings.assignment_feedback_external_fallback", True
        ):
            feedback = await generate_text_assignment_feedback(
                assignment,
                "哈希表是一种通过哈希函数把键映射到存储位置的数据结构。",
                fake_llm,
            )

        self.assertIn("优点：", feedback)
        self.assertIn("改进建议：", feedback)

    async def test_llm_failure_should_use_rule_fallback(self):
        assignment = DummyAssignment(
            title="作业1",
            description="什么是进程？",
            keywords="进程,调度",
            course_id=1,
        )

        async def fake_llm(_prompt: str) -> str:
            return "[GLM接口错误]timeout after 70s"

        with patch("app.services.assignment_feedback.settings.enable_assignment_feedback_sft_model", False), patch(
            "app.services.assignment_feedback.settings.assignment_feedback_external_fallback", True
        ):
            feedback = await generate_text_assignment_feedback(assignment, "线程是CPU调度单位。", fake_llm)

        self.assertTrue(feedback)
        self.assertIn("改进建议：", feedback)

    async def test_external_fallback_disabled_should_still_return_feedback(self):
        assignment = DummyAssignment(
            title="作业1",
            description="什么是进程？",
            keywords="进程,调度",
            course_id=1,
        )

        async def fake_llm(_prompt: str) -> str:
            return "优点：这段不会被使用"

        with patch("app.services.assignment_feedback.settings.enable_assignment_feedback_sft_model", False), patch(
            "app.services.assignment_feedback.settings.assignment_feedback_external_fallback", False
        ):
            feedback = await generate_text_assignment_feedback(assignment, "", fake_llm)

        self.assertIn("问题：", feedback)
        self.assertIn("改进建议：", feedback)


if __name__ == "__main__":
    unittest.main()
