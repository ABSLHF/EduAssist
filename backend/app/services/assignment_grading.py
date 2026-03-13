"""Compatibility wrapper.

This module is kept for backward compatibility and forwards all assignment
feedback logic to app.services.assignment_feedback.
"""

from app.services.assignment_feedback import (  # noqa: F401
    FeedbackDiagnostic,
    FeedbackTier,
    RelevanceLabel,
    RelevanceResult,
    evaluate_relevance,
    generate_text_assignment_feedback,
    parse_llm_feedback,
)
