"""Conservative local classification for professional-facing artifacts."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from chatgpt_archive_compiler.models import Conversation


class SafetyDecision(StrEnum):
    """Three-way professional-facing publication decision."""

    INCLUDE = "include"
    EXCLUDE = "exclude"
    REVIEW = "review"


class ClassifierSource(StrEnum):
    """Source of a professional-safety classification."""

    DETERMINISTIC_RULE = "deterministic rule"
    LOCAL_MODEL = "local model"
    REMOTE_MODEL = "remote model"
    USER_OVERRIDE = "user override"


class ProfessionalSafetyDecision(BaseModel):
    """User-visible, content-minimal local publication decision."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    schema_version: str = "1.0"
    conversation_identifier: str
    decision: SafetyDecision
    sensitivity_categories: tuple[str, ...] = ()
    reason: str
    classifier_source: ClassifierSource = ClassifierSource.DETERMINISTIC_RULE
    review_priority: float = Field(ge=0, le=1)
    timestamp: datetime


_EXCLUDE_RULES = {
    "credential": re.compile(r"(?i)(?:api[_ -]?key|password|secret)\s*[:=]\s*\S{6,}"),
    "intimate": re.compile(r"(?i)\b(intimate|dating|sexual|breakup|my partner)\b"),
    "medical": re.compile(r"(?i)\b(my diagnosis|my medication|therapist|mental health)\b"),
    "financial": re.compile(r"(?i)\b(my debt|financial hardship|cannot pay rent|account number)\b"),
    "explicitly-private": re.compile(
        r"(?i)\b(mark this private|do not share|confidential personal)\b"
    ),
}
_REVIEW_RULES = {
    "workplace": re.compile(r"(?i)\b(my manager|workplace grievance|coworker complaint)\b"),
    "personal": re.compile(r"(?i)\b(family conflict|religious belief|political belief|address)\b"),
}


def _visible_text(conversation: Conversation) -> str:
    parts = [conversation.title or ""]
    for message in conversation.current_path_messages:
        for block in message.content:
            if block.text:
                parts.append(block.text)
    return "\n".join(parts)


def classify_conversation(
    conversation: Conversation, identifier: str
) -> ProfessionalSafetyDecision:
    """Classify using narrow, explainable rules; uncertainty becomes review."""

    text = _visible_text(conversation)
    excluded = tuple(name for name, pattern in _EXCLUDE_RULES.items() if pattern.search(text))
    review = tuple(name for name, pattern in _REVIEW_RULES.items() if pattern.search(text))
    if excluded:
        decision, categories, reason, priority = (
            SafetyDecision.EXCLUDE,
            excluded,
            "A local high-risk pattern indicates this conversation needs private handling.",
            0.95,
        )
    elif review:
        decision, categories, reason, priority = (
            SafetyDecision.REVIEW,
            review,
            "Context may be professionally sensitive; review before sharing.",
            0.7,
        )
    else:
        decision, categories, reason, priority = (
            SafetyDecision.INCLUDE,
            (),
            "No narrow high-risk local rule matched; manual review is still recommended.",
            0.2,
        )
    return ProfessionalSafetyDecision(
        conversation_identifier=identifier,
        decision=decision,
        sensitivity_categories=categories,
        reason=reason,
        review_priority=priority,
        timestamp=datetime.now(UTC),
    )


def classify_archive(
    conversations: tuple[Conversation, ...],
) -> tuple[ProfessionalSafetyDecision, ...]:
    """Classify every conversation without transmitting or logging its content."""

    return tuple(
        classify_conversation(item, f"conversation-{index:06d}")
        for index, item in enumerate(conversations)
    )
