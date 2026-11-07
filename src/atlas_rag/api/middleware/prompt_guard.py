"""Prompt injection protection middleware.

Scans user queries for known prompt injection patterns and blocks or sanitizes
them before they reach the LLM generation pipeline.
"""

from __future__ import annotations

import logging
import re

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# Known injection patterns
_INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|above)\s+(instructions|prompts|rules)",
    r"(disregard|forget)\s+(everything|all|your)\s+(above|previous|instructions)",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"new\s+instructions?\s*:",
    r"system\s*:\s*",
    r"<\s*/?\s*system\s*>",
    r"```\s*system",
    r"act\s+as\s+(if\s+)?you\s+(are|were)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"override\s+(your\s+)?(instructions|rules|guidelines)",
    r"(reveal|show|print|output)\s+(your\s+)?(system\s+)?(prompt|instructions)",
    r"do\s+not\s+follow\s+(your\s+)?instructions",
    r"jailbreak",
    r"DAN\s+(mode|prompt)",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def check_prompt_injection(text: str) -> bool:
    """Check if text contains prompt injection attempts.

    Returns True if injection detected, False if clean.
    """
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(text):
            logger.warning(
                "Prompt injection detected: pattern=%s text=%r",
                pattern.pattern[:50],
                text[:100],
            )
            return True
    return False


def sanitize_query(text: str) -> str:
    """Remove or neutralize prompt injection attempts from user input."""
    sanitized = text

    # Remove potential system prompt markers
    sanitized = re.sub(r"<\s*/?\s*system\s*>", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"```\s*system\s*```", "", sanitized, flags=re.IGNORECASE)

    # Remove unicode tricks (zero-width chars, direction overrides)
    sanitized = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\ufeff]", "", sanitized)

    return sanitized.strip()


def guard_query(text: str) -> str:
    """Validate and sanitize a user query. Raises HTTPException if injection detected."""
    if check_prompt_injection(text):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query rejected: potential prompt injection detected.",
        )
    return sanitize_query(text)
