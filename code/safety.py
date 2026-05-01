"""
safety.py — Stage 1: Adversarial / validity safety check.

Runs BEFORE any retrieval or LLM call.
Fast, deterministic, keyword-based — no API cost.

Returns None if the ticket is safe to proceed.
Returns a pre-built escalation dict if the ticket is unsafe/invalid.

Safety policy:
  - Only escalate for PROMPT INJECTION attempts (adversarial attacks on the AI)
  - Do NOT escalate for financial/security support questions — those are legitimate
    support topics with documentation answers (e.g. "my card was stolen" → Visa has
    clear instructions for this).
"""
from config import ESCALATION_KEYWORDS


# Signals that strongly indicate prompt injection attempts
_INJECTION_SIGNALS = [
    "ignore previous",
    "disregard your",
    "you are now",
    "pretend you are",
    "new instructions:",
    "system prompt",
    "jailbreak",
    "ignore all previous instructions",
    "forget your instructions",
    "override your rules",
    "act as",
    "delete all files",
]


def check_safety(issue: str, subject: str) -> dict | None:
    """
    Returns None if the ticket is safe to proceed through the pipeline.
    Returns a pre-built escalation row dict if the ticket triggers safety rules.

    Args:
        issue:   The body text of the support ticket.
        subject: The subject / title of the support ticket.

    Returns:
        None (safe) or a dict with keys matching OUTPUT_COLUMNS.
    """
    combined = f"{subject} {issue}".lower()

    # ── Prompt injection detection ────────────────────────────────────────────
    for signal in _INJECTION_SIGNALS:
        if signal in combined:
            return {
                "status": "escalated",
                "product_area": "security",
                "response": (
                    "This request has been flagged for security review "
                    "and escalated to our team."
                ),
                "justification": (
                    "Input contains prompt injection signals "
                    "and cannot be processed automatically."
                ),
                "request_type": "invalid",
            }

    # Financial/security keywords like "stolen", "fraud", "dispute" are NOT
    # escalated here — they are legitimate support topics with documentation
    # answers. The LLM + retriever will handle them properly.

    return None  # safe to proceed
