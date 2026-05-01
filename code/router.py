"""
router.py — Stage 2: Domain detection.

Priority order:
  1. Use the 'company' field directly if it matches a known domain.
  2. Fall back to keyword-score matching on subject + issue text.
  3. Return 'unknown' if no domain scores above zero.

No LLM call; fully deterministic.
"""
from config import DOMAIN_HINTS

# Canonical domain names (lowercase)
_KNOWN_DOMAINS = {"hackerrank", "claude", "visa"}


def detect_domain(issue: str, subject: str, company: str) -> str:
    """
    Detect which product domain a ticket belongs to.

    Args:
        issue:   Ticket body text.
        subject: Ticket subject / title.
        company: Value from the 'company' CSV column (may be empty or 'None').

    Returns:
        One of: 'hackerrank' | 'claude' | 'visa' | 'unknown'
    """
    # ── 1. Direct company-field lookup ────────────────────────────────────────
    if company:
        normalized = company.strip().lower()
        if normalized in _KNOWN_DOMAINS:
            return normalized

    # ── 2. Keyword scoring ────────────────────────────────────────────────────
    combined = f"{subject} {issue}".lower()
    scores: dict[str, int] = {}
    for domain, hints in DOMAIN_HINTS.items():
        scores[domain] = sum(1 for h in hints if h in combined)

    best_domain = max(scores, key=scores.get)
    if scores[best_domain] > 0:
        return best_domain

    # ── 3. Unknown ────────────────────────────────────────────────────────────
    return "unknown"
