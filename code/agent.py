"""
agent.py — Stages 3 + 4: LLM triage call.

Takes the retrieved context chunks + ticket fields and produces a structured
JSON response with all 5 required output columns.

If retrieval confidence is below RETRIEVAL_CONFIDENCE_THRESHOLD, escalates
immediately WITHOUT making an API call (saves quota, avoids hallucination).

Hybrid approach:
  1. Try Google Gemini API first (best JSON quality)
  2. If Gemini returns 429 ResourceExhausted, fall back to local Ollama
  3. Retry with exponential backoff on each backend
  4. Only escalate if BOTH backends fail
"""
from __future__ import annotations

import json
import os
import re
import time
import random

import google.generativeai as genai

from config import (
    MODEL, MAX_TOKENS,
    RETRIEVAL_CONFIDENCE_THRESHOLD,
    API_MAX_RETRIES, API_BACKOFF_BASE, API_BACKOFF_JITTER, API_INTER_CALL_DELAY,
    OLLAMA_HOST, OLLAMA_MODEL,
)
from corpus import Chunk

# Lazy-initialised so import doesn't fail if key is missing at module load time
_gemini_client = None
_use_ollama_only = False  # True = skip Gemini entirely (--ollama flag)


def set_ollama_mode(enabled: bool) -> None:
    """Called by main.py when --ollama flag is present."""
    global _use_ollama_only
    _use_ollama_only = enabled


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        _gemini_client = genai.GenerativeModel(MODEL)
    return _gemini_client


def _call_ollama(prompt: str) -> str:
    """Call local Ollama API. Returns raw text response."""
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.3,
            "num_predict": MAX_TOKENS,
        },
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
        return data.get("response", "").strip()


def _call_gemini(prompt_parts: list, max_tokens: int, temperature: float) -> str:
    """Call Google Gemini API (single attempt, no retry)."""
    client = _get_gemini_client()
    response = client.generate_content(
        prompt_parts,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    return (response.text or "").strip()


def _is_quota_error(exc: Exception) -> bool:
    """Check if an exception is a 429/quota/rate-limit error."""
    exc_str = str(exc).lower()
    return any(signal in exc_str for signal in ["429", "resourceexhausted", "quota", "rate"])


def _call_with_retry_and_fallback(
    system_prompt: str,
    user_message: str,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.3,
) -> str:
    """
    Hybrid LLM call strategy:
    1. If --ollama mode: use Ollama only
    2. Otherwise: try Gemini with retries, then fall back to Ollama on quota errors
    """
    # ── Ollama-only mode ──────────────────────────────────────────────────────
    if _use_ollama_only:
        full_prompt = f"{system_prompt}\n\n{user_message}"
        return _call_ollama(full_prompt)

    # ── Gemini with retry ─────────────────────────────────────────────────────
    last_exc = None
    for attempt in range(API_MAX_RETRIES):
        try:
            return _call_gemini(
                [system_prompt, "\n\n", user_message],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            last_exc = exc
            if _is_quota_error(exc):
                wait = (API_BACKOFF_BASE ** attempt) + random.uniform(0, API_BACKOFF_JITTER)
                time.sleep(wait)
                continue
            # Non-quota error (404, auth, etc.) — skip straight to Ollama
            break

    # ── Gemini failed — try Ollama fallback ───────────────────────────────────
    try:
        full_prompt = f"{system_prompt}\n\n{user_message}"
        return _call_ollama(full_prompt)
    except Exception:
        pass  # Ollama also failed

    # Both failed — raise the original Gemini error
    if last_exc:
        raise last_exc
    raise RuntimeError("Both Gemini and Ollama backends failed")


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a professional support triage agent for a multi-product company.
You have been given a support ticket and relevant excerpts from the official support documentation.

YOUR RULES (NON-NEGOTIABLE):
1. Base ALL responses ONLY on the provided documentation excerpts. Never use outside knowledge.
2. If the documentation DOES clearly answer the question, you MUST set status to "replied" and provide the answer.
3. Only set status to "escalated" if the documentation truly cannot answer the question.
4. Never invent policies, prices, timelines, or features not in the provided context.
5. Only escalate for billing disputes, account security breaches, legal matters, or platform-wide outages.

IMPORTANT — prefer "replied" when docs are present:
- If you can see relevant documentation chunks, always try to answer. Being overly cautious and escalating answerable questions is WRONG.
- A common mistake is escalating when the docs clearly have the answer. Do NOT do this.

Classification guide for request_type:
- "bug": user reports something BROKEN or NOT WORKING (e.g. "error", "crash", "failing", "down", "not loading", "site is down")
- "feature_request": user asks for a NEW capability ("I wish", "please add", "can you implement")
- "invalid": spam, greetings, thank-you messages, or questions completely outside the products' scope
  Examples of "invalid": "Who is the actor in Iron Man?", "Thank you for helping me", "Thanks!", "Never mind"
- "product_issue": everything else — general questions, how-to, account help, settings, configuration

EXAMPLES of correct status classification:
- "How do I reset my HackerRank test timer?" + docs present → status: "replied"
- "How long do tests stay active in the system?" + docs present → status: "replied"
- "How do I add extra time for a candidate?" + docs present → status: "replied"
- "The entire website is down, nothing loads" → status: "escalated", request_type: "bug"
- "Thank you!" → status: "replied", request_type: "invalid"
- "Who is Batman?" → status: "replied", request_type: "invalid"

You MUST respond with ONLY valid JSON. No markdown, no explanation, no extra text.
The JSON must have exactly these 5 fields:

{"status": "replied", "product_area": "category", "response": "2-4 sentence reply to user", "justification": "1-2 sentence internal note", "request_type": "product_issue"}

Valid status values: "replied" or "escalated"
Valid request_type values: "product_issue", "feature_request", "bug", "invalid"
"""

# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_context_block(chunks: list[Chunk]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    sections = []
    for i, c in enumerate(chunks, start=1):
        sections.append(
            f"[{i}] Source: {c.domain}/{c.source_file} (chunk {c.chunk_idx})\n{c.text}"
        )
    return "\n\n---\n\n".join(sections)


def _parse_json(raw: str) -> dict:
    """
    Parse JSON from the LLM response.
    Falls back to regex extraction if the model wrapped it in markdown fences.
    """
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` or bare { ... }
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Last resort: return a safe fallback
    return {}


def _enforce_schema(result: dict, domain: str) -> dict:
    """Ensure all required keys are present and values are within allowed sets."""
    from config import VALID_STATUSES, VALID_REQUEST_TYPES, OUTPUT_COLUMNS

    # Fill in missing keys with safe defaults
    # If there IS a response text, default to "replied" rather than "escalated"
    has_response = bool(result.get("response", "").strip())
    defaults = {
        "status": "replied" if has_response else "escalated",
        "product_area": domain if domain != "unknown" else "general",
        "response": "Unable to generate a structured response. Escalating for human review.",
        "justification": "LLM response was missing required fields.",
        "request_type": "product_issue",
    }
    for col in OUTPUT_COLUMNS:
        if col not in result or not result[col]:
            result[col] = defaults[col]

    # Constrain enum fields
    if result["status"] not in VALID_STATUSES:
        result["status"] = "replied" if has_response else "escalated"
    if result["request_type"] not in VALID_REQUEST_TYPES:
        result["request_type"] = "product_issue"

    return result


def _safe_escalation(domain: str, reason: str, exc_name: str = "") -> dict:
    """Return a safe escalated response dict."""
    return {
        "status": "escalated",
        "product_area": domain if domain != "unknown" else "general",
        "response": (
            "We are currently unable to generate an automated response. "
            "Your request has been escalated to a human agent."
        ),
        "justification": (
            f"LLM provider call failed during triage "
            f"({exc_name or reason}). Escalated for safe handling."
        ),
        "request_type": "product_issue",
    }


# ── Main triage function ───────────────────────────────────────────────────────

def triage(
    issue: str,
    subject: str,
    domain: str,
    chunks: list[Chunk],
    confidence: float,
) -> dict:
    """
    Main triage call — Stage 3 + 4.

    Args:
        issue:      Ticket body text.
        subject:    Ticket subject / title.
        domain:     Detected domain ('hackerrank' | 'claude' | 'visa' | 'unknown').
        chunks:     Retrieved context chunks from the retriever.
        confidence: Mean hybrid retrieval score of the top-k chunks.

    Returns:
        Dict with keys: status, product_area, response, justification, request_type.
    """
    # ── Low-confidence short-circuit (no LLM call) ────────────────────────────
    if confidence < RETRIEVAL_CONFIDENCE_THRESHOLD:
        return {
            "status": "escalated",
            "product_area": domain if domain != "unknown" else "general",
            "response": (
                "We could not find sufficient documentation to address this request. "
                "A human agent will follow up."
            ),
            "justification": (
                f"Retrieval confidence {confidence:.3f} is below threshold "
                f"{RETRIEVAL_CONFIDENCE_THRESHOLD}. Insufficient context to answer safely."
            ),
            "request_type": "product_issue",
        }

    # ── Build user message ────────────────────────────────────────────────────
    context_block = _build_context_block(chunks)
    user_message = (
        f"TICKET\n"
        f"Subject: {subject or '(none)'}\n"
        f"Company/Product: {domain}\n"
        f"Issue:\n{issue}\n\n"
        f"DOCUMENTATION CONTEXT:\n{context_block}\n\n"
        f"Now produce the JSON output. Remember: output ONLY the JSON object, nothing else."
    )

    # ── Throttle: small delay to reduce burst API calls ───────────────────────
    time.sleep(API_INTER_CALL_DELAY)

    # ── LLM call (hybrid: Gemini first, Ollama fallback) ─────────────────────
    try:
        raw_text = _call_with_retry_and_fallback(
            SYSTEM_PROMPT,
            user_message,
            max_tokens=MAX_TOKENS,
            temperature=0.3,
        )
    except Exception as exc:
        # All backends exhausted — safe escalation
        return _safe_escalation(domain, "API error", type(exc).__name__)

    # ── Parse and validate ────────────────────────────────────────────────────
    parsed = _parse_json(raw_text)
    if not parsed:
        parsed = {
            "status": "escalated",
            "product_area": domain,
            "response": "Unable to generate a structured response. Escalating for human review.",
            "justification": "LLM response could not be parsed as valid JSON.",
            "request_type": "product_issue",
        }

    return _enforce_schema(parsed, domain)
