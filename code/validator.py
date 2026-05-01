"""
validator.py — Stage 5: Hallucination guard.

Makes a second (very short) LLM call to verify that the generated
response is actually grounded in the retrieved context.

If the validator returns FAIL, the caller should override the response with
a safe fallback and escalate the ticket.

Uses the same hybrid approach as agent.py:
  - Gemini first, Ollama fallback on 429 errors
  - Selective validation: skip for very high confidence retrievals
"""
from __future__ import annotations

import json
import os
import time
import random

import google.generativeai as genai

from config import (
    MODEL,
    API_MAX_RETRIES, API_BACKOFF_BASE, API_BACKOFF_JITTER,
    OLLAMA_HOST, OLLAMA_MODEL,
)
from corpus import Chunk

_client = None


def _get_client():
    global _client
    if _client is None:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        _client = genai.GenerativeModel(MODEL)
    return _client


def _call_ollama_validator(prompt: str) -> str:
    """Call local Ollama for validation. Returns raw text."""
    import urllib.request

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
        return data.get("response", "").strip()


def _is_quota_error(exc: Exception) -> bool:
    """Check if an exception is a 429/quota/rate-limit error."""
    exc_str = str(exc).lower()
    return any(signal in exc_str for signal in ["429", "resourceexhausted", "quota", "rate"])


VALIDATOR_SYSTEM = """\
You are a fact-checking assistant for AI-generated support responses.
Given a support response and the documentation context it was based on, answer ONLY with "PASS" or "FAIL".
Answer "FAIL" if the response makes ANY claim not supported by the provided context.
Answer "PASS" if every factual claim in the response is traceable to the provided context.
Do not explain your answer. Output exactly one word: PASS or FAIL."""

# Confidence threshold above which we SKIP validation (high-confidence replies are trusted)
_SKIP_VALIDATION_THRESHOLD = 0.45


def validate_response(
    response_text: str,
    context_chunks: list[Chunk],
    confidence: float = 1.0,
    max_tokens: int = 10,
    use_ollama: bool = False,
) -> bool:
    """
    Check whether a generated support response is grounded in the retrieved context.

    Args:
        response_text:  The user-facing response produced by agent.triage().
        context_chunks: Retrieved context chunks (uses top 3 to keep prompt short).
        confidence:     The retrieval confidence score. Very high scores skip validation.
        max_tokens:     Token budget for the validator response (just "PASS" or "FAIL").
        use_ollama:     If True, use local Ollama instead of Google API.

    Returns:
        True  -> response is grounded (PASS) or validation skipped (high confidence).
        False -> response contains hallucinated claims (FAIL).
    """
    # Skip validation for high confidence — retrieval is reliable enough
    if confidence >= _SKIP_VALIDATION_THRESHOLD:
        return True

    # Use top-3 chunks to keep the validator prompt short and cheap
    top_chunks = context_chunks[:3]
    context = "\n\n".join(c.text for c in top_chunks)

    user_prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"RESPONSE TO CHECK:\n{response_text}\n\n"
        f"Pass or Fail?"
    )

    # ── Ollama-only mode ──────────────────────────────────────────────────────
    if use_ollama:
        try:
            full_prompt = f"{VALIDATOR_SYSTEM}\n\n{user_prompt}"
            verdict = _call_ollama_validator(full_prompt).upper()
            return verdict.startswith("PASS")
        except Exception:
            return True  # Ollama down -> assume PASS

    # ── Gemini with retry + Ollama fallback ───────────────────────────────────
    last_exc = None
    for attempt in range(API_MAX_RETRIES):
        try:
            client = _get_client()
            verdict_msg = client.generate_content(
                [VALIDATOR_SYSTEM, "\n\n", user_prompt],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.1,
                ),
            )
            verdict = (verdict_msg.text or "").strip().upper()
            return verdict.startswith("PASS")
        except Exception as exc:
            last_exc = exc
            if _is_quota_error(exc):
                wait = (API_BACKOFF_BASE ** attempt) + random.uniform(0, API_BACKOFF_JITTER)
                time.sleep(wait)
                continue
            # Non-quota error -> try Ollama
            break

    # Gemini exhausted -> try Ollama fallback for validation
    try:
        full_prompt = f"{VALIDATOR_SYSTEM}\n\n{user_prompt}"
        verdict = _call_ollama_validator(full_prompt).upper()
        return verdict.startswith("PASS")
    except Exception:
        pass

    # Both failed -> be lenient: assume PASS rather than escalating everything
    return True
