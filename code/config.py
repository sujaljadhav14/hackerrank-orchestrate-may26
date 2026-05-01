"""
config.py — Global constants for the support triage pipeline.
All configuration lives here; other modules import from this file only.
"""
import random
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── LLM ──────────────────────────────────────────────────────────────────────
MODEL = "models/gemini-2.0-flash"
MAX_TOKENS = 1024

# ── Ollama (local LLM fallback) ───────────────────────────────────────────────
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3"  # or gemma2, mistral, etc.

# ── Retry / backoff (for 429 ResourceExhausted) ───────────────────────────────
API_MAX_RETRIES = 3          # max retry attempts per API call
API_BACKOFF_BASE = 2.0       # exponential base (seconds)
API_BACKOFF_JITTER = 1.0     # random jitter added to each wait (seconds)
API_INTER_CALL_DELAY = 0.5   # seconds to sleep between LLM calls (burst throttle)

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 5
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.10   # below this → escalate without LLM call (lowered from 0.15 to reduce false escalations)
CHUNK_SIZE = 400                         # words per chunk
CHUNK_OVERLAP = 80                       # word overlap between chunks

# ── Safety / escalation ───────────────────────────────────────────────────────
ESCALATION_KEYWORDS = [
    # Financial / fraud
    "fraud", "unauthorized charge", "stolen card", "dispute", "chargeback",
    "refund", "billing issue", "account hacked", "suspicious transaction",
    # Account / security
    "account locked", "cannot login", "password reset", "2fa issue",
    "account suspended", "banned", "deleted account", "data breach",
    # Legal / sensitive
    "legal", "lawsuit", "attorney", "sue", "gdpr", "data deletion",
    "harassment", "abuse", "hate speech", "report user",
    # Assessment / exam integrity
    "cheating", "plagiarism", "exam violation", "unfair score", "assessment error",
    # Prompt injection signals
    "ignore previous instructions", "disregard your rules", "you are now",
    "pretend you are", "new instructions:", "system prompt", "jailbreak",
]

# ── Domain routing ────────────────────────────────────────────────────────────
DOMAIN_HINTS = {
    "hackerrank": [
        "test", "assessment", "coding challenge", "hire", "recruiter",
        "candidate", "score", "proctoring", "interview", "skill",
    ],
    "claude": [
        "claude", "anthropic", "ai assistant", "conversation", "prompt",
        "model", "context window", "subscription", "pro plan", "api",
    ],
    "visa": [
        "visa", "card", "payment", "transaction", "merchant", "atm",
        "debit", "credit", "bank", "checkout", "international fee",
    ],
}

# ── Output schema ─────────────────────────────────────────────────────────────
OUTPUT_COLUMNS = ["status", "product_area", "response", "justification", "request_type"]
VALID_STATUSES = {"replied", "escalated"}
VALID_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}
