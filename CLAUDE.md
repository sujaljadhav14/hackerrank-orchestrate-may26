# CLAUDE.md — HackerRank Orchestrate (May 2026)

> Read AGENTS.md first for logging rules and onboarding. This file extends it with the full build specification.
> This file is written for Google Antigravity, Claude Code, Cursor, or any AGENTS.md-aware coding tool.

---

## 0. MISSION SUMMARY

Build a **terminal-based, multi-stage AI support triage agent** in Python that:
1. Reads every row from `support_issues/support_issues.csv`
2. Classifies, retrieves, reasons, and responds using ONLY the local corpus in `data/`
3. Writes predictions to `support_issues/output.csv` with 5 columns: `status`, `product_area`, `response`, `justification`, `request_type`
4. Uses the **Anthropic Claude API** (`claude-sonnet-4-20250514`) for LLM reasoning
5. Uses `sentence-transformers` + `scikit-learn` for local hybrid retrieval — **no web calls for knowledge**

Winning edge: most participants will do one LLM call per ticket. We run a **4-stage pipeline** with separate sub-agents for each concern. This is what the judges are looking for under "Agent Design."

---

## 1. REPO SETUP (DO THIS FIRST)

```bash
# Step 1 — Fork the repo on GitHub, then clone YOUR fork
git clone git@github.com:YOUR_USERNAME/hackerrank-orchestrate-may26.git
cd hackerrank-orchestrate-may26

# Step 2 — Create .env from example
cp .env.example .env
# Add ANTHROPIC_API_KEY=sk-ant-... to .env

# Step 3 — Replace CLAUDE.md in repo root with THIS file
# (already done if you're reading this)

# Step 4 — Explore the data before building anything
ls data/hackerrank/
ls data/claude/
ls data/visa/
head -5 support_issues/sample_support_issues.csv
```

**Important:** Do NOT modify anything outside `code/` and `support_issues/output.csv`. Leave `data/`, `support_issues/support_issues.csv`, `AGENTS.md`, `README.md`, `problem_statement.md`, and `evalutation_criteria.md` exactly as they are.

---

## 2. EXACT FILE STRUCTURE TO BUILD

Build all files inside `code/`. Do not deviate from these filenames.

```
code/
├── main.py              # Entry point — orchestrates the pipeline
├── corpus.py            # Loads + chunks all docs from data/
├── retriever.py         # Hybrid BM25 + semantic retrieval
├── safety.py            # Stage 1: adversarial / validity check
├── router.py            # Stage 2: domain detection (HackerRank / Claude / Visa)
├── agent.py             # Stage 3 + 4: Claude API calls for triage + response
├── validator.py         # Stage 5: hallucination guard on generated response
├── output.py            # Writes output.csv with correct column order
├── eval.py              # Self-evaluation against sample_support_issues.csv
├── config.py            # Constants, model name, thresholds
├── requirements.txt     # Pinned deps
├── .env.example         # ANTHROPIC_API_KEY=
└── README.md            # How to install and run
```

---

## 3. TECH STACK (PINNED)

```
anthropic==0.50.0
sentence-transformers==3.4.1
scikit-learn==1.6.1
rank-bm25==0.2.2
pandas==2.2.3
python-dotenv==1.1.0
rich==14.0.0
numpy==2.2.5
```

Write `code/requirements.txt` with exactly these. Run: `pip install -r code/requirements.txt`

---

## 4. ARCHITECTURE: THE 5-STAGE PIPELINE

Every row in the CSV goes through ALL 5 stages in sequence. Each stage is a separate function/class.

```
[CSV Row]
    │
    ▼
[Stage 1: safety.py]  ──────────────────────────────────►  ESCALATE if malicious/jailbreak
    │ (safe)
    ▼
[Stage 2: router.py]  — detect which domain: hackerrank / claude / visa / unknown
    │
    ▼
[Stage 3: retriever.py]  — hybrid BM25 + semantic → top 5 doc chunks
    │                         if retrieval_confidence < 0.25 → ESCALATE
    ▼
[Stage 4: agent.py]  — Claude API call → JSON output (status, product_area, response, justification, request_type)
    │
    ▼
[Stage 5: validator.py]  — second Claude call: "Does this response cite only provided context?"
    │                         if hallucination_detected → override response with safe fallback
    ▼
[output.csv row]
```

This pipeline is what separates you from every basic RAG submission.

---

## 5. DETAILED SPEC PER FILE

### 5.1 `config.py`

```python
import random
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024
RETRIEVAL_TOP_K = 5
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.25  # below this → escalate
CHUNK_SIZE = 400          # words per chunk
CHUNK_OVERLAP = 80        # word overlap between chunks

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

DOMAIN_HINTS = {
    "hackerrank": ["test", "assessment", "coding challenge", "hire", "recruiter",
                   "candidate", "score", "proctoring", "interview", "skill"],
    "claude": ["claude", "anthropic", "ai assistant", "conversation", "prompt",
               "model", "context window", "subscription", "pro plan", "api"],
    "visa": ["visa", "card", "payment", "transaction", "merchant", "atm",
             "debit", "credit", "bank", "checkout", "international fee"],
}

OUTPUT_COLUMNS = ["status", "product_area", "response", "justification", "request_type"]
VALID_STATUSES = {"replied", "escalated"}
VALID_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}
```

---

### 5.2 `corpus.py`

Load every `.txt`, `.md`, `.html`, `.json` file recursively from `data/`. Chunk each document into overlapping windows of `CHUNK_SIZE` words. Store metadata: source domain, filename, chunk index.

```python
# corpus.py skeleton
from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class Chunk:
    text: str
    domain: str        # "hackerrank" | "claude" | "visa"
    source_file: str
    chunk_idx: int

def load_corpus(data_dir: str = "data") -> List[Chunk]:
    """
    Walk data/hackerrank/, data/claude/, data/visa/.
    For each file, split into overlapping word-window chunks.
    Return list of Chunk objects.
    """
    ...

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Split text into word-window chunks with overlap."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks
```

**Important:** Precompute and cache the corpus on first load. Print chunk count per domain using `rich.console` so the user can verify at startup.

---

### 5.3 `retriever.py`

Hybrid retrieval: BM25 score + cosine similarity from `sentence-transformers`, averaged with equal weight. Most competitors will only use TF-IDF — this is a key differentiator.

```python
# retriever.py skeleton
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from corpus import Chunk
from config import RETRIEVAL_TOP_K, RANDOM_SEED

class HybridRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.texts = [c.text for c in chunks]
        
        # BM25
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)
        
        # Semantic embeddings (runs fully local, no API call)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.encoder.encode(self.texts, show_progress_bar=True)

    def retrieve(self, query: str, domain_filter: str | None = None, top_k: int = RETRIEVAL_TOP_K):
        """
        Returns (chunks, confidence_score).
        confidence_score is the mean similarity of top-k results.
        If domain_filter is set, boost chunks from that domain by 1.2x.
        """
        # BM25 scores
        bm25_scores = np.array(self.bm25.get_scores(query.lower().split()))
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)

        # Semantic scores
        q_emb = self.encoder.encode([query])
        sem_scores = cosine_similarity(q_emb, self.embeddings)[0]

        # Hybrid
        hybrid = 0.5 * bm25_scores + 0.5 * sem_scores

        # Domain boost
        if domain_filter:
            for i, chunk in enumerate(self.chunks):
                if chunk.domain == domain_filter:
                    hybrid[i] *= 1.2

        top_idx = np.argsort(hybrid)[::-1][:top_k]
        results = [self.chunks[i] for i in top_idx]
        confidence = float(np.mean([hybrid[i] for i in top_idx]))
        return results, confidence
```

---

### 5.4 `safety.py`

**Stage 1** — runs BEFORE any retrieval or LLM call. Fast, deterministic, keyword-based.

```python
# safety.py
from config import ESCALATION_KEYWORDS

def check_safety(issue: str, subject: str) -> dict | None:
    """
    Returns None if safe.
    Returns a pre-built escalation row dict if unsafe.
    """
    combined = f"{subject} {issue}".lower()
    
    # Prompt injection detection
    injection_signals = [
        "ignore previous", "disregard your", "you are now",
        "pretend you are", "new instructions:", "system prompt", "jailbreak"
    ]
    for signal in injection_signals:
        if signal in combined:
            return {
                "status": "escalated",
                "product_area": "security",
                "response": "This request has been flagged for security review and escalated to our team.",
                "justification": "Input contains prompt injection signals and cannot be processed automatically.",
                "request_type": "invalid",
            }
    
    # High-risk keyword escalation
    high_risk = [
        "fraud", "unauthorized charge", "stolen", "account hacked",
        "data breach", "legal", "lawsuit", "gdpr", "data deletion",
    ]
    for kw in high_risk:
        if kw in combined:
            return {
                "status": "escalated",
                "product_area": "security_and_compliance",
                "response": "This case involves sensitive account or financial matter and has been escalated for human review.",
                "justification": f"Ticket contains high-risk keyword '{kw}' requiring human agent review.",
                "request_type": "product_issue",
            }
    
    return None  # safe to proceed
```

---

### 5.5 `router.py`

**Stage 2** — domain detection. Uses the `company` field first, then falls back to keyword matching, then asks the LLM only if still ambiguous.

```python
# router.py
from config import DOMAIN_HINTS
import re

def detect_domain(issue: str, subject: str, company: str) -> str:
    """Returns 'hackerrank' | 'claude' | 'visa' | 'unknown'"""
    
    # Direct company field
    if company and company.strip().lower() in {"hackerrank", "claude", "visa"}:
        return company.strip().lower()
    
    # Keyword matching
    combined = f"{subject} {issue}".lower()
    scores = {}
    for domain, hints in DOMAIN_HINTS.items():
        scores[domain] = sum(1 for h in hints if h in combined)
    
    best_domain = max(scores, key=scores.get)
    if scores[best_domain] > 0:
        return best_domain
    
    return "unknown"
```

---

### 5.6 `agent.py`

**Stages 3 + 4** — the main Claude API call. Takes retrieved chunks + ticket, returns structured JSON with all 5 fields.

```python
# agent.py
import anthropic
import json
import os
from config import MODEL, MAX_TOKENS
from retriever import HybridRetriever
from corpus import Chunk

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SYSTEM_PROMPT = """You are a professional support triage agent for a multi-product company.
You have been given a support ticket and relevant excerpts from the official support documentation.

YOUR RULES (NON-NEGOTIABLE):
1. Base ALL responses ONLY on the provided documentation excerpts. Never use outside knowledge.
2. If the documentation does not clearly answer the question, set status to "escalated".
3. Never invent policies, prices, timelines, or features not in the provided context.
4. If the ticket is irrelevant, spam, or completely outside the products' scope, set request_type to "invalid".
5. Escalate any case involving billing disputes, account security, legal matters, or bugs you cannot confirm are documented.

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown fences, no explanation:
{
  "status": "replied" | "escalated",
  "product_area": "<most relevant support category from the documentation>",
  "response": "<user-facing response, 2-4 sentences, grounded in provided context>",
  "justification": "<1-2 sentences explaining your routing and response decision>",
  "request_type": "product_issue" | "feature_request" | "bug" | "invalid"
}"""

def triage(issue: str, subject: str, domain: str, chunks: list[Chunk], confidence: float) -> dict:
    """
    Main triage call. Returns dict with 5 output fields.
    If confidence is too low, returns escalation without calling LLM.
    """
    from config import RETRIEVAL_CONFIDENCE_THRESHOLD
    
    if confidence < RETRIEVAL_CONFIDENCE_THRESHOLD:
        return {
            "status": "escalated",
            "product_area": domain if domain != "unknown" else "general",
            "response": "We could not find sufficient documentation to address this request. A human agent will follow up.",
            "justification": f"Retrieval confidence {confidence:.2f} below threshold {RETRIEVAL_CONFIDENCE_THRESHOLD}. Insufficient context to answer safely.",
            "request_type": "product_issue",
        }
    
    context_block = "\n\n---\n\n".join([
        f"[Source: {c.domain}/{c.source_file}, chunk {c.chunk_idx}]\n{c.text}"
        for c in chunks
    ])
    
    user_message = f"""TICKET
Subject: {subject or '(none)'}
Company: {domain}
Issue:
{issue}

DOCUMENTATION CONTEXT:
{context_block}

Now produce the JSON output."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    
    raw = response.content[0].text.strip()
    
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract JSON from response
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            result = {
                "status": "escalated",
                "product_area": domain,
                "response": "Unable to generate a structured response. Escalating for human review.",
                "justification": "LLM response parsing failed.",
                "request_type": "product_issue",
            }
    
    # Enforce allowed values
    if result.get("status") not in {"replied", "escalated"}:
        result["status"] = "escalated"
    if result.get("request_type") not in {"product_issue", "feature_request", "bug", "invalid"}:
        result["request_type"] = "product_issue"
    
    return result
```

---

### 5.7 `validator.py`

**Stage 5** — hallucination guard. Makes a SECOND short Claude call to verify the generated response is grounded. This is the biggest differentiator — almost no participant will do this.

```python
# validator.py
import anthropic
import os
from config import MODEL

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

VALIDATOR_SYSTEM = """You are a fact-checking assistant for AI-generated support responses.
Given a support response and the documentation context it was based on, answer ONLY with "PASS" or "FAIL".
Answer "FAIL" if the response makes ANY claim not supported by the provided context.
Answer "PASS" if every factual claim in the response is traceable to the provided context."""

def validate_response(response_text: str, context_chunks: list, max_tokens: int = 10) -> bool:
    """Returns True if response passes hallucination check, False if it fails."""
    context = "\n\n".join([c.text for c in context_chunks[:3]])  # top 3 chunks
    
    verdict = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=VALIDATOR_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nRESPONSE TO CHECK:\n{response_text}\n\nPass or Fail?"
        }]
    )
    
    result = verdict.content[0].text.strip().upper()
    return result.startswith("PASS")
```

---

### 5.8 `output.py`

```python
# output.py
import pandas as pd
from config import OUTPUT_COLUMNS

def write_output(rows: list[dict], path: str = "support_issues/output.csv"):
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df.to_csv(path, index=False)
    return df
```

---

### 5.9 `eval.py`

Self-evaluation against `sample_support_issues.csv`. Run this to benchmark before submitting. Uses exact-match on `status` and `request_type`, prints a summary table with `rich`.

```python
# eval.py — run with: python code/eval.py
import pandas as pd
from rich.console import Console
from rich.table import Table

def evaluate(predicted_path: str, ground_truth_path: str):
    console = Console()
    pred = pd.read_csv(predicted_path)
    gt = pd.read_csv(ground_truth_path)
    
    # Match on status and request_type (exact)
    status_acc = (pred["status"] == gt["status"]).mean()
    rtype_acc = (pred["request_type"] == gt["request_type"]).mean()
    
    table = Table(title="Self-Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green")
    table.add_row("Status accuracy", f"{status_acc:.1%}")
    table.add_row("Request type accuracy", f"{rtype_acc:.1%}")
    console.print(table)
```

---

### 5.10 `main.py`

Orchestrates everything. Uses `rich` for a beautiful progress display.

```python
#!/usr/bin/env python3
"""
HackerRank Orchestrate — Support Triage Agent
Run: python code/main.py
"""
import os
import sys
import random
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

# Seeding for determinism
random.seed(42)
np.random.seed(42)

load_dotenv()
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
    sys.exit(1)

import pandas as pd
from corpus import load_corpus
from retriever import HybridRetriever
from safety import check_safety
from router import detect_domain
from agent import triage
from validator import validate_response
from output import write_output
from config import RETRIEVAL_TOP_K

console = Console()

def main():
    console.print(Panel.fit(
        "[bold cyan]HackerRank Orchestrate — Support Triage Agent[/bold cyan]\n"
        "[dim]Multi-stage pipeline: Safety → Route → Retrieve → Reason → Validate[/dim]",
        border_style="cyan"
    ))

    # 1. Load corpus
    console.print("\n[bold]Loading corpus...[/bold]")
    repo_root = Path(__file__).parent.parent
    chunks = load_corpus(str(repo_root / "data"))
    console.print(f"  [green]✓[/green] Loaded {len(chunks)} chunks")

    # 2. Build retriever
    console.print("[bold]Building hybrid retriever (BM25 + semantic)...[/bold]")
    retriever = HybridRetriever(chunks)
    console.print("  [green]✓[/green] Retriever ready")

    # 3. Load tickets
    tickets_path = repo_root / "support_issues" / "support_issues.csv"
    df = pd.read_csv(tickets_path)
    console.print(f"\n[bold]Processing {len(df)} tickets...[/bold]\n")

    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Triaging...", total=len(df))
        
        for _, row in df.iterrows():
            issue = str(row.get("issue", ""))
            subject = str(row.get("subject", ""))
            company = str(row.get("company", "None"))

            # Stage 1: Safety
            safety_result = check_safety(issue, subject)
            if safety_result:
                results.append(safety_result)
                progress.advance(task)
                continue

            # Stage 2: Route
            domain = detect_domain(issue, subject, company)

            # Stage 3: Retrieve
            query = f"{subject} {issue}".strip()
            retrieved_chunks, confidence = retriever.retrieve(
                query, domain_filter=domain if domain != "unknown" else None, top_k=RETRIEVAL_TOP_K
            )

            # Stage 4: Triage (LLM)
            result = triage(issue, subject, domain, retrieved_chunks, confidence)

            # Stage 5: Validate (hallucination guard)
            if result["status"] == "replied":
                passes = validate_response(result["response"], retrieved_chunks)
                if not passes:
                    result["status"] = "escalated"
                    result["justification"] += " [Escalated: response failed hallucination validation]"
                    result["response"] = "Your request has been escalated to a human agent for accurate assistance."

            results.append(result)
            progress.advance(task)

    # Write output
    output_path = repo_root / "support_issues" / "output.csv"
    df_out = write_output(results, str(output_path))
    
    # Summary
    status_counts = df_out["status"].value_counts()
    rtype_counts = df_out["request_type"].value_counts()
    
    table = Table(title="Output Summary")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_column("Count", style="green")
    for status, count in status_counts.items():
        table.add_row("status", status, str(count))
    for rtype, count in rtype_counts.items():
        table.add_row("request_type", rtype, str(count))
    console.print(table)
    
    console.print(f"\n[bold green]✓ Done! Output written to:[/bold green] {output_path}")

if __name__ == "__main__":
    main()
```

---

### 5.11 `code/README.md`

```markdown
# Support Triage Agent

Multi-stage AI agent for the HackerRank Orchestrate hackathon (May 2026).

## Architecture

5-stage pipeline: Safety Check → Domain Routing → Hybrid RAG Retrieval → Claude API Triage → Hallucination Validation

## Install

```bash
pip install -r code/requirements.txt
```

## Configure

```bash
cp code/.env.example .env
# Add: ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
python code/main.py
```

Output is written to `support_issues/output.csv`.

## Self-evaluate (against sample CSV)

```bash
python code/eval.py
```

## Design Decisions

- **Hybrid retrieval (BM25 + semantic)**: Combining lexical and semantic search gives better recall than either alone.
- **4-stage pipeline**: Separates safety, routing, retrieval, and generation as independent, testable concerns.
- **Pre-LLM safety layer**: Keyword-based escalation runs before any API call, making fraud/injection handling deterministic and free.
- **Confidence-based escalation**: If retrieval confidence is below 0.25, we escalate rather than hallucinate an answer.
- **Hallucination validator**: A second LLM call checks that the generated response is grounded in retrieved context.
- **Ollama**: Not used. Claude API provides substantially better structured JSON output quality.
- **No web calls**: All knowledge comes from the local `data/` corpus, as required by the problem statement.
```

---

## 6. WHAT NOT TO DO

- **Do NOT make web requests** during ticket processing. No `requests.get()`, no live lookups.
- **Do NOT hardcode the API key**. Read it only from `os.environ["ANTHROPIC_API_KEY"]`.
- **Do NOT modify files outside `code/`** except `support_issues/output.csv`.
- **Do NOT use Ollama** for generation. Use the Claude API. Ollama quality is insufficient for structured output.
- **Do NOT put everything in main.py**. The judges read the code and look for separation of concerns.
- **Do NOT skip the `code/README.md`**. It is required for scoring.

---

## 7. DIFFERENTIATORS (WHY THIS BEATS OTHER SUBMISSIONS)

Most competitors will:
- Use TF-IDF only → we use **hybrid BM25 + semantic** (better recall)
- Make one LLM call per ticket → we use a **5-stage pipeline** (explainable, debuggable)
- Skip safety checking → we have **keyword + injection detection** before any LLM call
- Trust the LLM blindly → we **validate for hallucinations** with a second call
- Use plain print() → we use **rich** for beautiful terminal output with progress bars
- Skip self-evaluation → we have **eval.py** to score against the sample CSV

These choices map directly to the four judging dimensions:
- **Agent Design**: Multi-module, principled architecture
- **Output CSV**: Higher accuracy through better retrieval + validation
- **AI Judge Interview**: Clear decisions to explain for every design choice
- **AI Fluency**: This CLAUDE.md itself is evidence of structured AI collaboration

---

## 8. SUBMISSION CHECKLIST

Before submitting on HackerRank:

- [ ] `python code/main.py` runs without errors
- [ ] `support_issues/output.csv` has same row count as `support_issues/support_issues.csv`
- [ ] `output.csv` has columns: `status, product_area, response, justification, request_type`
- [ ] All status values are `replied` or `escalated`
- [ ] All request_type values are `product_issue`, `feature_request`, `bug`, or `invalid`
- [ ] `code/README.md` explains how to install and run
- [ ] No API keys in any file
- [ ] `python code/eval.py` shows reasonable accuracy on sample CSV
- [ ] `$HOME/hackerrank_orchestrate/log.txt` exists and has your conversation history

### Upload to HackerRank:
1. **Code zip** — zip the `code/` directory only. Exclude `.env`, `__pycache__`, any venv.
2. **Predictions CSV** — upload `support_issues/output.csv`
3. **Chat transcript** — upload `$HOME/hackerrank_orchestrate/log.txt`

---

## 9. AI JUDGE INTERVIEW PREP

The AI judge will ask you about every decision in this file. Know these answers:

- **Why hybrid retrieval?** BM25 handles exact keyword matching (product names, error codes); semantic handles paraphrased queries. Neither alone is best.
- **Why escalate on low retrieval confidence?** Hallucinating an answer is worse than admitting uncertainty. Low confidence means the corpus doesn't cover this — a human should handle it.
- **Why a separate validator call?** The triage LLM can still hallucinate even with context. A second call is cheap and catches the most common failure mode.
- **Why Claude API over Ollama?** Structured JSON output quality from local models is inconsistent. Claude returns valid JSON reliably, which is critical when parsing 100+ rows.
- **Where does your agent fail?** Multi-request tickets (user asks two questions), company=None with ambiguous content, tickets in non-English.
- **How would you improve it?** Add a re-ranker after retrieval, fine-tune product area taxonomy, add caching so repeat queries don't double API costs.