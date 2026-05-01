# Support Triage Agent

Multi-stage AI agent for the **HackerRank Orchestrate** hackathon (May 2026).

---

## Architecture

```
[CSV Row]
    │
    ▼
[Stage 1: safety.py]      — keyword/injection check (deterministic, free)
    │ safe
    ▼
[Stage 2: router.py]      — domain detection: hackerrank / claude / visa / unknown
    │
    ▼
[Stage 3: retriever.py]   — hybrid BM25 + semantic top-k chunks
    │                        confidence < 0.25 → ESCALATE (no LLM call)
    ▼
[Stage 4: agent.py]       — Claude API → JSON {status, product_area, response, ...}
    │
    ▼
[Stage 5: validator.py]   — second Claude call: PASS / FAIL grounding check
    │                        FAIL → escalate + safe fallback response
    ▼
[output.csv row]
```

---

## Install

```bash
pip install -r code/requirements.txt
```

> The sentence-transformers model (`all-MiniLM-L6-v2`, ~22 MB) is downloaded
> automatically on first run and cached locally.

---

## Configure

```bash
# Copy the example and fill in your key
cp code/.env.example .env
# Edit .env and set:
#   ANTHROPIC_API_KEY=sk-ant-...
```

---

## Run

```bash
python code/main.py
```

Output is written to `support_tickets/output.csv`.

---

## Self-evaluate (against sample CSV)

```bash
python code/eval.py
```

Optional flags:

```bash
python code/eval.py \
  --predicted   support_tickets/output.csv \
  --ground-truth support_tickets/sample_support_tickets.csv
```

---

## File map

| File | Role |
|------|------|
| `main.py` | Entry point — orchestrates the 5-stage pipeline |
| `config.py` | Constants: model, thresholds, keywords, schema |
| `corpus.py` | Recursive document loader + overlapping chunker |
| `retriever.py` | Hybrid BM25 + semantic retrieval (local, no API) |
| `safety.py` | Stage 1: adversarial / injection / high-risk guard |
| `router.py` | Stage 2: domain detection (HackerRank / Claude / Visa) |
| `agent.py` | Stage 3 + 4: Claude API triage → structured JSON |
| `validator.py` | Stage 5: hallucination guard via second Claude call |
| `output.py` | Writes output.csv with correct column contract |
| `eval.py` | Self-evaluation against sample_support_tickets.csv |
| `requirements.txt` | Pinned dependencies |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Hybrid retrieval (BM25 + semantic)** | BM25 handles exact keyword / error-code matches; semantic handles paraphrased queries. Neither alone is best. |
| **5-stage pipeline** | Separates safety, routing, retrieval, generation, and validation as independent, testable, explainable concerns. |
| **Pre-LLM safety layer** | Keyword-based escalation runs before any API call — fraud/injection handling is deterministic and free. |
| **Confidence-based escalation** | If retrieval confidence < 0.25, we escalate rather than hallucinate an answer. |
| **Hallucination validator** | A second LLM call verifies the response is grounded in retrieved context — the biggest differentiator. |
| **Claude API (not Ollama)** | Structured JSON output quality from local models is inconsistent. Claude returns valid JSON reliably, critical for 50+ rows. |
| **No web calls** | All knowledge comes from the local `data/` corpus, as required by the problem statement. |

---

## Known Limitations

- Multi-request tickets (user asks two independent questions in one message)
- `company=None` tickets with ambiguous language may be misrouted
- Non-English tickets (one French/Spanish Visa example in the dataset)

---

## Submission Checklist

- [ ] `python code/main.py` runs without errors
- [ ] `support_tickets/output.csv` has same row count as `support_tickets/support_tickets.csv`
- [ ] All `status` values are `replied` or `escalated`
- [ ] All `request_type` values are `product_issue`, `feature_request`, `bug`, or `invalid`
- [ ] No API keys in any committed file
- [ ] `python code/eval.py` shows reasonable accuracy
