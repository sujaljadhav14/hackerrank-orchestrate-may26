# HackerRank Orchestrate: Support Triage Agent

This is a fully automated AI Support Triage Agent built for the HackerRank Orchestrate hackathon. It processes support tickets, retrieves relevant documentation, and determines the appropriate response, status, and request type.

## Features
- **Multi-Stage Pipeline**: Safety checks -> Domain Routing -> Hybrid Retrieval (BM25 + Semantic) -> LLM Triage -> Validation.
- **Hybrid LLM Fallback**: Uses Google Gemini as the primary LLM, with local Ollama (Llama 3) as a fallback mechanism for resilience against rate limits and 429 ResourceExhausted errors.
- **Fast-path Routing**: Handles basic conversational tickets (e.g., "Thank you") immediately without wasting LLM quota.
- **Evaluation Framework**: Built-in accurate evaluator that tests against ground-truth samples based on issue text matching.

## Prerequisites
- Python 3.10+
- Google Gemini API Key (`GOOGLE_API_KEY`)
- Optional: Local Ollama running with the `llama3` model (for fallback)

## Setup
1. Copy the example `.env` file and add your API key:
   ```bash
   cp .env.example .env
   # Edit .env and set GOOGLE_API_KEY=your_key_here
   ```
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**1. Run the Full Pipeline**
Processes all 29 unlabeled tickets in `support_tickets/support_tickets.csv` and generates the final `output.csv`.
```bash
python code/main.py
```

**2. Run the Sample Evaluation**
Run the pipeline specifically against the 10 ground-truth sample tickets:
```bash
python code/main.py --input support_tickets/sample_support_tickets.csv --output support_tickets/sample_output.csv
```
Then, evaluate the accuracy:
```bash
python code/eval.py --predicted support_tickets/sample_output.csv
```

## Architecture
- `main.py`: Orchestrator tying all stages together.
- `agent.py`: LLM reasoning and structured JSON generation.
- `corpus.py`: Document chunking and indexing.
- `retrieval.py`: Hybrid BM25 + embedding-based semantic search.
- `safety.py`: Prompt injection and sensitive topic detection.
- `output.py`: Safe atomic file writes.