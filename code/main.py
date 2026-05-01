#!/usr/bin/env python3
"""
main.py — HackerRank Orchestrate: Support Triage Agent (entry point).

5-stage pipeline per ticket:
  Stage 1: Safety check      (safety.py)   — keyword / injection detection
  Stage 2: Domain routing    (router.py)   — hackerrank / claude / visa / unknown
  Stage 3: Hybrid retrieval  (retriever.py)— BM25 + semantic top-k chunks
  Stage 4: LLM triage        (agent.py)    — Claude API → structured JSON
  Stage 5: Hallucination guard (validator.py) — second Claude call: PASS / FAIL

Run:
    python code/main.py
"""
from __future__ import annotations

import os
import sys
import random
import argparse
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Environment ───────────────────────────────────────────────────────────────
# Support both running from repo root AND from inside code/
_FILE_DIR = Path(__file__).parent
_REPO_ROOT = _FILE_DIR.parent

# Add code/ to sys.path so sibling modules resolve correctly regardless of CWD
if str(_FILE_DIR) not in sys.path:
    sys.path.insert(0, str(_FILE_DIR))

from dotenv import load_dotenv

# Load .env from repo root (preferred) then from code/ (fallback)
load_dotenv(_REPO_ROOT / ".env")
load_dotenv(_FILE_DIR / ".env")

if not os.environ.get("GOOGLE_API_KEY"):
    print(
        "ERROR: GOOGLE_API_KEY not set.\n"
        "Copy .env.example to .env and add your key:\n"
        "  cp code/.env.example .env\n"
        "  # Then edit .env and set GOOGLE_API_KEY=<your-google-ai-key>"
    )
    sys.exit(1)

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from corpus import load_corpus
from retriever import HybridRetriever
from safety import check_safety
from router import detect_domain
from agent import triage, set_ollama_mode
from validator import validate_response
from output import write_output
from config import RETRIEVAL_TOP_K, RETRIEVAL_CONFIDENCE_THRESHOLD

console = Console()

# ── Invalid-request fast-path patterns ───────────────────────────────────────
# These short, conversational, or clearly off-topic messages should be
# classified as `invalid` immediately without making an LLM call.
_INVALID_PHRASES = [
    "thank you", "thanks", "thank u", "thx", "ty ", "ty!",
    "you're welcome", "youre welcome", "no problem", "cheers",
    "great", "awesome", "perfect", "sounds good", "got it",
    "never mind", "nevermind", "forget it", "disregard",
]
_INVALID_MAX_WORDS = 8   # also flag very short messages (≤8 words) that match


def _quick_invalid_check(issue: str, subject: str) -> dict | None:
    """Return an 'invalid' result dict if the ticket is obviously out-of-scope."""
    combined = f"{subject} {issue}".lower().strip()
    word_count = len(combined.split())
    # Short message that matches a gratitude/closing pattern
    if word_count <= _INVALID_MAX_WORDS and any(p in combined for p in _INVALID_PHRASES):
        return {
            "status": "replied",
            "product_area": "conversation_management",
            "response": "Happy to help! Let us know if you have any other questions.",
            "justification": "Ticket is a conversational closing or thank-you message.",
            "request_type": "invalid",
        }
    return None

# ── Paths ─────────────────────────────────────────────────────────────────────
_DATA_DIR       = _REPO_ROOT / "data"
_TICKETS_PATH   = _REPO_ROOT / "support_tickets" / "support_tickets.csv"
_OUTPUT_PATH    = _REPO_ROOT / "support_tickets" / "output.csv"
_SAMPLE_PATH    = _REPO_ROOT / "support_tickets" / "sample_support_tickets.csv"


def main() -> None:
    # ── CLI arguments ─────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="HackerRank Orchestrate — Support Triage Agent")
    parser.add_argument(
        "--ollama",
        action="store_true",
        default=False,
        help="Use local Ollama LLM instead of Google Gemini API (no API key required)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to input tickets CSV (default: support_tickets/support_tickets.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write output CSV (default: support_tickets/output.csv)",
    )
    args = parser.parse_args()

    # Propagate Ollama mode to agent module
    if args.ollama:
        set_ollama_mode(True)
        console.print("[bold yellow]** Ollama mode enabled - using local LLM[/bold yellow]")

    console.print(Panel.fit(
        "[bold cyan]HackerRank Orchestrate - Support Triage Agent[/bold cyan]\n"
        "[dim]Pipeline: Safety -> Route -> Retrieve -> Reason -> Validate[/dim]",
        border_style="cyan",
    ))

    # ── 1. Load corpus ────────────────────────────────────────────────────────
    console.print("\n[bold]Loading corpus...[/bold]")
    chunks = load_corpus(str(_DATA_DIR))
    console.print(f"  [green]✓[/green] {len(chunks)} total chunks indexed")

    # ── 2. Build retriever ────────────────────────────────────────────────────
    console.print("\n[bold]Building hybrid retriever (BM25 + semantic)...[/bold]")
    retriever = HybridRetriever(chunks)
    console.print("  [green]✓[/green] Retriever ready\n")

    # ── 3. Load tickets ───────────────────────────────────────────────────────
    tickets_path = Path(args.input) if args.input else _TICKETS_PATH
    output_path  = Path(args.output) if args.output else _OUTPUT_PATH

    if not tickets_path.exists():
        console.print(f"[red]ERROR:[/red] Tickets CSV not found at {tickets_path}")
        sys.exit(1)

    df = pd.read_csv(tickets_path)
    # Input CSV may use title-cased headers (Issue, Subject, Company).
    # Normalize once so downstream row.get("issue") style access always works.
    df.columns = [str(c).strip().lower() for c in df.columns]
    console.print(f"[bold]Processing {len(df)} tickets...[/bold]\n")

    results: list[dict] = []
    stage_stats = {"safety_escalated": 0, "low_conf_escalated": 0, "replied": 0, "escalated": 0}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("[cyan]Triaging tickets...", total=len(df))

        for _, row in df.iterrows():
            issue   = str(row.get("issue", "") or "")
            subject = str(row.get("subject", "") or "")
            company = str(row.get("company", "") or "")

            # ── Invalid fast-path (before any API call) ───────────────────────
            invalid_result = _quick_invalid_check(issue, subject)
            if invalid_result:
                invalid_result["issue"] = issue
                results.append(invalid_result)
                stage_stats["replied"] = stage_stats.get("replied", 0) + 1
                progress.advance(task)
                continue

            # ── Stage 1: Safety ───────────────────────────────────────────────
            safety_result = check_safety(issue, subject)
            if safety_result:
                safety_result["issue"] = issue  # carry through for eval join
                results.append(safety_result)
                stage_stats["safety_escalated"] += 1
                progress.advance(task)
                continue

            # ── Stage 2: Route ────────────────────────────────────────────────
            domain = detect_domain(issue, subject, company)

            # ── Stage 3: Retrieve ─────────────────────────────────────────────
            query = f"{subject} {issue}".strip()
            retrieved_chunks, confidence = retriever.retrieve(
                query,
                domain_filter=domain if domain != "unknown" else None,
                top_k=RETRIEVAL_TOP_K,
            )

            # ── Stage 4: Triage (LLM) ─────────────────────────────────────────
            result = triage(issue, subject, domain, retrieved_chunks, confidence)

            # ── Stage 5: Validate (hallucination guard) ───────────────────────
            if result["status"] == "replied" and retrieved_chunks:
                passes = validate_response(
                    result["response"],
                    retrieved_chunks,
                    confidence=confidence,
                    use_ollama=args.ollama,
                )
                if not passes:
                    result["status"] = "escalated"
                    result["justification"] += (
                        " [Auto-escalated: response failed hallucination validation.]"
                    )
                    result["response"] = (
                        "Your request has been escalated to a human agent "
                        "for accurate, verified assistance."
                    )

            # Track stats
            if confidence < RETRIEVAL_CONFIDENCE_THRESHOLD:
                stage_stats["low_conf_escalated"] += 1
            stage_stats[result["status"]] = stage_stats.get(result["status"], 0) + 1

            result["issue"] = issue  # carry through for eval join
            results.append(result)
            progress.advance(task)

    # ── Write output ──────────────────────────────────────────────────────────
    df_out = write_output(results, str(output_path))

    # ── Summary table ─────────────────────────────────────────────────────────
    console.print()
    status_counts  = df_out["status"].value_counts()
    rtype_counts   = df_out["request_type"].value_counts()
    df_for_domains = pd.read_csv(tickets_path)
    df_for_domains.columns = [str(c).strip().lower() for c in df_for_domains.columns]
    domain_counts  = (
        df_for_domains
        .apply(
            lambda r: detect_domain(
                str(r.get("issue", "")),
                str(r.get("subject", "")),
                str(r.get("company", "")),
            ),
            axis=1,
        )
        .value_counts()
    )

    table = Table(title="Output Summary", show_header=True, header_style="bold magenta")
    table.add_column("Dimension", style="cyan")
    table.add_column("Value", style="white")
    table.add_column("Count", style="green", justify="right")

    for status, count in status_counts.items():
        table.add_row("status", status, str(count))
    table.add_section()
    for rtype, count in rtype_counts.items():
        table.add_row("request_type", rtype, str(count))
    table.add_section()
    for domain, count in domain_counts.items():
        table.add_row("domain", domain, str(count))
    table.add_section()
    table.add_row("safety escalated", "", str(stage_stats["safety_escalated"]))

    console.print(table)
    console.print(
        f"\n[bold green]✓ Done![/bold green]  "
        f"Output written to [cyan]{output_path}[/cyan]"
    )
    if args.input and "sample" in str(args.input):
        console.print(
            f"\n[dim]Run self-evaluation:[/dim]  "
            f"[bold]python code/eval.py --predicted {output_path}[/bold]\n"
        )
    else:
        console.print(
            "\n[dim]Run self-evaluation (sample):[/dim]  "
            "[bold]python code/main.py --input support_tickets/sample_support_tickets.csv "
            "--output support_tickets/sample_output.csv[/bold]\n"
            "[dim]Then:[/dim]  "
            "[bold]python code/eval.py --predicted support_tickets/sample_output.csv[/bold]\n"
        )


if __name__ == "__main__":
    main()
