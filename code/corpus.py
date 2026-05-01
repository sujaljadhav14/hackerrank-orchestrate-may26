"""
corpus.py — Document loader and chunker.

Walks data/hackerrank/, data/claude/, data/visa/ recursively.
Reads .txt, .md, .html, .json files.
Splits each document into overlapping word-window chunks.
Stores metadata: source domain, filename, chunk index.

Usage:
    from corpus import load_corpus
    chunks = load_corpus("data")
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from rich.console import Console

from config import CHUNK_SIZE, CHUNK_OVERLAP

console = Console()

# Supported file extensions and their text-extraction strategy
_SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".json"}

# Map top-level subdirectory name → domain label
_DOMAIN_MAP = {
    "hackerrank": "hackerrank",
    "claude": "claude",
    "visa": "visa",
}


@dataclass
class Chunk:
    """A single overlapping text window from a corpus document."""
    text: str
    domain: str        # "hackerrank" | "claude" | "visa"
    source_file: str   # relative path from data root
    chunk_idx: int     # 0-based index within the source file


# ── Text extraction ────────────────────────────────────────────────────────────

def _extract_text(file_path: Path) -> str:
    """Extract plain text from a file based on its extension."""
    ext = file_path.suffix.lower()

    if ext in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8", errors="replace")

    if ext == ".html":
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        # Minimal HTML → text: strip tags
        import re
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"&[a-zA-Z]+;", " ", text)   # basic HTML entities
        return text

    if ext == ".json":
        try:
            data = json.loads(file_path.read_text(encoding="utf-8", errors="replace"))
            # Flatten JSON to a string of key=value pairs
            return _flatten_json(data)
        except json.JSONDecodeError:
            return file_path.read_text(encoding="utf-8", errors="replace")

    return ""


def _flatten_json(obj, depth: int = 0) -> str:
    """Recursively flatten a JSON object to a readable string."""
    if depth > 6:
        return str(obj)
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            parts.append(f"{k}: {_flatten_json(v, depth + 1)}")
        return "\n".join(parts)
    if isinstance(obj, list):
        return "\n".join(_flatten_json(item, depth + 1) for item in obj)
    return str(obj)


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping word-window chunks.

    Args:
        text:    Input text (any length).
        size:    Number of words per chunk.
        overlap: Number of words to overlap between consecutive chunks.

    Returns:
        List of text chunks (may be empty if text has no words).
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    step = max(1, size - overlap)
    i = 0
    while i < len(words):
        window = words[i : i + size]
        chunks.append(" ".join(window))
        if i + size >= len(words):
            break
        i += step
    return chunks


# ── Corpus loader ──────────────────────────────────────────────────────────────

def load_corpus(data_dir: str = "data") -> List[Chunk]:
    """
    Walk data/{hackerrank,claude,visa}/ recursively.
    For each supported file, extract text and split into overlapping chunks.
    Print a per-domain chunk summary via rich.

    Args:
        data_dir: Path to the top-level data directory (absolute or relative).

    Returns:
        List of Chunk objects ordered by domain → file → chunk index.
    """
    root = Path(data_dir)
    all_chunks: List[Chunk] = []
    domain_counts: dict[str, int] = {d: 0 for d in _DOMAIN_MAP}

    for domain_dir_name, domain_label in _DOMAIN_MAP.items():
        domain_path = root / domain_dir_name
        if not domain_path.is_dir():
            console.print(f"  [yellow]⚠[/yellow]  data/{domain_dir_name}/ not found — skipping")
            continue

        for file_path in sorted(domain_path.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
                continue

            text = _extract_text(file_path)
            if not text.strip():
                continue

            rel_path = str(file_path.relative_to(root))
            text_chunks = chunk_text(text)

            for idx, chunk_text_str in enumerate(text_chunks):
                all_chunks.append(
                    Chunk(
                        text=chunk_text_str,
                        domain=domain_label,
                        source_file=rel_path,
                        chunk_idx=idx,
                    )
                )

            domain_counts[domain_label] += len(text_chunks)

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print("\n  [bold]Corpus loaded:[/bold]")
    for domain, count in domain_counts.items():
        console.print(f"    [cyan]{domain:12s}[/cyan]  {count:>5} chunks")
    console.print(f"    [green]{'TOTAL':12s}  {sum(domain_counts.values()):>5} chunks[/green]\n")

    return all_chunks
