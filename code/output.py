"""
output.py — Writes the final predictions to output.csv.

Ensures the column order matches the evaluation contract exactly:
  status, product_area, response, justification, request_type

Uses a write-to-temp-then-rename strategy so a locked file (open in an editor)
never crashes the pipeline — it falls back to writing a timestamped copy instead.
"""
from __future__ import annotations

import os
import tempfile
import shutil
from pathlib import Path

import pandas as pd

from config import OUTPUT_COLUMNS


def write_output(rows: list[dict], path: str = "support_tickets/output.csv") -> pd.DataFrame:
    """
    Write the list of result dicts to a CSV file.

    Args:
        rows: List of dicts, each with keys matching OUTPUT_COLUMNS.
        path: Destination CSV path (created/overwritten).

    Returns:
        The resulting DataFrame (useful for logging summaries).
    """
    # Only keep the 5 required columns in the exact contract order
    sanitized = []
    for row in rows:
        sanitized.append({col: row.get(col, "") for col in OUTPUT_COLUMNS})

    df = pd.DataFrame(sanitized, columns=OUTPUT_COLUMNS)

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file in the same directory, then rename atomically.
    # This prevents partial writes AND handles the case where the destination
    # is open/locked in an editor (Windows locks open files).
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
        try:
            os.close(tmp_fd)
            df.to_csv(tmp_path, index=False, encoding="utf-8")
            # Atomic replace — overwrites destination even if it exists
            shutil.move(tmp_path, dest)
        except Exception:
            # Clean up temp file if rename failed
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except PermissionError:
        # Destination is locked (open in editor) — write a timestamped fallback copy
        from datetime import datetime
        ts = datetime.now().strftime("%H%M%S")
        fallback = dest.parent / f"output_{ts}.csv"
        df.to_csv(str(fallback), index=False, encoding="utf-8")
        print(
            f"\n[WARNING] Could not write to {dest} (file is locked/open in editor).\n"
            f"          Results saved to fallback: {fallback}\n"
            f"          Close output.csv in your editor and rename the fallback file.\n"
        )

    return df
