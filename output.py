"""
Output module: writes triage results to CSV and SQLite.

CSV is the human-readable output you'll review.
SQLite stores the same data plus embedding vectors for later
clustering, similarity search, and deduplication.
"""

import csv
import json
import logging
import os
import sqlite3
from pathlib import Path

from processor import TriageResult
import config

logger = logging.getLogger(__name__)

# CSV column order
CSV_COLUMNS = [
    "url",
    "title",
    "summary",
    "category",
    "actionability",
    "implied_action",
    "importance",
    "effort",
    "staleness",
    "insight_density",
    "quick_win_score",
    "fetch_success",
    "fetch_error",
]


def ensure_output_dir():
    """Create the output directory if it doesn't exist."""
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def write_csv(results: list[TriageResult], filename: str = None,
              sort_by: str = "quick_win_score", descending: bool = True):
    """
    Write triage results to a CSV file, sorted by the specified column.

    Args:
        results: List of TriageResult objects.
        filename: Output filename (default from config).
        sort_by: Column name to sort by.
        descending: Sort in descending order (best first).
    """
    ensure_output_dir()
    filepath = os.path.join(config.OUTPUT_DIR, filename or config.CSV_FILENAME)

    # Convert to dicts and sort
    rows = [r.to_csv_dict() for r in results]
    if sort_by and sort_by in CSV_COLUMNS:
        rows.sort(key=lambda r: r.get(sort_by, 0), reverse=descending)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote {len(rows)} results to {filepath}")
    return filepath


def write_sqlite(results: list[TriageResult], filename: str = None):
    """
    Write triage results to a SQLite database, including embeddings.

    The embeddings are stored as JSON-encoded arrays in a TEXT column.
    This isn't the most space-efficient, but it's simple, portable,
    and easy to query with Python.

    Args:
        results: List of TriageResult objects.
        filename: Output filename (default from config).
    """
    ensure_output_dir()
    filepath = os.path.join(config.OUTPUT_DIR, filename or config.SQLITE_FILENAME)

    conn = sqlite3.connect(filepath)
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tabs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            title TEXT,
            summary TEXT,
            category TEXT,
            actionability INTEGER,
            implied_action TEXT,
            importance INTEGER,
            effort INTEGER,
            staleness INTEGER,
            insight_density INTEGER,
            quick_win_score REAL,
            fetch_success BOOLEAN,
            fetch_error TEXT,
            embedding TEXT,
            llm_raw_response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create useful indexes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_category ON tabs(category)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_actionability ON tabs(actionability)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_quick_win ON tabs(quick_win_score DESC)
    """)

    # Insert or replace results
    for r in results:
        embedding_json = json.dumps(r.embedding) if r.embedding else None

        cursor.execute("""
            INSERT OR REPLACE INTO tabs
            (url, title, summary, category, actionability, implied_action,
             importance, effort, staleness, insight_density, quick_win_score,
             fetch_success, fetch_error, embedding, llm_raw_response)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r.url, r.title, r.summary, r.category,
            r.actionability, r.implied_action,
            r.importance, r.effort, r.staleness, r.insight_density,
            r.quick_win_score,
            r.fetch_success, r.fetch_error,
            embedding_json, r.llm_raw_response,
        ))

    conn.commit()

    # Report stats
    cursor.execute("SELECT COUNT(*) FROM tabs")
    total = cursor.fetchone()[0]
    logger.info(f"Wrote {len(results)} results to {filepath} ({total} total rows)")

    conn.close()
    return filepath


def print_summary(results: list[TriageResult]):
    """Print a human-readable summary of the triage results to stdout."""
    total = len(results)
    successful = sum(1 for r in results if r.fetch_success)
    failed = total - successful

    # Category breakdown
    categories = {}
    for r in results:
        cat = r.category or "uncategorized"
        categories[cat] = categories.get(cat, 0) + 1

    # Actionability distribution
    action_dist = {i: 0 for i in range(1, 6)}
    for r in results:
        action_dist[r.actionability] = action_dist.get(r.actionability, 0) + 1

    # Top quick wins
    quick_wins = sorted(
        [r for r in results if r.actionability >= 3],
        key=lambda r: r.quick_win_score,
        reverse=True,
    )[:10]

    # Top reads (low actionability, high insight density)
    best_reads = sorted(
        [r for r in results if r.actionability <= 2],
        key=lambda r: r.insight_density,
        reverse=True,
    )[:10]

    print("\n" + "=" * 70)
    print(f"  TAB TRIAGE RESULTS — {total} tabs processed")
    print("=" * 70)

    print(f"\n  Fetched successfully: {successful}")
    print(f"  Failed to fetch:     {failed}")

    print(f"\n  Categories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"    {cat:30s}  {count:3d}")

    print(f"\n  Actionability distribution:")
    labels = {1: "Just reading", 2: "Vague inspiration", 3: "Specific advice",
              4: "Clear task", 5: "Action item itself"}
    for score in range(1, 6):
        count = action_dist[score]
        bar = "█" * count
        print(f"    {score} ({labels[score]:20s}): {count:3d}  {bar}")

    if quick_wins:
        print(f"\n  Top Quick Wins (high importance, low effort):")
        for i, r in enumerate(quick_wins[:5], 1):
            print(f"    {i}. [{r.quick_win_score:.1f}] {r.title[:55]}")
            print(f"       → {r.implied_action[:65]}")

    if best_reads:
        print(f"\n  Best Reads (high insight density):")
        for i, r in enumerate(best_reads[:5], 1):
            print(f"    {i}. [insight={r.insight_density}/5] {r.title[:55]}")
            print(f"       {r.summary[:65]}...")

    print("\n" + "=" * 70)
