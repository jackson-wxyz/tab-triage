"""
Processor module: orchestrates the per-tab pipeline.

For each URL: fetch content → send to LLM for triage → get embedding → return structured result.
"""

import json
import logging
import time
from dataclasses import dataclass, field

from fetcher import FetchResult, fetch_url
from llm_client import LLMClient
from prompts import (
    TRIAGE_SYSTEM_PROMPT,
    TRIAGE_USER_PROMPT_TEMPLATE,
    TRIAGE_URL_ONLY_PROMPT_TEMPLATE,
)
import config

logger = logging.getLogger(__name__)


@dataclass
class TriageResult:
    """The full processed result for a single tab."""
    url: str
    fetch_success: bool
    fetch_error: str = ""

    # From LLM
    title: str = ""
    summary: str = ""
    category: str = ""
    actionability: int = 3
    implied_action: str = ""
    importance: int = 3
    effort: int = 3
    staleness: int = 3
    insight_density: int = 3

    # Computed
    quick_win_score: float = 0.0  # importance / effort ratio

    # Embedding (stored in SQLite, not CSV)
    embedding: list[float] = field(default_factory=list, repr=False)

    # Processing metadata
    llm_raw_response: str = ""
    llm_parse_error: str = ""

    def to_csv_dict(self) -> dict:
        """Return a flat dict suitable for CSV output (no embedding)."""
        return {
            "url": self.url,
            "title": self.title,
            "summary": self.summary,
            "category": self.category,
            "actionability": self.actionability,
            "implied_action": self.implied_action,
            "importance": self.importance,
            "effort": self.effort,
            "staleness": self.staleness,
            "insight_density": self.insight_density,
            "quick_win_score": round(self.quick_win_score, 2),
            "fetch_success": self.fetch_success,
            "fetch_error": self.fetch_error,
        }


def process_tab(url: str, client: LLMClient) -> TriageResult:
    """
    Full pipeline for a single tab URL.

    1. Fetch the page content
    2. Send to LLM for triage analysis
    3. Get embedding vector
    4. Return structured result
    """
    result = TriageResult(url=url, fetch_success=False)

    # Step 1: Fetch
    fetch_result = fetch_url(url)
    result.fetch_success = fetch_result.success
    result.fetch_error = fetch_result.error

    if fetch_result.success and fetch_result.title:
        result.title = fetch_result.title

    # Step 2: LLM Triage
    try:
        if fetch_result.success and fetch_result.text:
            user_prompt = TRIAGE_USER_PROMPT_TEMPLATE.format(
                url=url,
                content=fetch_result.text,
            )
        else:
            # Can't fetch? Try to infer from URL alone
            user_prompt = TRIAGE_URL_ONLY_PROMPT_TEMPLATE.format(url=url)

        messages = [
            {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        raw_response = client.chat(
            messages,
        )
        result.llm_raw_response = raw_response

        # Parse the JSON response
        parsed = _parse_llm_response(raw_response)
        result.title = parsed.get("title", result.title or url)
        result.summary = parsed.get("summary", "")
        result.category = parsed.get("category", "uncategorized")
        result.actionability = _clamp(parsed.get("actionability", 3), 1, 5)
        result.implied_action = parsed.get("implied_action", "")
        result.importance = _clamp(parsed.get("importance", 3), 1, 5)
        result.effort = _clamp(parsed.get("effort", 3), 1, 5)
        result.staleness = _clamp(parsed.get("staleness", 3), 1, 5)
        result.insight_density = _clamp(parsed.get("insight_density", 3), 1, 5)

        # Compute quick-win score (importance / effort, weighted by staleness)
        if result.effort > 0:
            result.quick_win_score = (
                result.importance / result.effort
            ) * (result.staleness / 3.0)  # staleness=3 is neutral

    except Exception as e:
        result.llm_parse_error = str(e)
        logger.error(f"  LLM processing failed for {url}: {e}")

    # Step 3: Embedding
    # Prefer raw article text (clean prose from trafilatura, no HTML noise).
    # Fall back to LLM-generated title+summary only if fetch failed.
    try:
        if fetch_result.success and fetch_result.text:
            embed_text = fetch_result.text[:config.MAX_EMBED_CHARS]
        else:
            embed_text = f"{result.title}. {result.summary}"
        result.embedding = client.embed(embed_text)
    except Exception as e:
        logger.warning(f"  Embedding failed for {url}: {e}")
        result.embedding = []

    # Rate limiting
    time.sleep(config.REQUEST_DELAY)

    return result


def process_tabs(urls: list[str], client: LLMClient = None,
                 progress_callback=None) -> list[TriageResult]:
    """
    Process a batch of tab URLs through the full pipeline.

    Args:
        urls: List of URLs to process.
        client: LLMClient instance (creates one if not provided).
        progress_callback: Optional callable(i, total, result) for progress.

    Returns:
        List of TriageResult objects.
    """
    if client is None:
        client = LLMClient()

    results = []
    total = len(urls)

    for i, url in enumerate(urls):
        logger.info(f"Processing [{i+1}/{total}]: {url}")

        try:
            result = process_tab(url, client)
            logger.info(
                f"  → {result.title!r} | "
                f"cat={result.category} | "
                f"action={result.actionability}/5 | "
                f"importance={result.importance}/5"
            )
        except Exception as e:
            logger.error(f"  Unexpected error processing {url}: {e}")
            result = TriageResult(
                url=url, fetch_success=False,
                fetch_error=f"Processing error: {e}",
            )

        results.append(result)

        if progress_callback:
            progress_callback(i + 1, total, result)

    return results


def _parse_llm_response(raw: str) -> dict:
    """
    Parse the LLM's JSON response, handling common issues.

    Handles:
    - Qwen 3 <think>...</think> reasoning blocks (may contain { chars)
    - Markdown code fences (```json ... ```)
    - Trailing text after JSON
    """
    import re

    text = raw.strip()

    # Strip Qwen 3 thinking blocks — these often contain braces
    # and other JSON-like content that confuse the { } extraction below.
    # Handle both closed (<think>...</think>) and unclosed (<think>...)
    # tags — Qwen sometimes doesn't emit the closing tag.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove first line (```json or ```) and last line (```)
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Try to find JSON object in the text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        logger.warning(f"After stripping think tags, text starts with: {text[:200]}")
        logger.debug(f"Full raw response: {raw[:500]}")
        return {}


def _clamp(value, min_val, max_val):
    """Clamp a value to a range, handling non-numeric input."""
    try:
        v = int(value)
        return max(min_val, min(v, max_val))
    except (ValueError, TypeError):
        return (min_val + max_val) // 2
