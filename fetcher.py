"""
Fetcher module: takes URLs and extracts clean article text.

Uses trafilatura as the primary extractor (it's specifically designed for
article/blog content and handles boilerplate removal well), with a
BeautifulSoup fallback for pages trafilatura can't handle.
"""

import logging
import requests
from urllib.parse import urlparse

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

import config

logger = logging.getLogger(__name__)


class FetchResult:
    """Container for the result of fetching a URL."""

    def __init__(self, url: str, title: str = "", text: str = "",
                 success: bool = True, error: str = ""):
        self.url = url
        self.title = title
        self.text = text
        self.success = success
        self.error = error

    def __repr__(self):
        status = "OK" if self.success else f"FAIL: {self.error}"
        return f"FetchResult({self.url!r}, {status}, {len(self.text)} chars)"


def fetch_url(url: str) -> FetchResult:
    """
    Fetch a URL and extract clean article text.

    Tries trafilatura first (best for articles/blogs), falls back to
    BeautifulSoup for basic text extraction.
    """
    url = url.strip()
    if not url:
        return FetchResult(url, success=False, error="Empty URL")

    # Normalize URL
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        response = _download(url)
    except Exception as e:
        return FetchResult(url, success=False, error=f"Download failed: {e}")

    html = response.text

    # Try trafilatura first
    title, text = "", ""
    if HAS_TRAFILATURA:
        title, text = _extract_trafilatura(html, url)

    # Fallback to BeautifulSoup if trafilatura got nothing useful
    if not text and HAS_BS4:
        title_bs, text_bs = _extract_beautifulsoup(html)
        if not title:
            title = title_bs
        text = text_bs

    # Last resort: try to get at least a title from the HTML
    if not title:
        title = _extract_title_from_html(html)

    if not text:
        return FetchResult(url, title=title, success=False,
                           error="Could not extract text content")

    # Truncate to configured max length
    if len(text) > config.MAX_ARTICLE_CHARS:
        text = text[:config.MAX_ARTICLE_CHARS] + "\n\n[...truncated...]"

    return FetchResult(url, title=title, text=text, success=True)


def _download(url: str) -> requests.Response:
    """Download a URL with configured timeout and user agent."""
    headers = {"User-Agent": config.USER_AGENT}
    response = requests.get(
        url,
        headers=headers,
        timeout=config.FETCH_TIMEOUT,
        allow_redirects=True,
    )
    response.raise_for_status()
    return response


def _extract_trafilatura(html: str, url: str) -> tuple[str, str]:
    """Extract title and text using trafilatura."""
    try:
        metadata = trafilatura.extract_metadata(html)
        title = metadata.title if metadata and metadata.title else ""

        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            favor_recall=True,  # get more content rather than less
        )
        return title, text or ""
    except Exception as e:
        logger.debug(f"Trafilatura extraction failed for {url}: {e}")
        return "", ""


def _extract_beautifulsoup(html: str) -> tuple[str, str]:
    """Fallback: extract title and visible text using BeautifulSoup."""
    try:
        soup = BeautifulSoup(html, "html.parser")

        # Title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Remove script/style elements
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        # Get text from likely-content areas first
        text = ""
        for selector in ["article", "main", '[role="main"]',
                         ".post-content", ".entry-content", ".article-body"]:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(separator="\n", strip=True)
                break

        # If no content area found, get body text
        if not text:
            body = soup.find("body")
            if body:
                text = body.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        return title, text
    except Exception as e:
        logger.debug(f"BeautifulSoup extraction failed: {e}")
        return "", ""


def _extract_title_from_html(html: str) -> str:
    """Last-resort title extraction with basic string search."""
    import re
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def fetch_urls(urls: list[str], progress_callback=None) -> list[FetchResult]:
    """
    Fetch multiple URLs, with optional progress callback.

    Args:
        urls: List of URLs to fetch.
        progress_callback: Optional callable(i, total, result) for progress.

    Returns:
        List of FetchResult objects.
    """
    results = []
    total = len(urls)

    for i, url in enumerate(urls):
        logger.info(f"Fetching [{i+1}/{total}]: {url}")
        result = fetch_url(url)

        if not result.success:
            logger.warning(f"  Failed: {result.error}")
        else:
            logger.info(f"  OK: {result.title!r} ({len(result.text)} chars)")

        results.append(result)

        if progress_callback:
            progress_callback(i + 1, total, result)

    return results
