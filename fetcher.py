"""
Fetcher module: takes URLs and extracts clean article text.

Extraction strategy (in priority order):
  1. Site-specific extractors (YouTube, Substack, academic papers)
  2. Structured data in HTML (JSON-LD, OpenGraph, __NEXT_DATA__)
  3. Trafilatura (best general-purpose article extractor)
  4. BeautifulSoup fallback (raw visible text from content areas)

The goal is clean prose suitable for LLM summarization and embedding —
no nav bars, no cookie banners, no legal boilerplate.
"""

import json
import logging
import re
from html import unescape
from urllib.parse import urlparse, parse_qs

import requests

# curl_cffi impersonates Chrome's TLS fingerprint, which is critical for
# bypassing Cloudflare and similar bot detection on sites like Substack,
# YouTube, Marginal Revolution, academic publishers, etc.  Without it,
# Python's requests library gets flagged by its distinctive JA3 fingerprint
# and you get either 403s or empty Cloudflare challenge pages.
try:
    from curl_cffi import requests as cffi_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

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

if HAS_CURL_CFFI:
    logger.info("curl_cffi available — using Chrome TLS impersonation")
else:
    logger.warning(
        "curl_cffi not installed — some sites may block requests. "
        "Install with: pip install curl_cffi"
    )


class FetchResult:
    """Container for the result of fetching a URL."""

    def __init__(self, url: str, title: str = "", text: str = "",
                 success: bool = True, error: str = "",
                 extraction_method: str = ""):
        self.url = url
        self.title = title
        self.text = text
        self.success = success
        self.error = error
        self.extraction_method = extraction_method

    def __repr__(self):
        status = "OK" if self.success else f"FAIL: {self.error}"
        method = f" [{self.extraction_method}]" if self.extraction_method else ""
        return f"FetchResult({self.url!r}, {status}, {len(self.text)} chars{method})"


# ── Main entry point ───────────────────────────────────────────────────

def fetch_url(url: str) -> FetchResult:
    """
    Fetch a URL and extract clean article text.

    Tries multiple extraction strategies in order of specificity,
    from site-specific handlers down to generic text extraction.
    """
    url = url.strip()
    if not url:
        return FetchResult(url, success=False, error="Empty URL")

    # Normalize URL
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")

    try:
        response = _download(url)
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        return FetchResult(url, success=False,
                           error=f"HTTP {status} from {domain}")
    except Exception as e:
        return FetchResult(url, success=False, error=f"Download failed: {e}")

    html = response.text
    title, text, method = "", "", ""

    # ── Layer 1: Site-specific extractors ──────────────────────────
    if _is_youtube(domain, parsed):
        title, text = _extract_youtube(html, url)
        method = "youtube"

    elif _is_substack(html, domain):
        title, text = _extract_substack_nextdata(html)
        method = "substack/__NEXT_DATA__"

    elif _is_academic(domain):
        title, text = _extract_academic(html, domain)
        method = "academic/json-ld"

    # ── Layer 2: Structured data (JSON-LD, OpenGraph) ─────────────
    if not text:
        title_sd, text_sd = _extract_structured_data(html)
        if text_sd and len(text_sd) > 200:  # only use if substantial
            title = title_sd or title
            text = text_sd
            method = "json-ld/opengraph"

    # ── Layer 3: Trafilatura ──────────────────────────────────────
    if not text and HAS_TRAFILATURA:
        title_tr, text_tr = _extract_trafilatura(html, url)
        if text_tr:
            title = title_tr or title
            text = text_tr
            method = "trafilatura"

    # ── Layer 4: BeautifulSoup fallback ───────────────────────────
    if not text and HAS_BS4:
        title_bs, text_bs = _extract_beautifulsoup(html)
        if text_bs:
            title = title_bs or title
            text = text_bs
            method = "beautifulsoup"

    # ── Last resort: at least get a title ─────────────────────────
    if not title:
        title = _extract_title_from_html(html)

    if not text:
        return FetchResult(url, title=title, success=False,
                           error="Could not extract text content")

    # Truncate to configured max length
    if len(text) > config.MAX_ARTICLE_CHARS:
        text = text[:config.MAX_ARTICLE_CHARS] + "\n\n[...truncated...]"

    logger.debug(f"  Extracted via {method}: {len(text)} chars")
    return FetchResult(url, title=title, text=text, success=True,
                       extraction_method=method)


# ── Download ───────────────────────────────────────────────────────────

class _DownloadResponse:
    """Unified response wrapper so the rest of the code doesn't care
    whether we used curl_cffi or requests under the hood."""
    def __init__(self, text: str, status_code: int, url: str):
        self.text = text
        self.status_code = status_code
        self.url = url  # final URL after redirects

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                response=type("R", (), {"status_code": self.status_code})()
            )


def _download(url: str) -> _DownloadResponse:
    """Download a URL, using curl_cffi (Chrome impersonation) if available."""
    headers = {
        "User-Agent": config.USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    if HAS_CURL_CFFI:
        # Chrome TLS impersonation — bypasses Cloudflare JA3 fingerprinting
        resp = cffi_requests.get(
            url,
            headers=headers,
            timeout=config.FETCH_TIMEOUT,
            allow_redirects=True,
            impersonate="chrome",
        )
        result = _DownloadResponse(resp.text, resp.status_code, str(resp.url))
        result.raise_for_status()
        return result
    else:
        # Plain requests — works for non-Cloudflare sites
        resp = requests.get(
            url,
            headers=headers,
            timeout=config.FETCH_TIMEOUT,
            allow_redirects=True,
        )
        resp.raise_for_status()
        return _DownloadResponse(resp.text, resp.status_code, resp.url)


# ── Site detection helpers ─────────────────────────────────────────────

def _is_youtube(domain: str, parsed) -> bool:
    return domain in ("youtube.com", "m.youtube.com", "youtu.be")


def _is_substack(html: str, domain: str) -> bool:
    """Detect Substack — either substack.com or custom domains serving Substack."""
    if "substack.com" in domain:
        return True
    # Custom-domain Substacks all embed __NEXT_DATA__ with substack markers
    if "__NEXT_DATA__" in html and ("substack" in html.lower()[:5000]
                                     or '"pub"' in html[:5000]):
        return True
    return False


def _is_academic(domain: str) -> bool:
    """Detect academic paper sites."""
    academic_domains = {
        "papers.ssrn.com", "ssrn.com",
        "pnas.org",
        "sciencedirect.com",
        "nature.com",
        "science.org",
        "arxiv.org",
        "doi.org",
        "pubmed.ncbi.nlm.nih.gov",
        "scholar.google.com",
        "jstor.org",
        "springer.com", "link.springer.com",
        "wiley.com", "onlinelibrary.wiley.com",
        "tandfonline.com",
        "ncbi.nlm.nih.gov",
    }
    return any(domain.endswith(d) for d in academic_domains)


# ── YouTube extractor ──────────────────────────────────────────────────

def _extract_youtube(html: str, url: str) -> tuple[str, str]:
    """
    Extract video title and description from YouTube's embedded JSON.

    YouTube pages contain ytInitialPlayerResponse as a JS variable in a
    <script> tag.  The challenge is that the JSON blob is followed by more
    JS code, so simple regex capture doesn't give us clean JSON.  We use
    json.JSONDecoder.raw_decode() which parses exactly one JSON object from
    the start of a string and ignores trailing data.

    Fallback chain: ytInitialPlayerResponse → OpenGraph meta tags.
    """
    try:
        title = ""
        description = ""

        # Try ytInitialPlayerResponse (has the cleanest description)
        marker = "var ytInitialPlayerResponse = "
        idx = html.find(marker)
        if idx >= 0:
            json_start = idx + len(marker)
            try:
                decoder = json.JSONDecoder()
                player_data, _ = decoder.raw_decode(html, json_start)
                vd = player_data.get("videoDetails", {})
                title = vd.get("title", "")
                description = vd.get("shortDescription", "")
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug(f"YouTube ytInitialPlayerResponse parse failed: {e}")

        # Fallback: ytInitialData (larger, more nested, but has description)
        if not description:
            marker2 = "var ytInitialData = "
            idx2 = html.find(marker2)
            if idx2 >= 0:
                json_start2 = idx2 + len(marker2)
                try:
                    decoder = json.JSONDecoder()
                    yt_data, _ = decoder.raw_decode(html, json_start2)
                    contents = (yt_data.get("contents", {})
                                .get("twoColumnWatchNextResults", {})
                                .get("results", {})
                                .get("results", {})
                                .get("contents", []))
                    for item in contents:
                        vpr = item.get("videoPrimaryInfoRenderer", {})
                        if vpr:
                            title_runs = vpr.get("title", {}).get("runs", [])
                            if title_runs:
                                title = title_runs[0].get("text", title)
                        vsr = item.get("videoSecondaryInfoRenderer", {})
                        if vsr:
                            desc_data = (vsr.get("attributedDescription", {})
                                         .get("content", ""))
                            if desc_data:
                                description = desc_data
                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug(f"YouTube ytInitialData parse failed: {e}")

        # Fallback: OpenGraph meta tags (always present, truncated but usable)
        if not title:
            og_title = re.search(
                r'<meta\s+property="og:title"\s+content="([^"]*)"',
                html, re.IGNORECASE
            )
            if og_title:
                title = unescape(og_title.group(1))

        if not description:
            og_desc = re.search(
                r'<meta\s+(?:property="og:description"|name="description")\s+content="([^"]*)"',
                html, re.IGNORECASE
            )
            if og_desc:
                description = unescape(og_desc.group(1))

        if title or description:
            text = f"{title}\n\n{description}" if title and description else (title or description)
            return title, text

        return title, ""

    except Exception as e:
        logger.debug(f"YouTube extraction failed: {e}")
        return "", ""


# ── Substack extractor ─────────────────────────────────────────────────

def _extract_substack_nextdata(html: str) -> tuple[str, str]:
    """
    Extract post content from Substack's __NEXT_DATA__ JSON blob.

    All Substack pages (including custom domains) embed the full post
    body in a <script id="__NEXT_DATA__"> tag as a Next.js data payload.
    The body is HTML, which we then strip to plain text.
    """
    try:
        match = re.search(
            r'<script\s+id="__NEXT_DATA__"\s+type="application/json">(.*?)</script>',
            html, re.DOTALL
        )
        if not match:
            return "", ""

        data = json.loads(match.group(1))

        # Navigate to the post object — structure varies slightly
        post = None
        props = data.get("props", {}).get("pageProps", {})
        # Direct post
        if "post" in props:
            post = props["post"]
        # Sometimes nested under initialState
        elif "initialState" in props:
            posts = (props["initialState"].get("post", {})
                     .get("posts", {}))
            if posts:
                post = next(iter(posts.values()), None)

        if not post:
            return "", ""

        title = post.get("title", "")
        subtitle = post.get("subtitle", "")

        # The body is HTML — strip it to text
        body_html = post.get("body_html", "") or post.get("body", "")
        if body_html and HAS_BS4:
            soup = BeautifulSoup(body_html, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
        elif body_html:
            # Crude HTML strip if no BS4
            text = re.sub(r"<[^>]+>", " ", body_html)
            text = re.sub(r"\s+", " ", text).strip()
        else:
            # Fall back to truncated preview text
            text = post.get("truncated_body_text", "")

        if subtitle and text:
            text = f"{subtitle}\n\n{text}"

        return title, text

    except Exception as e:
        logger.debug(f"Substack __NEXT_DATA__ extraction failed: {e}")
        return "", ""


# ── Academic paper extractor ───────────────────────────────────────────

def _extract_academic(html: str, domain: str) -> tuple[str, str]:
    """
    Extract title and abstract from academic paper pages.

    Academic sites often block scraping of full text but always have
    metadata available in JSON-LD, meta tags, and sometimes visible
    abstract sections. We grab everything we can.
    """
    title, abstract = "", ""

    # Try JSON-LD first (richest structured data)
    try:
        ld_blocks = re.findall(
            r'<script\s+type="application/ld\+json">(.*?)</script>',
            html, re.DOTALL
        )
        for block in ld_blocks:
            try:
                ld = json.loads(block)
                # Handle both single object and array
                items = ld if isinstance(ld, list) else [ld]
                for item in items:
                    item_type = item.get("@type", "")
                    if item_type in ("ScholarlyArticle", "Article",
                                     "MedicalScholarlyArticle",
                                     "TechArticle", "NewsArticle"):
                        title = item.get("headline", "") or item.get("name", "")
                        abstract = item.get("abstract", "")
                        if abstract:
                            break
                    elif item_type == "WebPage":
                        if not title:
                            title = item.get("name", "")
                        desc = item.get("description", "")
                        if desc and len(desc) > len(abstract):
                            abstract = desc
            except json.JSONDecodeError:
                continue
    except Exception as e:
        logger.debug(f"Academic JSON-LD extraction failed: {e}")

    # Try meta tags
    if not abstract:
        for pattern in [
            r'<meta\s+name="(?:citation_abstract|description|DC\.description)"\s+content="([^"]*)"',
            r'<meta\s+property="og:description"\s+content="([^"]*)"',
        ]:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                candidate = unescape(match.group(1))
                if len(candidate) > len(abstract):
                    abstract = candidate

    if not title:
        for pattern in [
            r'<meta\s+name="citation_title"\s+content="([^"]*)"',
            r'<meta\s+property="og:title"\s+content="([^"]*)"',
        ]:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                title = unescape(match.group(1))
                break

    # Try to grab visible abstract section via BeautifulSoup
    if HAS_BS4 and (not abstract or len(abstract) < 200):
        try:
            soup = BeautifulSoup(html, "html.parser")
            # Common abstract selectors across academic sites
            for selector in [
                "#abstract", ".abstract", '[class*="abstract"]',
                "#Abs1", ".article-abstract",
                'section[id="abstract"]', 'div[role="paragraph"]',
            ]:
                el = soup.select_one(selector)
                if el:
                    candidate = el.get_text(separator=" ", strip=True)
                    # Remove the "Abstract" heading itself
                    candidate = re.sub(r"^Abstract\s*", "", candidate, flags=re.IGNORECASE)
                    if len(candidate) > len(abstract):
                        abstract = candidate
                    break
        except Exception:
            pass

    # Also grab authors for context
    authors = ""
    author_match = re.findall(
        r'<meta\s+name="citation_author"\s+content="([^"]*)"',
        html, re.IGNORECASE
    )
    if author_match:
        authors = ", ".join(author_match[:5])  # cap at 5
        if len(author_match) > 5:
            authors += f" et al. ({len(author_match)} authors)"

    # Compose the text
    parts = []
    if authors:
        parts.append(f"Authors: {authors}")
    if abstract:
        parts.append(f"\n{abstract}")

    text = "\n".join(parts)
    return title, text


# ── Structured data extraction (generic) ──────────────────────────────

def _extract_structured_data(html: str) -> tuple[str, str]:
    """
    Extract title and description from JSON-LD and OpenGraph meta tags.

    This is a generic fallback that works on many sites even when
    trafilatura fails, because structured data is always in the
    initial HTML (no JS rendering needed).
    """
    title, description = "", ""

    # JSON-LD
    try:
        ld_blocks = re.findall(
            r'<script\s+type="application/ld\+json">(.*?)</script>',
            html, re.DOTALL
        )
        for block in ld_blocks:
            try:
                ld = json.loads(block)
                items = ld if isinstance(ld, list) else [ld]
                for item in items:
                    if not title:
                        title = (item.get("headline", "")
                                 or item.get("name", ""))
                    body = (item.get("articleBody", "")
                            or item.get("text", "")
                            or item.get("description", ""))
                    if body and len(body) > len(description):
                        description = body
            except json.JSONDecodeError:
                continue
    except Exception:
        pass

    # OpenGraph / meta tags as supplement
    if not title:
        match = re.search(
            r'<meta\s+property="og:title"\s+content="([^"]*)"',
            html, re.IGNORECASE
        )
        if match:
            title = unescape(match.group(1))

    if not description or len(description) < 200:
        for pattern in [
            r'<meta\s+property="og:description"\s+content="([^"]*)"',
            r'<meta\s+name="description"\s+content="([^"]*)"',
        ]:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                candidate = unescape(match.group(1))
                if len(candidate) > len(description):
                    description = candidate

    return title, description


# ── Trafilatura ────────────────────────────────────────────────────────

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


# ── BeautifulSoup fallback ─────────────────────────────────────────────

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


# ── Utility ────────────────────────────────────────────────────────────

def _extract_title_from_html(html: str) -> str:
    """Last-resort title extraction with basic string search."""
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if match:
        return unescape(match.group(1).strip())
    return ""


# ── Batch fetch ────────────────────────────────────────────────────────

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
            logger.warning(f"  ✗ {result.error}")
        else:
            logger.info(f"  ✓ [{result.extraction_method}] "
                        f"{result.title!r} ({len(result.text)} chars)")

        results.append(result)

        if progress_callback:
            progress_callback(i + 1, total, result)

    return results
