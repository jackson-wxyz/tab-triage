#!/usr/bin/env python3
"""
Diagnostic script: test the fetcher against problem URLs and report
exactly what's happening at each extraction layer.

Run on your desktop:
    python diagnose_fetch.py

Outputs a detailed report showing HTTP status, content type, HTML size,
what each extractor found (or didn't), and the first ~300 chars of
extracted text. Use this to identify which sites need further fixes.
"""

import json
import re
import sys
from html import unescape
from urllib.parse import urlparse

import requests

try:
    from curl_cffi import requests as cffi_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

# Borrow config for headers/timeout
try:
    import config
    USER_AGENT = config.USER_AGENT
    FETCH_TIMEOUT = config.FETCH_TIMEOUT
except ImportError:
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
    FETCH_TIMEOUT = 15

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


# ── Problem URLs to test ──────────────────────────────────────────────

TEST_URLS = [
    # Substack (custom domains)
    "https://secondperson.dating/p/welcome-to-second-person", #this loads normally for me
    "https://volts.wtf/p/the-fight-to-build-faster-in-california", #this loads normally for me
    # Substack (standard domain) — control
    "https://www.astralcodexten.com/p/your-book-review-the-educated-mind", #this loads normally for me
    # YouTube
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ", #LOL, thanks for the rickroll claude
    # Academic
    "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4003127", #when i go to the site manually, it performs a bot check which the web fetcher might fail
    "https://www.pnas.org/doi/10.1073/pnas.2307354120", #this loads normally for me
    "https://www.sciencedirect.com/science/article/pii/S0378775323009564", #this loads normally for me
    # WordPress blog
    "https://marginalrevolution.com/marginalrevolution/2024/01/saturday-assorted-links-438", #this loads normally for me
    # Paywall (expected to fail gracefully)
    "https://www.nytimes.com/2026/03/15/business/the-billionaire-backlash-against-a-philanthropic-dream.html", #this loads normally for me, but of course i'm logged into my account when i visit manually
]


def diagnose_url(url: str) -> dict:
    """Run all extraction layers against a URL and report findings."""
    result = {
        "url": url,
        "domain": urlparse(url).netloc,
        "http_status": None,
        "content_type": None,
        "html_size": 0,
        "layers": {},
    }

    # ── Step 1: Download ──────────────────────────────────────────
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    try:
        if HAS_CURL_CFFI:
            resp = cffi_requests.get(url, headers=headers, timeout=FETCH_TIMEOUT,
                                     allow_redirects=True, impersonate="chrome")
            result["downloader"] = "curl_cffi (Chrome impersonation)"
        else:
            resp = requests.get(url, headers=headers, timeout=FETCH_TIMEOUT,
                                allow_redirects=True)
            result["downloader"] = "requests (no TLS impersonation)"
        result["http_status"] = resp.status_code
        result["content_type"] = resp.headers.get("Content-Type", "?")
        result["html_size"] = len(resp.text)
        result["final_url"] = str(resp.url)
        html = resp.text
    except Exception as e:
        result["download_error"] = str(e)
        result["downloader"] = "curl_cffi" if HAS_CURL_CFFI else "requests"
        return result

    if resp.status_code >= 400:
        result["download_error"] = f"HTTP {resp.status_code}"
        return result

    # ── Step 2: Check for structured data in HTML ─────────────────

    # __NEXT_DATA__ (Substack, Next.js sites)
    nd_match = re.search(
        r'<script\s+id="__NEXT_DATA__"\s+type="application/json">(.*?)</script>',
        html, re.DOTALL
    )
    if nd_match:
        try:
            nd_data = json.loads(nd_match.group(1))
            post = nd_data.get("props", {}).get("pageProps", {}).get("post", {})
            result["layers"]["__NEXT_DATA__"] = {
                "found": True,
                "has_post": bool(post),
                "title": post.get("title", "")[:100] if post else "",
                "body_html_len": len(post.get("body_html", "")) if post else 0,
                "keys_at_root": list(nd_data.get("props", {}).get("pageProps", {}).keys())[:10],
            }
        except json.JSONDecodeError as e:
            result["layers"]["__NEXT_DATA__"] = {"found": True, "parse_error": str(e)}
    else:
        result["layers"]["__NEXT_DATA__"] = {"found": False}

    # ytInitialPlayerResponse (YouTube)
    yt_player_match = re.search(
        r"var\s+ytInitialPlayerResponse\s*=\s*(\{.*?\});\s*</script>",
        html, re.DOTALL
    )
    if yt_player_match:
        try:
            yt_data = json.loads(yt_player_match.group(1))
            vd = yt_data.get("videoDetails", {})
            result["layers"]["ytInitialPlayerResponse"] = {
                "found": True,
                "title": vd.get("title", "")[:100],
                "description_len": len(vd.get("shortDescription", "")),
                "description_preview": vd.get("shortDescription", "")[:200],
            }
        except json.JSONDecodeError as e:
            result["layers"]["ytInitialPlayerResponse"] = {
                "found": True, "parse_error": str(e)
            }
    else:
        result["layers"]["ytInitialPlayerResponse"] = {"found": False}

    # JSON-LD
    ld_blocks = re.findall(
        r'<script\s+type="application/ld\+json">(.*?)</script>',
        html, re.DOTALL
    )
    ld_results = []
    for i, block in enumerate(ld_blocks):
        try:
            ld = json.loads(block)
            items = ld if isinstance(ld, list) else [ld]
            for item in items:
                ld_results.append({
                    "@type": item.get("@type", "?"),
                    "headline": (item.get("headline", "") or item.get("name", ""))[:100],
                    "has_articleBody": bool(item.get("articleBody", "")),
                    "articleBody_len": len(item.get("articleBody", "")),
                    "description_len": len(item.get("description", "")),
                    "abstract_len": len(item.get("abstract", "")),
                })
        except json.JSONDecodeError:
            ld_results.append({"parse_error": True, "block_index": i})
    result["layers"]["json-ld"] = ld_results if ld_results else "none found"

    # OpenGraph meta tags
    og_title = ""
    og_desc = ""
    og_match = re.search(r'<meta\s+property="og:title"\s+content="([^"]*)"', html, re.I)
    if og_match:
        og_title = unescape(og_match.group(1))[:100]
    og_match = re.search(r'<meta\s+property="og:description"\s+content="([^"]*)"', html, re.I)
    if og_match:
        og_desc = unescape(og_match.group(1))[:200]
    result["layers"]["opengraph"] = {"title": og_title, "description": og_desc}

    # citation meta tags (academic)
    cit_title = ""
    cit_match = re.search(r'<meta\s+name="citation_title"\s+content="([^"]*)"', html, re.I)
    if cit_match:
        cit_title = unescape(cit_match.group(1))[:100]
    cit_abstract = ""
    cit_abs_match = re.search(r'<meta\s+name="citation_abstract"\s+content="([^"]*)"', html, re.I)
    if cit_abs_match:
        cit_abstract = unescape(cit_abs_match.group(1))[:200]
    result["layers"]["citation_meta"] = {
        "title": cit_title, "abstract_preview": cit_abstract
    }

    # ── Step 3: Run extractors ────────────────────────────────────

    # Trafilatura
    if HAS_TRAFILATURA:
        try:
            traf_text = trafilatura.extract(
                html, url=url, include_comments=False,
                include_tables=True, favor_recall=True,
            )
            result["layers"]["trafilatura"] = {
                "extracted": bool(traf_text),
                "length": len(traf_text) if traf_text else 0,
                "preview": (traf_text[:300] + "...") if traf_text and len(traf_text) > 300 else traf_text or "",
            }
        except Exception as e:
            result["layers"]["trafilatura"] = {"error": str(e)}
    else:
        result["layers"]["trafilatura"] = {"available": False}

    # BeautifulSoup
    if HAS_BS4:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            for sel in ["article", "main", '[role="main"]',
                        ".post-content", ".entry-content"]:
                el = soup.select_one(sel)
                if el:
                    bs_text = el.get_text(separator="\n", strip=True)
                    result["layers"]["beautifulsoup"] = {
                        "selector_hit": sel,
                        "length": len(bs_text),
                        "preview": (bs_text[:300] + "...") if len(bs_text) > 300 else bs_text,
                    }
                    break
            else:
                body = soup.find("body")
                body_text = body.get_text(separator="\n", strip=True) if body else ""
                result["layers"]["beautifulsoup"] = {
                    "selector_hit": "body (fallback)",
                    "length": len(body_text),
                    "preview": (body_text[:300] + "...") if len(body_text) > 300 else body_text,
                }
        except Exception as e:
            result["layers"]["beautifulsoup"] = {"error": str(e)}

    return result


def main():
    # Allow passing custom URLs as args
    urls = sys.argv[1:] if len(sys.argv) > 1 else TEST_URLS

    print("=" * 70)
    print("  TAB TRIAGE — FETCH DIAGNOSTICS")
    print("=" * 70)
    print(f"\nTesting {len(urls)} URLs...\n")

    all_results = []

    for i, url in enumerate(urls):
        print(f"\n{'─' * 70}")
        print(f"[{i+1}/{len(urls)}] {url}")
        print(f"{'─' * 70}")

        diag = diagnose_url(url)
        all_results.append(diag)

        if "download_error" in diag:
            print(f"  ✗ DOWNLOAD FAILED: {diag['download_error']}")
            continue

        print(f"  HTTP {diag['http_status']} | {diag['content_type']} | "
              f"{diag['html_size']:,} chars HTML")
        if diag.get("final_url") != url:
            print(f"  Redirected to: {diag['final_url']}")

        # Report on each layer
        layers = diag["layers"]

        # __NEXT_DATA__
        nd = layers.get("__NEXT_DATA__", {})
        if nd.get("found"):
            if nd.get("has_post"):
                print(f"  ✓ __NEXT_DATA__: post found — "
                      f"title={nd['title']!r}, body={nd['body_html_len']:,} chars")
            else:
                print(f"  ~ __NEXT_DATA__: found but no post object. "
                      f"Keys: {nd.get('keys_at_root', [])}")

        # YouTube
        yt = layers.get("ytInitialPlayerResponse", {})
        if yt.get("found"):
            if yt.get("parse_error"):
                print(f"  ~ ytInitialPlayerResponse: found but parse failed: {yt['parse_error']}")
            else:
                print(f"  ✓ ytInitialPlayerResponse: "
                      f"title={yt['title']!r}, desc={yt['description_len']} chars")
                if yt.get("description_preview"):
                    print(f"    Preview: {yt['description_preview'][:100]}...")

        # JSON-LD
        ld = layers.get("json-ld", [])
        if ld and ld != "none found":
            for item in ld:
                if item.get("parse_error"):
                    print(f"  ~ JSON-LD: parse error")
                else:
                    parts = [f"@type={item['@type']!r}"]
                    if item.get("headline"):
                        parts.append(f"headline={item['headline']!r}")
                    if item.get("has_articleBody"):
                        parts.append(f"articleBody={item['articleBody_len']:,} chars")
                    if item.get("abstract_len"):
                        parts.append(f"abstract={item['abstract_len']} chars")
                    if item.get("description_len"):
                        parts.append(f"description={item['description_len']} chars")
                    print(f"  {'✓' if item.get('has_articleBody') or item.get('abstract_len') else '~'} "
                          f"JSON-LD: {', '.join(parts)}")
        else:
            print(f"  · JSON-LD: none found")

        # OpenGraph
        og = layers.get("opengraph", {})
        if og.get("title") or og.get("description"):
            print(f"  {'✓' if og.get('description') else '~'} OpenGraph: "
                  f"title={og['title']!r}, desc={len(og.get('description', ''))} chars")

        # Citation meta
        cit = layers.get("citation_meta", {})
        if cit.get("title") or cit.get("abstract_preview"):
            print(f"  ✓ Citation meta: title={cit['title']!r}, "
                  f"abstract={len(cit.get('abstract_preview', ''))} chars")

        # Trafilatura
        traf = layers.get("trafilatura", {})
        if traf.get("extracted"):
            print(f"  ✓ Trafilatura: {traf['length']:,} chars")
            print(f"    Preview: {traf['preview'][:150]}...")
        elif traf.get("error"):
            print(f"  ✗ Trafilatura error: {traf['error']}")
        elif traf.get("available") is False:
            print(f"  · Trafilatura: not installed")
        else:
            print(f"  ✗ Trafilatura: extracted nothing")

        # BeautifulSoup
        bs = layers.get("beautifulsoup", {})
        if bs.get("length", 0) > 0:
            print(f"  {'✓' if bs['length'] > 200 else '~'} BeautifulSoup: "
                  f"{bs['length']:,} chars via {bs['selector_hit']}")
        elif bs.get("error"):
            print(f"  ✗ BeautifulSoup error: {bs['error']}")

        # Summary verdict
        print()
        best = "NONE"
        if nd.get("has_post"):
            best = f"__NEXT_DATA__ ({nd['body_html_len']:,} chars)"
        elif yt.get("found") and not yt.get("parse_error") and yt.get("description_len", 0) > 0:
            best = f"YouTube ({yt['description_len']} chars)"
        elif any(item.get("has_articleBody") for item in (ld if isinstance(ld, list) else [])):
            best = "JSON-LD articleBody"
        elif traf.get("extracted"):
            best = f"Trafilatura ({traf['length']:,} chars)"
        elif bs.get("length", 0) > 200:
            best = f"BeautifulSoup ({bs['length']:,} chars)"
        elif og.get("description"):
            best = f"OpenGraph description ({len(og['description'])} chars)"

        print(f"  → BEST SOURCE: {best}")

    # Write full JSON report
    report_path = "output/fetch_diagnostics.json"
    try:
        import os
        os.makedirs("output", exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n{'=' * 70}")
        print(f"Full diagnostic JSON written to: {report_path}")
        print(f"{'=' * 70}")
    except Exception as e:
        print(f"\nCouldn't write JSON report: {e}")


if __name__ == "__main__":
    main()
