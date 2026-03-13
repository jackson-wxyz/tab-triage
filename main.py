#!/usr/bin/env python3
"""
Tab Triage Pipeline — Main Entry Point

Usage:
    # Process a file containing one URL per line:
    python main.py urls.txt

    # Process specific URLs directly:
    python main.py --urls "https://example.com" "https://other.com"

    # Process in mock mode (no LM Studio needed):
    python main.py urls.txt --mock

    # Process with real LM Studio:
    python main.py urls.txt --no-mock

    # Custom output directory:
    python main.py urls.txt --output-dir ./my_results

    # Change sort order:
    python main.py urls.txt --sort-by importance
"""

import argparse
import logging
import os
import sys
import time

import config
from llm_client import LLMClient
from processor import process_tabs
from output import write_csv, write_sqlite, print_summary


def load_urls_from_file(filepath: str) -> list[str]:
    """Load URLs from a text file, one per line. Skips blanks and comments."""
    urls = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def setup_logging(verbose: bool = False):
    """Configure logging to both console and file."""
    level = logging.DEBUG if verbose else logging.INFO

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))

    # File handler (always verbose)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(config.OUTPUT_DIR, "triage.log"),
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    ))

    logging.basicConfig(level=logging.DEBUG, handlers=[console, file_handler])


def main():
    parser = argparse.ArgumentParser(
        description="Tab Triage Pipeline — process browser tabs with local LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_file", nargs="?",
        help="Text file with one URL per line",
    )
    parser.add_argument(
        "--urls", nargs="+",
        help="URLs to process (alternative to input file)",
    )
    parser.add_argument(
        "--mock", action="store_true", default=None,
        help="Force mock mode (no LM Studio needed)",
    )
    parser.add_argument(
        "--no-mock", action="store_true",
        help="Force real mode (requires LM Studio)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=f"Output directory (default: {config.OUTPUT_DIR})",
    )
    parser.add_argument(
        "--sort-by", default="quick_win_score",
        choices=["quick_win_score", "actionability", "importance",
                 "insight_density", "staleness", "category"],
        help="Sort CSV results by this column (default: quick_win_score)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging output",
    )

    args = parser.parse_args()

    # Determine URLs
    urls = []
    if args.input_file:
        if not os.path.isfile(args.input_file):
            print(f"Error: file not found: {args.input_file}", file=sys.stderr)
            sys.exit(1)
        urls = load_urls_from_file(args.input_file)
    elif args.urls:
        urls = args.urls
    else:
        parser.print_help()
        print("\nError: provide either an input file or --urls", file=sys.stderr)
        sys.exit(1)

    if not urls:
        print("No URLs to process.", file=sys.stderr)
        sys.exit(1)

    # Apply config overrides
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.mock is True or (args.mock is None and not args.no_mock):
        config.MOCK_MODE = True
    if args.no_mock:
        config.MOCK_MODE = False

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Banner
    mock_label = " [MOCK MODE]" if config.MOCK_MODE else ""
    print(f"\n{'='*60}")
    print(f"  Tab Triage Pipeline{mock_label}")
    print(f"  Processing {len(urls)} URLs")
    print(f"{'='*60}\n")

    # Process
    client = LLMClient()
    start_time = time.time()

    def progress(i, total, result):
        status = "✓" if result.fetch_success else "✗"
        title = result.title[:45] if result.title else "(no title)"
        print(f"  [{i:3d}/{total}] {status} {title}")

    results = process_tabs(urls, client, progress_callback=progress)

    elapsed = time.time() - start_time

    # Write outputs
    csv_path = write_csv(results, sort_by=args.sort_by)
    sqlite_path = write_sqlite(results)

    # Print summary
    print_summary(results)

    print(f"\n  Time elapsed: {elapsed:.1f}s "
          f"({elapsed/len(urls):.1f}s per URL)")
    print(f"\n  Outputs:")
    print(f"    CSV:    {csv_path}")
    print(f"    SQLite: {sqlite_path}")
    print(f"    Log:    {os.path.join(config.OUTPUT_DIR, 'triage.log')}")
    print()


if __name__ == "__main__":
    main()
