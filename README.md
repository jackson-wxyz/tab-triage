# Tab Triage Pipeline

A Python tool that processes hundreds of open browser tabs through a local LLM (via LM Studio), producing a prioritized spreadsheet of summaries, categories, actionability scores, and extracted action items — plus a SQLite database with embedding vectors for later clustering and similarity search.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

That's it — just `requests`, `trafilatura`, and `beautifulsoup4`. No PyTorch, no GPU libraries. All LLM work goes through LM Studio's HTTP API.

### 2. Export your tabs

**Chrome:** Install the "Copy All URLs" or "OneTab" extension, copy your tab URLs.

**Safari (iPhone):** Long-press the tab count indicator → "Copy Links" gives you every open tab URL.

Paste them into a text file, one URL per line:

```
https://www.lesswrong.com/posts/...
https://forum.effectivealtruism.org/posts/...
https://www.nytimes.com/2025/...
```

### 3. Test with mock mode (no LM Studio needed)

make sure you cd into the folder with main.py, then open command prompt and enter:

```bash
python main.py input-tab-lists/my_tabs.txt --mock
```

This runs the full pipeline with fake LLM responses, verifying that fetching, parsing, and output all work.

### 4. Run for real with LM Studio

1. Open LM Studio on your desktop
2. Load your chat model (e.g., Qwen 3.5) and an embedding model (e.g., `nomic-embed-text-v1.5`)
3. Start the local server (default: `http://localhost:1234`)
4. Edit `config.py` to set `MOCK_MODE = False` and verify your model names
5. Run:

```bash
python main.py input-tab-lists/my_tabs.txt --no-mock #python main.py input-tab-lists/march_2026_working_urls.txt --no-mock
```

## Output

The pipeline produces three files in the `output/` directory:

- **`triage_results.csv`** — Human-readable spreadsheet sorted by "quick win score" (importance/effort ratio weighted by staleness). Open this in Excel, Google Sheets, etc. to review and act on your tabs.

- **`triage_results.db`** — SQLite database with the same data plus 768-dimensional embedding vectors. Use this for clustering, similarity search, and deduplication later.

- **`triage.log`** — Full processing log for debugging.

### CSV Columns

| Column | Description |
|--------|-------------|
| `url` | The original tab URL |
| `title` | Article title (from page or LLM) |
| `summary` | 2-3 sentence summary |
| `category` | Topic tag (AI safety, health, gaming, etc.) |
| `actionability` | 1-5: pure reading → action item itself |
| `implied_action` | LLM's best guess at what you'd do with this tab |
| `importance` | 1-5: how much would completing this improve your life? |
| `effort` | 1-5: five minutes → multi-week project |
| `staleness` | 1-5: probably expired → timeless content |
| `insight_density` | 1-5: familiar rehash → perspective-changing |
| `quick_win_score` | importance/effort × staleness/3 (higher = do this first) |

## Configuration

Edit `config.py` to adjust:

- **`LM_STUDIO_BASE_URL`** — Where LM Studio's API lives (default: `http://localhost:1234/v1`)
- **`CHAT_MODEL`** / **`EMBEDDING_MODEL`** — Model names as they appear in LM Studio
- **`MAX_ARTICLE_CHARS`** — How much article text to send to the LLM (default: 8000 chars ≈ 2000 tokens)
- **`REQUEST_DELAY`** — Seconds between LLM calls (default: 1.0, be kind to your GPU)
- **`MOCK_MODE`** — `True` for testing without LM Studio

## Architecture

```
main.py          CLI entry point, argument parsing, orchestration
config.py        All knobs and settings
fetcher.py       URL → clean article text (trafilatura + BS4 fallback)
llm_client.py    LM Studio API wrapper (chat + embeddings + mock mode)
processor.py     Per-tab pipeline: fetch → summarize → score → embed
prompts.py       System/user prompt templates for Qwen
output.py        CSV + SQLite writer, summary printer
```
Oh, and claude also later added "analyze.py", which produces the serialization and graphs.  Right now, run it with "python analyze.py", although later we might want to work the serialization component more actively into the process of sorting tabs.

## Embedding Models in LM Studio

To use embeddings (for later clustering/dedup), load an embedding model in LM Studio alongside your chat model. Good options:

- `nomic-embed-text-v1.5` (768 dims, fast, good quality)
- `mxbai-embed-large-v1` (1024 dims, slightly better, slightly slower)

The code calls `/v1/embeddings` on the same LM Studio server — no separate service needed.

## Performance Expectations

With Qwen 3.5 35B on an RTX 5070 Ti (~15 tok/s generation):

- Each tab takes roughly 15-30 seconds (fetch + LLM summary + LLM scoring + embedding)
- 500 tabs ≈ 3-4 hours
- Run it overnight or while doing other things

With Qwen 9B (~75 tok/s generation):

- Each tab takes roughly 5-10 seconds
- 500 tabs ≈ 45-90 minutes
- Quality will be somewhat lower but still useful for triage

## Future Extensions

This pipeline creates shared infrastructure for:

- **AI Digest Newsletters** (Project 3) — reuse the fetcher + summarizer for RSS feeds
- **Audio Pipeline** (Project 4) — pipe article text to local TTS
- **Recurring triage** — run monthly on emails/tabs that have accumulated
- **Clustering** — use the SQLite embeddings with UMAP + HDBSCAN to find natural topic groups and duplicates
