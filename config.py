"""
Configuration for the Tab Triage Pipeline.

When deploying on your desktop, update LM_STUDIO_BASE_URL and model names
to match your LM Studio setup. Everything else should work out of the box.
"""

# ── LM Studio Connection ──────────────────────────────────────────────
LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"

# Model names as they appear in LM Studio's model list
CHAT_MODEL = "qwen/qwen3.5-9b"           # adjust to match your loaded model name
EMBEDDING_MODEL = "embeddinggemma-300m-GGUF"  # or mxbai-embed-large-v1

# If LM Studio requires an API key (check LM Studio > Server settings)
LM_STUDIO_API_KEY = "sk-lm-FaiipYw6:FVNsw8bZvEgxp61tn10X" 

# ── LLM Parameters ────────────────────────────────────────────────────
CHAT_TEMPERATURE = 0.5      # low temp for consistent structured output
CHAT_MAX_TOKENS = 1500      # enough for summary + scores + action items
CONTEXT_WINDOW = 16384      # for qwen 35b; adjust per model

# ── Pipeline Settings ─────────────────────────────────────────────────
# Max characters of article text to send to the LLM.
# Qwen 3.5 35B at ~15 tok/s means we want to keep prompts reasonable.
# ~16000 chars ≈ ~4000 tokens of article content.
MAX_ARTICLE_CHARS = 16000

# How many seconds to wait between LLM calls (be kind to your GPU)
REQUEST_DELAY = 0.5

# Retry settings for failed requests
MAX_RETRIES = 3
RETRY_DELAY = 5.0

# ── Fetcher Settings ──────────────────────────────────────────────────
FETCH_TIMEOUT = 15          # seconds per URL
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# ── Output Settings ───────────────────────────────────────────────────
import os as _os
OUTPUT_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "output")
CSV_FILENAME = "triage_results.csv"
SQLITE_FILENAME = "triage_results.db"

# ── Mock Mode ─────────────────────────────────────────────────────────
# Set to True to run the full pipeline with fake LLM responses.
# Useful for testing fetcher, output formatting, etc. without LM Studio.
MOCK_MODE = False
