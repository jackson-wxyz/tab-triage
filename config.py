"""
Configuration for the Tab Triage Pipeline.

When deploying on your desktop, update LM_STUDIO_BASE_URL and model names
to match your LM Studio setup. Everything else should work out of the box.
"""

# ── LM Studio Connection ──────────────────────────────────────────────
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

# Model names as they appear in LM Studio's model list
CHAT_MODEL = "qwen/qwen3.5-35b-a3b" # or qwen/qwen3.5-9b
EMBEDDING_MODEL = "text-embedding-jina-embeddings-v5-text-nano-clustering"  # or embeddinggemma-300m-GGUF

# If LM Studio requires an API key (check LM Studio > Server settings)
LM_STUDIO_API_KEY = "sk-lm-FaiipYw6:FVNsw8bZvEgxp61tn10X" 

# ── LLM Parameters ────────────────────────────────────────────────────
CHAT_TEMPERATURE = 0.5      # low temp for consistent structured output
CHAT_MAX_TOKENS =10000      # enough for thinking + summary + scores + action items
CONTEXT_WINDOW = 16384      # 16384 for qwen 35b; up to 65536 for qwen 9b

# ── Pipeline Settings ─────────────────────────────────────────────────
# Max characters of article text to send to the LLM.
# Qwen 3.5 35B at ~15 tok/s means we want to keep prompts reasonable.
# ~16000 chars ≈ ~4000 tokens of article content.
MAX_ARTICLE_CHARS = 16000 #about 4000 tokens -- can do much more for 9b

# Max characters of article text to send to the embedding model.
# ~4 chars/token, so 6000 chars ≈ 1500 tokens — safe for 2048-token models.
# Bump higher if using a model with a larger context window.
MAX_EMBED_CHARS = 24000 #24000 for jina, 6000 for gemma's 2048 limit

# How many seconds to wait between LLM calls (be kind to your GPU)
REQUEST_DELAY = 0.1

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