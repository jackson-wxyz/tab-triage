"""
LLM Client module: wraps LM Studio's OpenAI-compatible API.

Provides chat completions and embeddings with retry logic.
Includes a mock mode that returns realistic fake responses for testing
the pipeline without a running LM Studio instance.
"""

import json
import logging
import time
import hashlib
import random

import requests

import config

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for LM Studio's OpenAI-compatible API."""

    def __init__(self, mock: bool = None):
        self.mock = mock if mock is not None else config.MOCK_MODE
        self.base_url = config.LM_STUDIO_BASE_URL
        self.api_key = config.LM_STUDIO_API_KEY
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        })

        if self.mock:
            logger.info("LLM Client running in MOCK MODE")
        else:
            logger.info(f"LLM Client connecting to {self.base_url}")

    def chat(self, messages: list[dict], temperature: float = None,
             max_tokens: int = None) -> str:
        """
        Send a chat completion request.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.
            temperature: Override config.CHAT_TEMPERATURE.
            max_tokens: Override config.CHAT_MAX_TOKENS.

        Returns:
            The assistant's response text.
        """
        if self.mock:
            return self._mock_chat(messages)

        payload = {
            "model": config.CHAT_MODEL,
            "messages": messages,
            "temperature": temperature or config.CHAT_TEMPERATURE,
            "max_tokens": max_tokens or config.CHAT_MAX_TOKENS,
        }

        return self._request_with_retry(
            f"{self.base_url}/chat/completions",
            payload,
            extract_fn=lambda r: r["choices"][0]["message"]["content"],
        )

    def embed(self, text: str) -> list[float]:
        """
        Get an embedding vector for a piece of text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats (the embedding vector).
        """
        if self.mock:
            return self._mock_embed(text)

        # Truncate very long texts for embedding (most models cap at ~8k tokens)
        if len(text) > 8000:
            text = text[:8000]

        payload = {
            "model": config.EMBEDDING_MODEL,
            "input": text,
        }

        return self._request_with_retry(
            f"{self.base_url}/embeddings",
            payload,
            extract_fn=lambda r: r["data"][0]["embedding"],
        )

    def _request_with_retry(self, url: str, payload: dict,
                            extract_fn) -> any:
        """Make an API request with retry logic."""
        last_error = None

        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                resp = self.session.post(url, json=payload,
                                         timeout=120)
                resp.raise_for_status()
                data = resp.json()
                return extract_fn(data)

            except requests.exceptions.Timeout:
                last_error = "Request timed out"
                logger.warning(f"  Attempt {attempt}: timeout")
            except requests.exceptions.ConnectionError:
                last_error = "Connection failed — is LM Studio running?"
                logger.warning(f"  Attempt {attempt}: connection error")
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP {e.response.status_code}"
                logger.warning(f"  Attempt {attempt}: {last_error}")
                # Don't retry on 4xx (bad request, not a transient error)
                if e.response.status_code < 500:
                    break
            except (KeyError, json.JSONDecodeError) as e:
                # Show what we actually got back — crucial for diagnosing
                # path/routing issues (e.g. missing /v1/ prefix)
                body_preview = ""
                try:
                    body_preview = resp.text[:300]
                except Exception:
                    pass
                last_error = f"Unexpected response format: {e}"
                logger.warning(f"  Attempt {attempt}: {last_error}")
                if body_preview:
                    logger.warning(f"  Response body: {body_preview}")
                    logger.warning(
                        f"  (Check that LM_STUDIO_BASE_URL ends with /v1 "
                        f"— current: {self.base_url})"
                    )
                break
            except Exception as e:
                last_error = str(e)
                logger.warning(f"  Attempt {attempt}: {last_error}")

            if attempt < config.MAX_RETRIES:
                time.sleep(config.RETRY_DELAY)

        raise RuntimeError(f"LLM request failed after {config.MAX_RETRIES} "
                           f"attempts: {last_error}")

    # ── Mock implementations ──────────────────────────────────────────

    def _mock_chat(self, messages: list[dict]) -> str:
        """
        Return a realistic fake JSON response for testing.
        Uses the content of the last message to generate deterministic
        but varied mock data.
        """
        last_msg = messages[-1]["content"] if messages else ""

        # Determine what kind of response is expected based on the system prompt
        system_msg = ""
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
                break

        # Generate a deterministic seed from the full input (not truncated,
        # since the template prefix is the same for all URLs)
        seed = int(hashlib.md5(last_msg.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        categories = [
            "AI safety", "health/nutrition", "personal finance",
            "home improvement", "rationality/self-improvement",
            "space industry", "gaming", "EA/effective altruism",
            "technology/software", "politics/policy", "science",
            "philosophy", "career/productivity", "travel",
        ]

        actions = [
            "Read and take notes on key frameworks",
            "Research and compare options, then purchase",
            "Schedule time to implement the suggested changes",
            "Save for reference — revisit when relevant",
            "Share with partner and discuss",
            "Add to reading list for weekend",
            "Extract the recipe/instructions and try this week",
            "Follow up on the recommendation mentioned",
            "No specific action — interesting background reading",
            "Bookmark and revisit quarterly",
        ]

        mock_response = {
            "title": f"[Mock] Article about {rng.choice(categories).lower()}",
            "summary": (
                f"This article discusses key developments in "
                f"{rng.choice(categories).lower()}. "
                f"The author argues for a nuanced approach, presenting "
                f"evidence from multiple sources and suggesting practical "
                f"takeaways for readers."
            ),
            "category": rng.choice(categories),
            "actionability": rng.randint(1, 5),
            "implied_action": rng.choice(actions),
            "importance": rng.randint(1, 5),
            "effort": rng.randint(1, 5),
            "staleness": rng.randint(1, 5),
            "insight_density": rng.randint(1, 5),
        }

        time.sleep(0.05)  # simulate a tiny delay
        return json.dumps(mock_response)

    def _mock_embed(self, text: str) -> list[float]:
        """
        Return a fake embedding vector of the right dimensionality.
        Uses a hash of the input so the same text always gets the same vector,
        which makes clustering tests reproducible.
        """
        # 768 dimensions (matches nomic-embed-text-v1.5)
        seed = int(hashlib.md5(text[:500].encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        vector = [rng.gauss(0, 1) for _ in range(768)]
        # Normalize to unit length (as real embedding models do)
        norm = sum(x * x for x in vector) ** 0.5
        vector = [x / norm for x in vector]
        return vector
