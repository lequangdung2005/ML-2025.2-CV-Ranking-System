from __future__ import annotations

"""Project configuration constants.

Keep this file minimal: only settings used by the current CV ranking pipeline code.
"""

# Logging
LOG_LEVEL: str = "INFO"

# LLM/VLM (OpenAI-compatible)
# These defaults are meant to be a sensible starting point for OpenAI-style APIs.
# If your provider uses different model IDs, change them here.
OCR_MODEL: str = "gpt-4o-mini"  # must support vision inputs
TEXT_MODEL: str = "gpt-4o-mini"  # used for extraction + judge

# Embeddings (OpenAI-compatible endpoint)
EMBEDDING_BASE_URL: str | None = None
EMBEDDING_API_KEY_PATH: str = "../keys/embedding_api_key.key"
EMBEDDING_MODEL: str = "text-embedding-3-small"

# Optional override for OpenAI-compatible endpoints.
OPENAI_BASE_URL: str | None = None

# API key handling
# Store one key per file; content is read and stripped.
API_KEY_PATH: str = "../keys/api_key.key"
# HTTP client
CVRS_HTTP_TIMEOUT_S: float = 60
CVRS_HTTP_MAX_RETRIES: int = 6
CVRS_HTTP_INITIAL_BACKOFF_S: float = 1
CVRS_HTTP_MAX_BACKOFF_S: float = 20

# Local artifacts
ARTIFACT_DIR: str = "./cvrs_data"
