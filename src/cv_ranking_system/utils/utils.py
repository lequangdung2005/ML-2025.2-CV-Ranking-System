from __future__ import annotations

import hashlib
from pathlib import Path

from cv_ranking_system import config


def sha256_hex(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def read_key_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def provider_api_key() -> str:
    key_path = config.API_KEY_PATH.strip()
    if not key_path:
        raise ValueError("Missing provider API key path: set config.API_KEY_PATH")
    return read_key_file(key_path)


def provider_base_url() -> str:
    return (config.OPENAI_BASE_URL or "").strip() or "https://api.openai.com/v1"


def ocr_model() -> str:
    return config.OCR_MODEL.strip()


def text_model() -> str:
    return config.TEXT_MODEL.strip()


def embedding_api_key() -> str:
    key_path = config.EMBEDDING_API_KEY_PATH.strip()
    if not key_path:
        raise ValueError("Missing embedding API key path: set config.EMBEDDING_API_KEY_PATH")
    return read_key_file(key_path)


def embedding_base_url() -> str:
    return (config.EMBEDDING_BASE_URL or "").strip() or "https://api.openai.com/v1"


def embedding_model() -> str:
    return config.EMBEDDING_MODEL.strip()
