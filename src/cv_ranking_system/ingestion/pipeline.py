from __future__ import annotations

import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from cv_ranking_system import config
from cv_ranking_system.ingestion.render import normalize_image_to_png, render_pdf_to_png_pages
from cv_ranking_system.llms.openai import OpenAIClient, RetryConfig
from cv_ranking_system.storage.local import build_paths, put_bytes
from cv_ranking_system.utils.utils import ocr_model, provider_api_key, provider_base_url, sha256_hex

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    raw_path: str
    markdown_path: str


def _guess_content_type(path: Path) -> str:
    ct, _ = mimetypes.guess_type(str(path))
    return ct or "application/octet-stream"


def ingest_document(
    *,
    path: Path,
    trace_id: str,
    dpi: int = 200,
) -> IngestResult:
    data = path.read_bytes()
    doc_id = sha256_hex(data)[:32]

    logger.info(
        "Ingest start",
        extra={"event": "ingest_start", "trace_id": trace_id, "doc_id": doc_id, "path": str(path)},
    )

    root_dir = Path(config.ARTIFACT_DIR)
    paths = build_paths(root_dir=root_dir, doc_id=doc_id, source_name=path.name)
    put_bytes(path=paths.raw_path, data=data)

    suffix = path.suffix.lower()
    images: list[bytes]
    if suffix == ".pdf":
        pages = render_pdf_to_png_pages(data, dpi=dpi)
        images = [p.png_bytes for p in pages]
    else:
        # Assume image input.
        images = [normalize_image_to_png(data)]

    model = ocr_model()
    if not model:
        raise ValueError("Missing OCR model: set config.OCR_MODEL")

    oai = OpenAIClient(
        base_url=provider_base_url(),
        api_key=provider_api_key(),
        timeout_s=config.CVRS_HTTP_TIMEOUT_S,
        retry=RetryConfig(
            max_retries=config.CVRS_HTTP_MAX_RETRIES,
            initial_backoff_s=config.CVRS_HTTP_INITIAL_BACKOFF_S,
            max_backoff_s=config.CVRS_HTTP_MAX_BACKOFF_S,
        ),
    )
    md, usage = oai.ocr_markdown_from_images(
        images,
        model=model,
        trace_id=trace_id,
    )

    logger.info(
        "OCR complete",
        extra={
            "event": "ocr_complete",
            "trace_id": trace_id,
            "doc_id": doc_id,
            "provider": "openai",
            "model": model,
        },
    )
    if usage.total_tokens is not None:
        logger.info(
            "Token usage",
            extra={
                "event": "token_usage",
                "trace_id": trace_id,
                "doc_id": doc_id,
                "provider": "openai",
                "model": model,
            },
        )

    put_bytes(path=paths.markdown_path, data=(md + "\n").encode("utf-8"))

    logger.info(
        "Ingest complete",
        extra={
            "event": "ingest_complete",
            "trace_id": trace_id,
            "doc_id": doc_id,
            "path": str(path),
        },
    )

    return IngestResult(
        doc_id=doc_id,
        raw_path=str(paths.raw_path),
        markdown_path=str(paths.markdown_path),
    )
