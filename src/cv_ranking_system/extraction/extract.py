from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from cv_ranking_system import config
from cv_ranking_system.extraction.schemas import CVStructured
from cv_ranking_system.llms.openai import OpenAIClient, RetryConfig
from cv_ranking_system.storage.local import build_paths, put_bytes
from cv_ranking_system.utils.utils import provider_api_key, provider_base_url, text_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractResult:
    doc_id: str
    cv_json_path: str


def _extract_prompt(md: str) -> str:
    # Keep prompt small and deterministic; ask for JSON only.
    return (
        "Extract a structured resume JSON object from the OCR markdown below. "
        "Return JSON only (no markdown). Use null for missing fields. "
        "Populate: name,email,phone,location,links[],summary,skills[],experience[],education[],"
        "certifications[],projects[]. "
        "experience items: company,role,start_date,end_date,location,highlights[]. "
        "education items: institution,degree,field_of_study,start_date,end_date,notes[].\n\n"
        f"OCR_MARKDOWN:\n{md}"
    )


def _repair_prompt(md: str, *, errors: str, prior_json: str) -> str:
    return (
        "The JSON you returned did not validate against the required schema. "
        "Fix the JSON to satisfy the schema. Return JSON only.\n\n"
        f"VALIDATION_ERRORS:\n{errors}\n\n"
        f"OCR_MARKDOWN:\n{md}\n\n"
        f"YOUR_PREVIOUS_JSON:\n{prior_json}"
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("Model returned JSON but not an object")
    return data


def extract_structured_cv(*, doc_id: str, trace_id: str) -> ExtractResult:
    root_dir = Path(config.ARTIFACT_DIR)
    paths = build_paths(root_dir=root_dir, doc_id=doc_id, source_name="")
    md_path = paths.markdown_path
    if not md_path.exists():
        raise FileNotFoundError(f"Missing OCR markdown for doc_id={doc_id}: {md_path}")
    md = md_path.read_text(encoding="utf-8")

    model = text_model()
    if not model:
        raise ValueError("Missing TEXT model: set config.TEXT_MODEL")

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

    logger.info(
        "Extraction start",
        extra={"event": "extract_start", "trace_id": trace_id, "doc_id": doc_id, "model": model},
    )

    raw = oai.chat_text(prompt=_extract_prompt(md), model=model, trace_id=trace_id)
    try:
        obj = _parse_json_object(raw)
        cv = CVStructured.model_validate(obj)
    except (ValueError, ValidationError) as e:
        # One repair attempt with explicit errors.
        errors = str(e)
        repaired = oai.chat_text(
            prompt=_repair_prompt(md, errors=errors, prior_json=raw),
            model=model,
            trace_id=trace_id,
        )
        obj2 = _parse_json_object(repaired)
        cv = CVStructured.model_validate(obj2)

    cv_json_path = root_dir / "artifacts" / doc_id / "cv.json"
    put_bytes(path=cv_json_path, data=(cv.model_dump_json(indent=2) + "\n").encode("utf-8"))

    logger.info(
        "Extraction complete",
        extra={
            "event": "extract_complete",
            "trace_id": trace_id,
            "doc_id": doc_id,
            "model": model,
            "path": str(cv_json_path),
        },
    )
    return ExtractResult(doc_id=doc_id, cv_json_path=str(cv_json_path))
