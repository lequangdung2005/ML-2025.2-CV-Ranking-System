from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cv_ranking_system import config
from cv_ranking_system.llms.openai import OpenAIClient, RetryConfig
from cv_ranking_system.utils.utils import (
    embedding_api_key,
    embedding_base_url,
    embedding_model,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexBuildResult:
    num_documents: int
    metadata_path: str
    embeddings_path: str


def _cv_text_for_embedding(cv: dict) -> str:
    # Minimal, robust text serialization.
    parts: list[str] = []
    if isinstance(cv.get("summary"), str) and cv["summary"].strip():
        parts.append(f"Summary: {cv['summary'].strip()}")
    skills = cv.get("skills")
    if isinstance(skills, list) and skills:
        parts.append("Skills: " + ", ".join(str(s) for s in skills if str(s).strip()))
    exp = cv.get("experience")
    if isinstance(exp, list) and exp:
        lines: list[str] = []
        for item in exp:
            if not isinstance(item, dict):
                continue
            company = str(item.get("company") or "").strip()
            role = str(item.get("role") or "").strip()
            highlights = item.get("highlights")
            htxt = ""
            if isinstance(highlights, list) and highlights:
                htxt = " ".join(str(h) for h in highlights if str(h).strip())
            header = " ".join(x for x in [role, company] if x)
            if header or htxt:
                lines.append((header + ": " + htxt).strip(": "))
        if lines:
            parts.append("Experience: " + " | ".join(lines))
    edu = cv.get("education")
    if isinstance(edu, list) and edu:
        lines: list[str] = []
        for item in edu:
            if not isinstance(item, dict):
                continue
            inst = str(item.get("institution") or "").strip()
            degree = str(item.get("degree") or "").strip()
            fos = str(item.get("field_of_study") or "").strip()
            text = " ".join(x for x in [degree, fos, inst] if x)
            if text:
                lines.append(text)
        if lines:
            parts.append("Education: " + " | ".join(lines))
    certs = cv.get("certifications")
    if isinstance(certs, list) and certs:
        parts.append("Certifications: " + ", ".join(str(c) for c in certs if str(c).strip()))
    projects = cv.get("projects")
    if isinstance(projects, list) and projects:
        parts.append("Projects: " + " | ".join(str(p) for p in projects if str(p).strip()))
    return "\n".join(parts).strip()


def build_local_index(*, trace_id: str) -> IndexBuildResult:
    model = embedding_model()
    if not model:
        raise ValueError("Missing embedding model: set config.EMBEDDING_MODEL")

    root = Path(config.ARTIFACT_DIR)
    artifacts_dir = root / "artifacts"
    cv_paths = sorted(artifacts_dir.glob("*/cv.json"))

    logger.info(
        "Index build start",
        extra={"event": "index_build_start", "trace_id": trace_id, "model": model},
    )

    records: list[dict[str, str]] = []
    texts: list[str] = []
    for p in cv_paths:
        doc_id = p.parent.name
        cv = json.loads(p.read_text(encoding="utf-8"))
        text = _cv_text_for_embedding(cv)
        if not text:
            # Still include an empty placeholder to keep mapping explicit.
            text = ""
        records.append({"doc_id": doc_id, "cv_json_path": str(p)})
        texts.append(text)

    oai = OpenAIClient(
        base_url=embedding_base_url(),
        api_key=embedding_api_key(),
        timeout_s=config.CVRS_HTTP_TIMEOUT_S,
        retry=RetryConfig(
            max_retries=config.CVRS_HTTP_MAX_RETRIES,
            initial_backoff_s=config.CVRS_HTTP_INITIAL_BACKOFF_S,
            max_backoff_s=config.CVRS_HTTP_MAX_BACKOFF_S,
        ),
    )

    embs: list[list[float]] = []
    # Simple batching to avoid request limits.
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs.extend(oai.embed_texts(texts=batch, model=model, trace_id=trace_id))

    mat = np.asarray(embs, dtype=np.float32)
    if mat.ndim != 2 and len(records) > 0:
        raise ValueError("Embeddings matrix is not 2D")

    index_dir = root / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = index_dir / "metadata.jsonl"
    embeddings_path = index_dir / "embeddings.npy"

    metadata_path.write_text(
        "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in records), encoding="utf-8"
    )
    np.save(embeddings_path, mat)

    logger.info(
        "Index build complete",
        extra={
            "event": "index_build_complete",
            "trace_id": trace_id,
            "model": model,
            "path": str(index_dir),
        },
    )
    return IndexBuildResult(
        num_documents=len(records),
        metadata_path=str(metadata_path),
        embeddings_path=str(embeddings_path),
    )


def load_index(*, root_dir: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    index_dir = root_dir / "index"
    metadata_path = index_dir / "metadata.jsonl"
    embeddings_path = index_dir / "embeddings.npy"
    if not metadata_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError("Missing local index. Run `cvrs index` first.")

    records: list[dict[str, str]] = []
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    mat = np.load(embeddings_path)
    if len(records) != mat.shape[0]:
        raise ValueError("Index metadata/embeddings row count mismatch")
    return records, mat
