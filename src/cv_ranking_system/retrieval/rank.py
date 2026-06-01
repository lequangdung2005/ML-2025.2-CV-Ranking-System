from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cv_ranking_system import config
from cv_ranking_system.llms.openai import OpenAIClient, RetryConfig
from cv_ranking_system.retrieval.index import load_index
from cv_ranking_system.utils.utils import (
    embedding_api_key,
    embedding_base_url,
    embedding_model,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankResult:
    run_id: str
    ranking_path: str


def _cosine_scores(matrix: np.ndarray, v: np.ndarray) -> np.ndarray:
    # Avoid division by zero by adding a tiny epsilon.
    eps = 1e-12
    v_norm = np.linalg.norm(v) + eps
    m_norm = np.linalg.norm(matrix, axis=1) + eps
    return (matrix @ v) / (m_norm * v_norm)


def rank_for_jd(*, jd_text: str, top_k: int, trace_id: str) -> RankResult:
    root = Path(config.ARTIFACT_DIR)
    records, mat = load_index(root_dir=root)
    model = embedding_model()
    if not model:
        raise ValueError("Missing embedding model: set config.EMBEDDING_MODEL")

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
    q = oai.embed_texts(texts=[jd_text], model=model, trace_id=trace_id)[0]
    qv = np.asarray(q, dtype=np.float32)

    if mat.size == 0:
        raise ValueError("Index is empty. Add CVs and run `cvrs extract` then `cvrs index`.")
    if qv.shape[0] != mat.shape[1]:
        raise ValueError("Embedding dimension mismatch between query and index")

    scores = _cosine_scores(mat, qv)
    k = min(max(int(top_k), 1), len(records))
    idx = np.argsort(-scores)[:k]
    ranked = [
        {
            "doc_id": records[int(i)]["doc_id"],
            "score": float(scores[int(i)]),
            "cv_json_path": records[int(i)]["cv_json_path"],
        }
        for i in idx
    ]

    run_id = uuid.uuid4().hex[:12]
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = run_dir / "ranking.json"
    payload = {
        "run_id": run_id,
        "ts": int(time.time()),
        "top_k": k,
        "embedding_model": model,
        "results": ranked,
    }
    ranking_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )

    logger.info(
        "Rank complete",
        extra={"event": "rank_complete", "trace_id": trace_id, "path": str(ranking_path)},
    )
    return RankResult(run_id=run_id, ranking_path=str(ranking_path))
