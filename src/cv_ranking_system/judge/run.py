from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from cv_ranking_system import config
from cv_ranking_system.judge.evaluate import evaluate_candidate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeRunResult:
    run_id: str
    judgments_dir: str


def _load_ranking_doc_ids(*, run_dir: Path, top_k: int) -> list[str]:
    ranking_path = run_dir / "ranking.json"
    if not ranking_path.exists():
        raise FileNotFoundError(f"Missing ranking.json in run: {run_dir}")
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    results = ranking.get("results")
    if not isinstance(results, list):
        raise ValueError("ranking.json missing results")
    doc_ids: list[str] = []
    for item in results[:top_k]:
        if isinstance(item, dict) and isinstance(item.get("doc_id"), str):
            doc_ids.append(item["doc_id"])
    return doc_ids


def judge_candidates(
    *,
    jd_text: str,
    trace_id: str,
    run_id: str | None,
    doc_id: str | None,
    top_k: int | None,
) -> JudgeRunResult:
    root = Path(config.ARTIFACT_DIR)
    if run_id is None:
        run_id = uuid.uuid4().hex[:12]
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    judgments_dir = run_dir / "judgments"
    judgments_dir.mkdir(parents=True, exist_ok=True)

    if doc_id is not None:
        doc_ids = [doc_id]
    else:
        if top_k is None:
            raise ValueError("top_k required when doc_id is not provided")
        doc_ids = _load_ranking_doc_ids(run_dir=run_dir, top_k=top_k)
        if not doc_ids:
            raise ValueError("No doc_ids found to judge")

    logger.info(
        "Judge run start",
        extra={"event": "judge_run_start", "trace_id": trace_id, "path": str(run_dir)},
    )
    for did in doc_ids:
        cv_path = root / "artifacts" / did / "cv.json"
        if not cv_path.exists():
            raise FileNotFoundError(f"Missing extracted cv.json for doc_id={did}: {cv_path}")
        cv = json.loads(cv_path.read_text(encoding="utf-8"))
        judgment = evaluate_candidate(jd_text=jd_text, cv=cv, trace_id=trace_id)
        out_path = judgments_dir / f"{did}.json"
        out_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "doc_id": did,
                    "ts": int(time.time()),
                    "judgment": judgment.model_dump(),
                },
                ensure_ascii=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    logger.info(
        "Judge run complete",
        extra={"event": "judge_run_complete", "trace_id": trace_id, "path": str(judgments_dir)},
    )
    return JudgeRunResult(run_id=run_id, judgments_dir=str(judgments_dir))
