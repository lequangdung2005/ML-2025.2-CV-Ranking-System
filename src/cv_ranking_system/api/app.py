from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from cv_ranking_system import config
from cv_ranking_system.extraction.extract import extract_structured_cv
from cv_ranking_system.ingestion.pipeline import ingest_document
from cv_ranking_system.judge.run import judge_candidates
from cv_ranking_system.retrieval.index import build_local_index
from cv_ranking_system.retrieval.rank import rank_for_jd
from cv_ranking_system.utils.logging_utils import Trace

app = FastAPI(title="CV Ranking System", version="0.1")


class RankRequest(BaseModel):
    jd_text: str
    top_k: int = 20


class JudgeRequest(BaseModel):
    jd_text: str
    doc_id: str | None = None
    run_id: str | None = None
    top_k: int | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile, dpi: int = 200) -> dict[str, str]:
    trace = Trace.new()
    tmp_dir = Path(config.ARTIFACT_DIR) / "_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / file.filename
    tmp_path.write_bytes(await file.read())
    try:
        res = ingest_document(path=tmp_path, trace_id=trace.trace_id, dpi=dpi)
        return {"doc_id": res.doc_id, "markdown_path": res.markdown_path, "raw_path": res.raw_path}
    finally:
        # Best-effort cleanup.
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/extract/{doc_id}")
def extract(doc_id: str) -> dict[str, str]:
    trace = Trace.new()
    try:
        res = extract_structured_cv(doc_id=doc_id, trace_id=trace.trace_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"doc_id": res.doc_id, "cv_json_path": res.cv_json_path}


@app.post("/index")
def build_index() -> dict[str, str | int]:
    trace = Trace.new()
    res = build_local_index(trace_id=trace.trace_id)
    return {
        "num_documents": res.num_documents,
        "metadata_path": res.metadata_path,
        "embeddings_path": res.embeddings_path,
    }


@app.post("/rank")
def rank(req: RankRequest) -> dict:
    trace = Trace.new()
    res = rank_for_jd(jd_text=req.jd_text, top_k=req.top_k, trace_id=trace.trace_id)
    return json.loads(Path(res.ranking_path).read_text(encoding="utf-8"))


@app.post("/judge")
def judge(req: JudgeRequest) -> dict[str, str]:
    trace = Trace.new()
    if req.doc_id is None and req.top_k is None:
        raise HTTPException(status_code=400, detail="Provide doc_id or top_k")
    res = judge_candidates(
        jd_text=req.jd_text,
        trace_id=trace.trace_id,
        run_id=req.run_id,
        doc_id=req.doc_id,
        top_k=req.top_k,
    )
    return {"run_id": res.run_id, "judgments_dir": res.judgments_dir}
