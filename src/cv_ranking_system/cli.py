from __future__ import annotations

import argparse
from pathlib import Path

from cv_ranking_system import config
from cv_ranking_system.ingestion.pipeline import ingest_document
from cv_ranking_system.utils.logging_utils import Trace, setup_logging


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cvrs", description="CV Ranking System CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="Ingest a CV (PDF/image) and produce OCR markdown")
    ingest.add_argument("path", type=Path, help="Path to a PDF or image file")
    ingest.add_argument("--dpi", type=int, default=200, help="PDF render DPI")

    extract = sub.add_parser("extract", help="Extract structured CV JSON from OCR markdown")
    extract.add_argument("doc_id", type=str, help="Document ID from ingest")

    sub.add_parser("index", help="Build a local embedding index from extracted CV JSON")

    rank = sub.add_parser("rank", help="Rank candidates for a job description (JD)")
    rank.add_argument(
        "--jd", type=Path, required=True, help="Path to a text file containing the JD"
    )
    rank.add_argument("--topk", type=int, default=20, help="Number of candidates to return")

    judge = sub.add_parser("judge", help="LLM judge: score and explain CV(s) against a JD")
    judge.add_argument(
        "--jd", type=Path, required=True, help="Path to a text file containing the JD"
    )
    g = judge.add_mutually_exclusive_group(required=True)
    g.add_argument("--doc-id", type=str, help="Judge a single candidate doc_id")
    g.add_argument("--topk", type=int, help="Judge the top K from a rank run")
    judge.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Existing run_id to reuse (defaults to creating a new run)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "ingest":
        # Configuration comes from cv_ranking_system/config.py.
        setup_logging(level=config.LOG_LEVEL)
        trace = Trace.new()
        result = ingest_document(path=args.path, trace_id=trace.trace_id, dpi=args.dpi)
        # Human-friendly final line (still plain text, not structured logs)
        print(
            f"doc_id={result.doc_id} raw_path={result.raw_path} "
            f"markdown_path={result.markdown_path}"
        )
        return 0

    if args.cmd == "extract":
        setup_logging(level=config.LOG_LEVEL)
        from cv_ranking_system.extraction.extract import extract_structured_cv

        trace = Trace.new()
        out = extract_structured_cv(doc_id=args.doc_id, trace_id=trace.trace_id)
        print(f"doc_id={args.doc_id} cv_json_path={out.cv_json_path}")
        return 0

    if args.cmd == "index":
        setup_logging(level=config.LOG_LEVEL)
        from cv_ranking_system.retrieval.index import build_local_index

        trace = Trace.new()
        out = build_local_index(trace_id=trace.trace_id)
        print(
            f"indexed={out.num_documents} metadata_path={out.metadata_path} "
            f"embeddings_path={out.embeddings_path}"
        )
        return 0

    if args.cmd == "rank":
        setup_logging(level=config.LOG_LEVEL)
        from cv_ranking_system.retrieval.rank import rank_for_jd

        trace = Trace.new()
        jd_text = args.jd.read_text(encoding="utf-8")
        out = rank_for_jd(jd_text=jd_text, top_k=args.topk, trace_id=trace.trace_id)
        print(f"run_id={out.run_id} ranking_path={out.ranking_path}")
        return 0

    if args.cmd == "judge":
        setup_logging(level=config.LOG_LEVEL)
        from cv_ranking_system.judge.run import judge_candidates

        trace = Trace.new()
        jd_text = args.jd.read_text(encoding="utf-8")
        out = judge_candidates(
            jd_text=jd_text,
            trace_id=trace.trace_id,
            run_id=args.run_id,
            doc_id=args.doc_id,
            top_k=args.topk,
        )
        print(f"run_id={out.run_id} judgments_dir={out.judgments_dir}")
        return 0

    raise AssertionError(f"Unhandled cmd: {args.cmd}")
