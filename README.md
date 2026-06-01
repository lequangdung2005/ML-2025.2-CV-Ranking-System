# CV Ranking System

AI-powered resume screening and evaluation pipeline.

The current source of truth for the intended architecture is `proposal.md` (OCR/VLM ingestion -> schema-bound extraction -> embedding retrieval + rerank -> LLM-as-a-judge scoring).

## Repo Status

This repository is currently scaffolded for Python development but does not yet include implemented pipeline code.

Phase 1 is now implemented: provider API integration (OpenAI-compatible), and an ingestion/OCR CLI that produces clean Markdown and stores artifacts locally.

## Development Setup

This repo is set up to be used via the devcontainer + Poetry.

1. Open in the devcontainer.
2. Dependencies are installed automatically via `.devcontainer/postCreateCommand.sh` (`poetry lock` then `poetry install`).
3. If this checkout is not a git repo, pre-commit hook installation will fail; initialize git first (`git init`) or skip hooks.

Run commands through Poetry:

```bash
poetry run ruff check .
```

Format:

```bash
poetry run ruff format .
```

## Phase 1: Run Locally

1. Configure `src/cv_ranking_system/config.py`:

Set `OCR_MODEL` (vision), `TEXT_MODEL` (extraction/judge), and ensure `API_KEY_PATH` points to a file containing your provider API key.

For ranking, set `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, and `EMBEDDING_API_KEY_PATH` for your embedding provider.

2. Ingest a resume:

```bash
poetry run cvrs ingest path/to/resume.pdf
```

Artifacts are stored on disk (default `./cvrs_data/`) under `raw/<doc_id>/...` and `artifacts/<doc_id>/ocr.md`.

3. Extract structured CV JSON:

```bash
poetry run cvrs extract <doc_id>
```

4. Build the local embedding index and rank against a JD:

```bash
poetry run cvrs index
poetry run cvrs rank --jd path/to/jd.txt --topk 20
```

5. Judge candidates:

```bash
poetry run cvrs judge --jd path/to/jd.txt --doc-id <doc_id>
# or judge the top K from a run's ranking.json
poetry run cvrs judge --jd path/to/jd.txt --run-id <run_id> --topk 10
```

## Implementation Roadmap (High Level)

1. Ingestion/OCR: CV PDFs/images -> structured Markdown/text.
2. Extraction: Markdown/text -> validated structured CV JSON.
3. Retrieval: embeddings + vector DB Top-K + cross-encoder rerank.
4. Judge: JD + CV JSON -> score (0-100) + strengths/weaknesses.
