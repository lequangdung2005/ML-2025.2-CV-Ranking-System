from __future__ import annotations

import json
from pathlib import Path

import pytest

from cv_ranking_system import config
from cv_ranking_system.extraction.extract import extract_structured_cv


class _FakeOAI:
    def __init__(self, responses: list[str]):
        self._responses = responses
        self.calls: list[str] = []

    def chat_text(self, *, prompt: str, model: str, trace_id: str) -> str:
        self.calls.append(prompt)
        return self._responses.pop(0)


def test_extract_structured_cv_repairs_invalid_json(monkeypatch, tmp_path: Path) -> None:
    # Arrange local artifact layout.
    config.ARTIFACT_DIR = str(tmp_path)
    doc_id = "abcd"
    md_path = tmp_path / "artifacts" / doc_id / "ocr.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("# Resume\nSkills: Python\n", encoding="utf-8")

    config.TEXT_MODEL = "text"
    config.OPENAI_BASE_URL = "https://example.test"
    config.API_KEY_PATH = str(tmp_path / "k")
    (tmp_path / "k").write_text("x", encoding="utf-8")

    fake = _FakeOAI(
        responses=[
            "{not json}",
            json.dumps({"skills": ["Python"], "experience": [], "education": []}),
        ]
    )

    monkeypatch.setattr("cv_ranking_system.extraction.extract.OpenAIClient", lambda **_: fake)

    # Act
    res = extract_structured_cv(doc_id=doc_id, trace_id="t")

    # Assert
    out = json.loads(Path(res.cv_json_path).read_text(encoding="utf-8"))
    assert out["skills"] == ["Python"]
    assert len(fake.calls) == 2


def test_extract_structured_cv_requires_md(monkeypatch, tmp_path: Path) -> None:
    config.ARTIFACT_DIR = str(tmp_path)
    config.TEXT_MODEL = "text"
    config.OPENAI_BASE_URL = "https://example.test"
    config.API_KEY_PATH = str(tmp_path / "k")
    (tmp_path / "k").write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "cv_ranking_system.extraction.extract.OpenAIClient", lambda **_: _FakeOAI([])
    )
    with pytest.raises(FileNotFoundError):
        extract_structured_cv(doc_id="missing", trace_id="t")
