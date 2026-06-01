from __future__ import annotations

import json

import pytest

from cv_ranking_system import config
from cv_ranking_system.judge.evaluate import evaluate_candidate


class _FakeJudgeOAI:
    def __init__(self, response: str):
        self._response = response

    def chat_text(self, *, prompt: str, model: str, trace_id: str) -> str:
        return self._response


def test_evaluate_candidate_schema(monkeypatch, tmp_path) -> None:
    config.TEXT_MODEL = "text"
    config.OPENAI_BASE_URL = "https://example.test"
    config.API_KEY_PATH = str(tmp_path / "k")
    (tmp_path / "k").write_text("x", encoding="utf-8")

    resp = json.dumps(
        {
            "score_0_100": 77,
            "strengths": [
                {
                    "statement": "Python experience",
                    "evidence": {"references": ["skills[0]"], "quotes": ["Python"]},
                }
            ],
            "weaknesses": [],
            "must_haves_met": [],
            "must_haves_missing": [],
        }
    )
    monkeypatch.setattr(
        "cv_ranking_system.judge.evaluate.OpenAIClient", lambda **_: _FakeJudgeOAI(resp)
    )
    j = evaluate_candidate(jd_text="x", cv={"skills": ["Python"]}, trace_id="t")
    assert j.score_0_100 == 77


def test_evaluate_candidate_rejects_out_of_range(monkeypatch, tmp_path) -> None:
    config.TEXT_MODEL = "text"
    config.OPENAI_BASE_URL = "https://example.test"
    config.API_KEY_PATH = str(tmp_path / "k")
    (tmp_path / "k").write_text("x", encoding="utf-8")

    resp = json.dumps(
        {
            "score_0_100": 177,
            "strengths": [],
            "weaknesses": [],
            "must_haves_met": [],
            "must_haves_missing": [],
        }
    )
    monkeypatch.setattr(
        "cv_ranking_system.judge.evaluate.OpenAIClient", lambda **_: _FakeJudgeOAI(resp)
    )
    with pytest.raises(ValueError, match="out of range"):
        evaluate_candidate(jd_text="x", cv={}, trace_id="t")
