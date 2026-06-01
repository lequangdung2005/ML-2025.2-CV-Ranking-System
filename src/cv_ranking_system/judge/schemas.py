from __future__ import annotations

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    # Free-form references to fields in cv.json.
    references: list[str] = Field(default_factory=list)
    quotes: list[str] = Field(default_factory=list)


class JudgmentItem(BaseModel):
    statement: str
    evidence: Evidence = Field(default_factory=Evidence)


class Judgment(BaseModel):
    score_0_100: int
    strengths: list[JudgmentItem] = Field(default_factory=list)
    weaknesses: list[JudgmentItem] = Field(default_factory=list)
    must_haves_met: list[JudgmentItem] = Field(default_factory=list)
    must_haves_missing: list[JudgmentItem] = Field(default_factory=list)
