from __future__ import annotations

from cv_ranking_system.extraction.schemas import CVStructured


def test_cvstructured_schema_accepts_minimal() -> None:
    cv = CVStructured.model_validate({})
    assert cv.skills == []
    assert cv.experience == []
    assert cv.education == []


def test_cvstructured_schema_accepts_nested() -> None:
    cv = CVStructured.model_validate(
        {
            "name": "A",
            "skills": ["Python"],
            "experience": [
                {
                    "company": "C",
                    "role": "R",
                    "highlights": ["Did X"],
                }
            ],
            "education": [{"institution": "U", "degree": "BSc"}],
        }
    )
    assert cv.name == "A"
    assert cv.experience[0].company == "C"
