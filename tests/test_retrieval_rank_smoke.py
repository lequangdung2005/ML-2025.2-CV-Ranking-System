from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cv_ranking_system import config
from cv_ranking_system.retrieval.index import build_local_index
from cv_ranking_system.retrieval.rank import rank_for_jd


class _FakeEmbedOAI:
    def embed_texts(self, *, texts: list[str], model: str, trace_id: str) -> list[list[float]]:
        # 3-dim toy embedding: [len, count_python, count_java]
        out: list[list[float]] = []
        for t in texts:
            tl = t.lower()
            out.append([float(len(tl)), float(tl.count("python")), float(tl.count("java"))])
        return out


def test_build_index_and_rank(monkeypatch, tmp_path: Path) -> None:
    config.ARTIFACT_DIR = str(tmp_path)
    config.EMBEDDING_MODEL = "embed"
    config.EMBEDDING_BASE_URL = "https://embed.test"
    config.EMBEDDING_API_KEY_PATH = str(tmp_path / "ek")
    (tmp_path / "ek").write_text("x", encoding="utf-8")

    # Two CVs.
    for did, skills in [("a", ["Python"]), ("b", ["Java"])]:
        p = tmp_path / "artifacts" / did / "cv.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"skills": skills, "experience": [], "education": []}), encoding="utf-8"
        )

    monkeypatch.setattr(
        "cv_ranking_system.retrieval.index.OpenAIClient", lambda **_: _FakeEmbedOAI()
    )
    monkeypatch.setattr(
        "cv_ranking_system.retrieval.rank.OpenAIClient", lambda **_: _FakeEmbedOAI()
    )

    build_local_index(trace_id="t")
    res = rank_for_jd(jd_text="Need python", top_k=1, trace_id="t")
    ranking = json.loads(Path(res.ranking_path).read_text(encoding="utf-8"))
    assert ranking["results"][0]["doc_id"] == "a"

    # Ensure embeddings.npy is loadable.
    mat = np.load(tmp_path / "index" / "embeddings.npy")
    assert mat.shape[0] == 2
