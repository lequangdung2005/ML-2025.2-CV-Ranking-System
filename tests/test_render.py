from __future__ import annotations

import pytest

from cv_ranking_system.ingestion.render import normalize_image_to_png, render_pdf_to_png_pages


def test_render_pdf_to_png_pages_smoke() -> None:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.insert_text((10, 20), "Hello")
    pdf_bytes = doc.tobytes()

    pages = render_pdf_to_png_pages(pdf_bytes, dpi=72)
    assert len(pages) == 1
    assert pages[0].page_index == 0
    assert pages[0].png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_normalize_image_to_png_smoke(tmp_path) -> None:
    pil_image = pytest.importorskip("PIL.Image")
    img = pil_image.new("RGB", (10, 10), color=(255, 0, 0))
    p = tmp_path / "x.jpg"
    img.save(p)
    png = normalize_image_to_png(p.read_bytes())
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
