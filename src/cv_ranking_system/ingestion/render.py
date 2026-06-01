from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

try:
    import fitz  # PyMuPDF
except ModuleNotFoundError:  # pragma: no cover
    fitz = None


@dataclass(frozen=True)
class RenderedPage:
    page_index: int
    png_bytes: bytes


def _image_to_png_bytes(img: Any) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_pdf_to_png_pages(pdf_bytes: bytes, *, dpi: int = 200) -> list[RenderedPage]:
    if fitz is None:
        raise ModuleNotFoundError(
            "Missing optional dependency 'pymupdf' (import name 'fitz'). "
            "Install it to enable PDF rendering."
        )
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[RenderedPage] = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pages.append(RenderedPage(page_index=i, png_bytes=pix.tobytes("png")))
    return pages


def normalize_image_to_png(img_bytes: bytes) -> bytes:
    from PIL import Image

    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert("RGB")
    return _image_to_png_bytes(img)
