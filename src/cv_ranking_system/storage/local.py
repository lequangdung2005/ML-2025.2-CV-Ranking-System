from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalArtifactPaths:
    root_dir: Path
    raw_path: Path
    markdown_path: Path


def build_paths(*, root_dir: Path, doc_id: str, source_name: str) -> LocalArtifactPaths:
    raw_path = root_dir / "raw" / doc_id / source_name
    markdown_path = root_dir / "artifacts" / doc_id / "ocr.md"
    return LocalArtifactPaths(root_dir=root_dir, raw_path=raw_path, markdown_path=markdown_path)


def put_bytes(*, path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
