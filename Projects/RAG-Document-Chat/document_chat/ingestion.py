"""PDF validation, text extraction, and page-aware chunking."""

import hashlib
import io
import re
from typing import Iterable

from config import CHUNK_CHARS, CHUNK_OVERLAP, MAX_FILE_BYTES, MAX_FILES, MAX_PAGES_PER_FILE
from document_chat.models import Chunk


def chunk_page(text: str, source: str, page: int, digest: str) -> list[Chunk]:
    """Split at word boundaries while retaining short overlaps and page identity."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    chunks: list[Chunk] = []
    start = 0
    while start < len(normalized):
        end = min(start + CHUNK_CHARS, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + CHUNK_CHARS // 2, end)
            if boundary > start:
                end = boundary
        part = normalized[start:end].strip()
        if part:
            chunks.append(Chunk(f"{digest}:{page}:{start}", source, page, part))
        if end >= len(normalized):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)
    return chunks


def extract_pdfs(files: Iterable[tuple[str, bytes]]) -> tuple[list[Chunk], list[str]]:
    """Read uploaded bytes only; never persist the original PDFs on disk."""
    from pypdf import PdfReader

    items = list(files)
    if not items or len(items) > MAX_FILES:
        raise ValueError(f"Upload between 1 and {MAX_FILES} PDF files.")
    chunks: list[Chunk] = []
    warnings: list[str] = []
    for name, data in items:
        source = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if not source.lower().endswith(".pdf") or not data.startswith(b"%PDF-"):
            raise ValueError(f"{source}: expected a PDF file.")
        if len(data) > MAX_FILE_BYTES:
            raise ValueError(f"{source}: file exceeds the 10 MB limit.")
        try:
            reader = PdfReader(io.BytesIO(data), strict=False)
            if reader.is_encrypted:
                raise ValueError(f"{source}: encrypted PDFs are not supported.")
            if len(reader.pages) > MAX_PAGES_PER_FILE:
                raise ValueError(f"{source}: file exceeds {MAX_PAGES_PER_FILE} pages.")
            digest = hashlib.sha256(data).hexdigest()[:16]
            before = len(chunks)
            for page_no, page in enumerate(reader.pages, 1):
                chunks.extend(chunk_page(page.extract_text() or "", source, page_no, digest))
            if len(chunks) == before:
                warnings.append(f"{source}: no selectable text found; scanned PDFs need OCR.")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"{source}: could not read PDF ({type(exc).__name__}).") from exc
    if not chunks:
        raise ValueError("No selectable text was found in the uploaded PDFs.")
    return chunks, warnings
