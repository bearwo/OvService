from __future__ import annotations

from pathlib import Path


def parse_file(file_path: str | Path) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return parse_pdf(path)
    elif suffix in (".txt", ".md", ".markdown"):
        return path.read_text(encoding="utf-8")
    elif suffix == ".json":
        import json
        return json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)
    else:
        return path.read_text(encoding="utf-8", errors="replace")


def parse_pdf(file_path: str | Path) -> str:
    import fitz
    doc = fitz.open(str(file_path))
    parts = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        if text.strip():
            parts.append(f"--- Page {page_num + 1} ---\n{text}")
    doc.close()
    return "\n\n".join(parts)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    if overlap >= chunk_size:
        overlap = chunk_size // 4

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
