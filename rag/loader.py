"""
Loads PDFs and splits them into chunks that know where they came from:
document, page and section. That metadata is what the citations show.
"""
import re
from collections import Counter
from io import BytesIO
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader

# Numbered headings such as "4. Passwords and Authentication" or "3.2 Access reviews"
HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][^\n]{2,80})$", re.MULTILINE)
PAGE_NUMBER = re.compile(r"^(page\s+)?\d+(\s*(/|of)\s*\d+)?$", re.IGNORECASE)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def read_pages(pdf, name: str) -> tuple[str, list[str]]:
    """
    Returns the document title and the text of each page, without
    page numbers and without headers or footers repeated on every page.
    `pdf` is a path or a file-like object (e.g. a Streamlit upload).
    """
    data = pdf.read() if hasattr(pdf, "read") else Path(pdf).read_bytes()
    reader = PdfReader(BytesIO(data))
    pages = [(page.extract_text() or "").splitlines() for page in reader.pages]

    repeated = set()
    if len(pages) > 1:
        counts = Counter(line.strip() for lines in pages for line in set(lines) if line.strip())
        repeated = {line for line, n in counts.items() if n == len(pages)}

    texts = ["\n".join(line for line in lines
                       if line.strip() and line.strip() not in repeated and not PAGE_NUMBER.match(line.strip()))
             for lines in pages]

    title = (reader.metadata.title if reader.metadata and reader.metadata.title else "") or Path(name).stem
    return title, texts


def chunk_document(title: str, pages: list[str], source: str, by_section: bool = True) -> list[Document]:
    """
    Splits a document into chunks with source, title, page and section metadata.

    by_section=True keeps every chunk inside one section and starts it with the
    section heading, so a chunk about "within 14 days" still says it is about
    patching. by_section=False is plain fixed-size chunking, kept to compare
    both in the evaluation.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", ". ", " "],
    )
    chunks = []
    section = ""  # carried over to the next page until a new heading appears

    for page_number, text in enumerate(pages, start=1):
        if not text.strip():
            continue
        if not by_section:
            for part in splitter.split_text(text):
                start = max(text.find(part[:50]), 0)
                end = start + len(part)
                before = [m for m in HEADING.finditer(text) if m.start() <= start]
                chunk_section = f"{before[-1].group(1)}. {before[-1].group(2)}" if before else section
                # A fixed-size chunk can span several sections; record all of them
                inside = [f"{m.group(1)}. {m.group(2)}" for m in HEADING.finditer(text) if start < m.start() < end]
                doc = _doc(part, source, title, page_number, chunk_section)
                doc.metadata["sections_covered"] = " | ".join([chunk_section] + inside)
                chunks.append(doc)
            last = list(HEADING.finditer(text))
            section = f"{last[-1].group(1)}. {last[-1].group(2)}" if last else section
            continue

        # Text before the first heading on this page belongs to the previous section
        bounds = [(m.start(), f"{m.group(1)}. {m.group(2)}") for m in HEADING.finditer(text)]
        blocks = [(0, section)] + bounds
        for i, (start, name) in enumerate(blocks):
            end = blocks[i + 1][0] if i + 1 < len(blocks) else len(text)
            body = text[start:end].strip()
            if name and body.startswith(name):
                body = body[len(name):].strip()
            if not body:
                continue
            for part in splitter.split_text(body):
                content = f"{title} - {name}\n{part}" if name else f"{title}\n{part}"
                chunks.append(_doc(content, source, title, page_number, name))
        section = bounds[-1][1] if bounds else section

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{source}#{i}"
    return chunks


def _doc(content, source, title, page, section):
    return Document(page_content=content, metadata={"source": source, "title": title, "page": page, "section": section})


def load_and_chunk(pdf, name: str | None = None, by_section: bool = True) -> list[Document]:
    """Reads one PDF (path or upload) and returns its chunks."""
    name = name or getattr(pdf, "name", None) or str(pdf)
    title, pages = read_pages(pdf, name)
    return chunk_document(title, pages, Path(name).name, by_section=by_section)
