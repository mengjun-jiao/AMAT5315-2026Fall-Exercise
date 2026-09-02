#!/usr/bin/env python3
"""Download or open a PDF and extract its text with pypdf."""

import argparse
from pathlib import Path
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pypdf import PdfReader


def is_web_address(source: str) -> bool:
    return urlparse(source).scheme in {"http", "https"}


def download_pdf(source: str) -> Path:
    request = Request(source, headers={"User-Agent": "Codex tutor skill"})
    with urlopen(request) as response:
        data = response.read()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary:
        temporary.write(data)
        return Path(temporary.name)


def extract_text(source: str) -> str:
    downloaded_path = download_pdf(source) if is_web_address(source) else None
    pdf_path = downloaded_path or Path(source)

    try:
        reader = PdfReader(pdf_path)
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(f"--- Page {page_number} ---\n{text}")
        return "\n\n".join(pages)
    finally:
        if downloaded_path is not None:
            downloaded_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text from a local PDF or an HTTP(S) PDF."
    )
    parser.add_argument("source", help="Local PDF path or HTTP(S) address")
    parser.add_argument("output", type=Path, help="Destination text file")
    args = parser.parse_args()

    args.output.write_text(extract_text(args.source), encoding="utf-8")


if __name__ == "__main__":
    main()
