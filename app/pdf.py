from io import BytesIO

from pypdf import PdfReader


def extract_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)
