from pathlib import Path
from fastapi import UploadFile

# Optional parsers
try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None

try:
    from docx import Document  # type: ignore
except Exception:
    Document = None

try:
    from pptx import Presentation  # type: ignore
except Exception:
    Presentation = None


def save_upload(course_id: int, upload: UploadFile, base_dir: str = "storage") -> str:
    course_dir = Path(base_dir) / f"course_{course_id}"
    course_dir.mkdir(parents=True, exist_ok=True)
    dest = course_dir / upload.filename
    with dest.open("wb") as f:
        f.write(upload.file.read())
    return str(dest)


def extract_text(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix in [".txt", ".md"]:
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf" and pdfplumber is not None:
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts)
    if suffix in [".docx", ".doc"] and Document is not None:
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text])
    if suffix in [".pptx", ".ppt"] and Presentation is not None:
        prs = Presentation(file_path)
        slides_text = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    slides_text.append(shape.text)
        return "\n".join(slides_text)
    # fallback: try plain text
    return Path(file_path).read_text(encoding="utf-8", errors="ignore")
