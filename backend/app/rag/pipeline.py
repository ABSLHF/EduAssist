import re
from typing import List, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingModelUnavailable(RuntimeError):
    pass


_model_cache: dict[str, SentenceTransformer] = {}
_model_error = None
client = chromadb.PersistentClient(
    path=settings.chroma_path,
    settings=Settings(anonymized_telemetry=False),
)


EXERCISE_PATTERN = re.compile(
    r"(\u7b2c\s*\d+\s*\u7ae0)?(\u7ec3\u4e60|\u4e60\u9898|\u9009\u62e9\u9898|\u7b54\u6848)",
    re.IGNORECASE,
)
OPTION_LINE_PATTERN = re.compile(r"^\s*[A-D][\.\u3001\)]")
QUESTION_PATTERN = re.compile(r"\d+\.|[?\uFF1F]")
TOKEN_SPLIT_PATTERN = re.compile(r"[^\w\u4e00-\u9fff]+")


def _load_model(model_name: str) -> SentenceTransformer:
    if model_name in _model_cache:
        return _model_cache[model_name]
    model = SentenceTransformer(model_name)
    _model_cache[model_name] = model
    return model


def get_model() -> SentenceTransformer:
    global _model_error
    if _model_error is not None:
        raise EmbeddingModelUnavailable(_model_error)
    model_names = [settings.embedding_model_name, settings.embedding_fallback_model]
    last_error = ""
    for model_name in model_names:
        try:
            return _load_model(model_name)
        except Exception as exc:
            last_error = f"{model_name}: {exc}"
    _model_error = last_error or "No embedding model available"
    raise EmbeddingModelUnavailable(_model_error)


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + size)
        chunks.append(text[start:end])
        if end >= length:
            break
        start = max(0, end - overlap)
    return chunks


def is_noisy_chunk(text: str) -> bool:
    t = text.strip()
    if not t:
        return True

    if EXERCISE_PATTERN.search(t):
        return True

    lines = [line.strip() for line in t.splitlines() if line.strip()]
    option_lines = sum(1 for line in lines if OPTION_LINE_PATTERN.match(line))
    if option_lines >= 2:
        return True

    if QUESTION_PATTERN.search(t) and len(t) < 260 and option_lines >= 1:
        return True

    # Too many one-character lines usually indicates broken OCR/PPT noise.
    short_lines = sum(1 for line in lines if len(line) <= 2)
    if lines and short_lines / max(len(lines), 1) > 0.45:
        return True

    return False


def upsert_material(course_id: int, material_id: int, text: str) -> int:
    if not text or not text.strip():
        return 0
    collection = client.get_or_create_collection(f"course_{course_id}")
    chunks = chunk_text(text, settings.doc_chunk_size, settings.doc_chunk_overlap)
    if not chunks:
        return 0

    model = get_model()
    embeddings = model.encode(chunks).tolist()
    ids = [f"{material_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"material_id": material_id, "chunk_id": i} for i in range(len(chunks))]
    collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=chunks)
    return len(chunks)


def _query_terms(question: str) -> list[str]:
    q = question.strip()
    for token in ("\u4ec0\u4e48\u662f", "\u89e3\u91ca\u4e00\u4e0b", "\u8bf4\u660e", "\u4ecb\u7ecd", "\uFF1F", "?"):
        q = q.replace(token, " ")
    terms = [t.strip() for t in TOKEN_SPLIT_PATTERN.split(q) if len(t.strip()) >= 2]

    # Extra split for conjunctions in Chinese titles, e.g. "A与B".
    expanded: list[str] = []
    for term in terms:
        expanded.append(term)
        for sep in ("\u4e0e", "\u548c", "\u53ca", "\u3001"):
            if sep in term:
                expanded.extend([p.strip() for p in term.split(sep) if len(p.strip()) >= 2])

    deduped: list[str] = []
    seen = set()
    for term in expanded:
        if term not in seen:
            deduped.append(term)
            seen.add(term)
    return deduped


def _lexical_score(question: str, text: str) -> int:
    terms = _query_terms(question)
    if not terms:
        return 0
    return sum(1 for term in terms if term in text)


def retrieve(course_id: int, question: str, top_k: int = 3) -> List[Tuple[str, dict]]:
    collection = client.get_or_create_collection(f"course_{course_id}")
    query_n = max(top_k * 3, 8)
    result = None
    query_errors: list[str] = []

    for model_name in [settings.embedding_model_name, settings.embedding_fallback_model]:
        try:
            model = _load_model(model_name)
            query_emb = model.encode([question]).tolist()[0]
            result = collection.query(query_embeddings=[query_emb], n_results=query_n)
            break
        except Exception as exc:
            query_errors.append(f"{model_name}: {exc}")

    if result is None:
        raise EmbeddingModelUnavailable("; ".join(query_errors) or "embedding query failed")

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]

    filtered = []
    for doc, meta in zip(docs, metas):
        if not is_noisy_chunk(doc):
            filtered.append((doc, meta))

    ranked = sorted(filtered, key=lambda item: _lexical_score(question, item[0]), reverse=True)
    return ranked[:top_k]
