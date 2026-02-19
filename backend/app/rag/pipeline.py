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
SINGLE_CHAR_TERMS = {"\u6811", "\u6808", "\u56fe", "\u5806"}
QUESTION_NOISE_TOKENS = [
    "\u4ec0\u4e48\u662f",
    "\u89e3\u91ca\u4e00\u4e0b",
    "\u89e3\u91ca",
    "\u8bf4\u660e",
    "\u4ecb\u7ecd",
    "\u4e3a\u4ec0\u4e48",
    "\u600e\u4e48",
    "\u5982\u4f55",
    "\u662f\u4ec0\u4e48",
    "\u4f1a\u53d1\u751f",
    "\u57fa\u672c\u601d\u60f3",
    "\u6838\u5fc3\u7279\u6027",
    "\u7279\u70b9",
    "\u5b9a\u4e49",
    "\u533a\u522b",
    "\u4e0d\u540c",
    "\uFF1F",
    "?",
]
GENERIC_TERMS = {
    "\u7279\u70b9",
    "\u5b9a\u4e49",
    "\u533a\u522b",
    "\u4e0d\u540c",
    "\u539f\u56e0",
    "\u4f5c\u7528",
    "\u6548\u679c",
    "\u57fa\u672c\u601d\u60f3",
    "\u6838\u5fc3\u7279\u6027",
    "\u4f1a\u53d1\u751f",
}
TERM_SYNONYMS = {
    "\u987a\u5e8f\u8868": ["\u6570\u7ec4", "\u987a\u5e8f\u5b58\u50a8"],
    "\u94fe\u8868": ["\u6307\u9488", "\u7ed3\u70b9"],
    "\u7ebf\u6027\u8868": ["\u987a\u5e8f\u8868", "\u94fe\u8868"],
    "\u6811": ["\u4e8c\u53c9\u6811", "\u5c42\u6b21\u7ed3\u6784"],
    "\u54c8\u5e0c\u8868": ["\u6563\u5217\u8868", "\u54c8\u5e0c", "hash"],
    "\u7a33\u5b9a\u6392\u5e8f": ["\u76f8\u5bf9\u987a\u5e8f", "\u5173\u952e\u5b57"],
    "\u7d22\u5f15": ["b+\u6811", "\u67e5\u8be2", "\u6570\u636e\u5e93\u7d22\u5f15"],
    "\u9012\u5f52": ["\u8c03\u7528\u6808", "\u6808\u6ea2\u51fa"],
    "\u5feb\u901f\u6392\u5e8f": ["\u5feb\u6392", "\u57fa\u51c6", "\u9000\u5316"],
    "\u7ea2\u9ed1\u6811": ["\u5e73\u8861", "\u65cb\u8f6c", "\u67d3\u8272"],
    "acid": ["\u4e8b\u52a1", "\u539f\u5b50\u6027", "\u4e00\u81f4\u6027", "\u9694\u79bb\u6027", "\u6301\u4e45\u6027"],
    "\u6b7b\u9501": ["\u4e92\u65a5", "\u5faa\u73af\u7b49\u5f85", "\u5e76\u53d1"],
}


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
    def _keep_term(term: str) -> bool:
        return len(term) >= 2 or term in SINGLE_CHAR_TERMS

    q = question.strip()
    for token in QUESTION_NOISE_TOKENS:
        q = q.replace(token, " ")

    terms: list[str] = []
    for raw in TOKEN_SPLIT_PATTERN.split(q):
        term = raw.strip().lower()
        if not term:
            continue
        if "\u7684" in term:
            terms.extend([p.strip() for p in term.split("\u7684") if p.strip()])
        else:
            terms.append(term)
    terms = [t for t in terms if _keep_term(t) and t not in GENERIC_TERMS]

    # Extra split for conjunctions in Chinese titles, e.g. "A与B".
    expanded: list[str] = []
    for term in terms:
        expanded.append(term)
        for sep in ("\u4e0e", "\u548c", "\u53ca", "\u3001"):
            if sep in term:
                expanded.extend([p.strip().lower() for p in term.split(sep) if _keep_term(p.strip())])

    deduped: list[str] = []
    seen = set()
    for term in expanded:
        if term not in seen:
            deduped.append(term)
            seen.add(term)
    return deduped


def _expand_terms(terms: list[str]) -> list[str]:
    extras: list[str] = []
    for term in terms:
        extras.extend(TERM_SYNONYMS.get(term, []))
    out: list[str] = []
    seen = set()
    for term in extras:
        t = term.strip().lower()
        if not t or t in seen:
            continue
        out.append(t)
        seen.add(t)
    return out


def _lexical_score(question: str, text: str) -> int:
    text_norm = text.lower()
    terms = _query_terms(question)
    if not terms:
        return 0
    related_terms = _expand_terms(terms)
    exact_hits = sum(1 for term in terms if term and term in text_norm)
    related_hits = sum(1 for term in related_terms if term and term in text_norm)
    score = exact_hits * 3 + related_hits
    if ("\u533a\u522b" in question or "\u4e0d\u540c" in question) and ("\u533a\u522b" in text or "\u4e0d\u540c" in text):
        score += 1
    if "\u4e3a\u4ec0\u4e48" in question and ("\u539f\u56e0" in text or "\u5bfc\u81f4" in text):
        score += 1
    return score


def retrieve(course_id: int, question: str, top_k: int = 3) -> List[Tuple[str, dict]]:
    collection = client.get_or_create_collection(f"course_{course_id}")
    query_n = max(top_k * 10, 30)
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
    if ranked and _lexical_score(question, ranked[0][0]) > 0:
        return ranked[:top_k]

    # Fallback: when semantic retrieval misses exact terms, scan current course chunks lexically.
    try:
        all_rows = collection.get(include=["documents", "metadatas"])
        all_docs = all_rows.get("documents", [])
        all_metas = all_rows.get("metadatas", [])
        lexical_hits = []
        for doc, meta in zip(all_docs, all_metas):
            if not doc or is_noisy_chunk(doc):
                continue
            score = _lexical_score(question, doc)
            if score > 0:
                lexical_hits.append((score, doc, meta))
        if lexical_hits:
            lexical_hits.sort(key=lambda item: item[0], reverse=True)
            return [(doc, meta) for _score, doc, meta in lexical_hits[:top_k]]
    except Exception:
        pass

    return ranked[:top_k]
