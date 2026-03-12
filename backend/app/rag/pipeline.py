import hashlib
import re
from typing import Any, List, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingModelUnavailable(RuntimeError):
    pass


_model_cache: dict[str, SentenceTransformer] = {}
_model_dim_cache: dict[str, int] = {}
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
    "\u8ffd\u95ee",
    "\u8fdb\u4e00\u6b65",
    "\u8fdb\u4e00\u6b65\u4ecb\u7ecd",
    "\u7ee7\u7eed",
    "\u8865\u5145",
    "\u8be6\u7ec6",
    "\uFF1F",
    "?",
]
QUESTION_TRAILING_SYMBOLS = " \t\r\n'\"`‘’“”＇＂，,。！？?；;：:"
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

SEMANTIC_WEIGHT = 0.65
LEXICAL_WEIGHT = 0.35
RETRIEVE_MIN_SCORE = 0.18
RETRIEVE_MIN_LEXICAL_HITS = 1
MAX_CHUNKS_PER_MATERIAL = 2


def _load_model(model_name: str) -> SentenceTransformer:
    if model_name in _model_cache:
        return _model_cache[model_name]
    model = SentenceTransformer(model_name)
    _model_cache[model_name] = model
    return model


def _embedding_dimension(model_name: str, model: SentenceTransformer) -> int:
    if model_name in _model_dim_cache:
        return _model_dim_cache[model_name]
    dim = model.get_sentence_embedding_dimension()
    if dim is None:
        dim = len(model.encode(["dimension_probe"]).tolist()[0])
    _model_dim_cache[model_name] = int(dim)
    return int(dim)


def _model_slug(model_name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    normalized = normalized[:24] if normalized else "model"
    digest = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:8]
    return f"{normalized}_{digest}"


def _model_collection_name(course_id: int, model_name: str, model: SentenceTransformer) -> str:
    dim = _embedding_dimension(model_name, model)
    return f"course_{course_id}_d{dim}_{_model_slug(model_name)}"


def _legacy_collection_name(course_id: int) -> str:
    return f"course_{course_id}"


def _collection_candidates(course_id: int, model_name: str, model: SentenceTransformer) -> list[str]:
    model_collection = _model_collection_name(course_id, model_name, model)
    legacy_collection = _legacy_collection_name(course_id)
    if model_collection == legacy_collection:
        return [model_collection]
    return [model_collection, legacy_collection]


def _get_collection_if_exists(name: str):
    try:
        return client.get_collection(name)
    except Exception:
        return None


def get_model() -> tuple[str, SentenceTransformer]:
    global _model_error
    if _model_error is not None:
        raise EmbeddingModelUnavailable(_model_error)
    model_names = [settings.embedding_model_name, settings.embedding_fallback_model]
    last_error = ""
    for model_name in model_names:
        try:
            return model_name, _load_model(model_name)
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


def _normalize_query_text(question: str) -> str:
    q = (question or "").strip()
    q = q.rstrip(QUESTION_TRAILING_SYMBOLS)
    q = re.sub(r"\s+", " ", q).strip()
    return q


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
    chunks = chunk_text(text, settings.doc_chunk_size, settings.doc_chunk_overlap)
    if not chunks:
        return 0

    model_name, model = get_model()
    collection = client.get_or_create_collection(_model_collection_name(course_id, model_name, model))
    embeddings = model.encode(chunks).tolist()
    ids = [f"{material_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"material_id": material_id, "chunk_id": i, "course_id": course_id} for i in range(len(chunks))]
    collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=chunks)
    return len(chunks)


def _query_terms(question: str) -> list[str]:
    def _keep_term(term: str) -> bool:
        return len(term) >= 2 or term in SINGLE_CHAR_TERMS

    q = _normalize_query_text(question)
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


def _lexical_match_stats(question: str, text: str) -> tuple[int, float]:
    terms = _query_terms(question)
    if not terms:
        return 0, 0.0

    text_norm = text.lower()
    related_terms = _expand_terms(terms)
    exact_hits = sum(1 for term in terms if term and term in text_norm)
    related_hits = sum(1 for term in related_terms if term and term in text_norm)

    weighted_hits = exact_hits * 2 + related_hits
    lexical_hits = exact_hits + related_hits
    lexical_norm = min(1.0, weighted_hits / 4.0) if weighted_hits > 0 else 0.0

    if ("\u533a\u522b" in question or "\u4e0d\u540c" in question) and ("\u533a\u522b" in text or "\u4e0d\u540c" in text):
        lexical_hits = max(lexical_hits, 1)
        lexical_norm = min(1.0, lexical_norm + 0.15)
    if "\u4e3a\u4ec0\u4e48" in question and ("\u539f\u56e0" in text or "\u5bfc\u81f4" in text):
        lexical_hits = max(lexical_hits, 1)
        lexical_norm = min(1.0, lexical_norm + 0.1)

    return lexical_hits, lexical_norm


def _semantic_score(distance: float | None, rank: int, total: int) -> float:
    if distance is not None:
        return 1.0 / (1.0 + max(distance, 0.0))
    denominator = max(total - 1, 1)
    return max(0.0, 1.0 - (rank / denominator))


def _candidate_key(doc: str, meta: dict) -> tuple[Any, Any, str]:
    return meta.get("material_id"), meta.get("chunk_id"), doc


def _material_id(meta: dict) -> int | None:
    material_id = meta.get("material_id")
    if material_id is None:
        return None
    try:
        return int(material_id)
    except Exception:
        return None


def _prefer_candidate(new_row: dict[str, Any], old_row: dict[str, Any]) -> bool:
    new_distance = new_row.get("distance")
    old_distance = old_row.get("distance")
    if new_distance is not None and old_distance is not None and new_distance != old_distance:
        return new_distance < old_distance
    if new_distance is not None and old_distance is None:
        return True
    if new_distance is None and old_distance is not None:
        return False
    return new_row.get("rank", 0) < old_row.get("rank", 0)


def _apply_material_diversity(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    per_material_count: dict[int | None, int] = {}

    for row in rows:
        material_id = _material_id(row.get("meta", {}))
        used = per_material_count.get(material_id, 0)
        if used >= MAX_CHUNKS_PER_MATERIAL:
            continue
        selected.append(row)
        per_material_count[material_id] = used + 1
        if len(selected) >= top_k:
            break
    return selected


def retrieve(course_id: int, question: str, top_k: int = 3) -> List[Tuple[str, dict]]:
    normalized_question = _normalize_query_text(question)
    query_n = max(top_k * 12, 40)
    query_errors: list[str] = []
    semantic_candidates: list[dict[str, Any]] = []
    lexical_collections: dict[str, Any] = {}
    model_names = [settings.embedding_model_name, settings.embedding_fallback_model]

    for model_name in model_names:
        try:
            model = _load_model(model_name)
            query_emb = model.encode([normalized_question]).tolist()[0]
        except Exception as exc:
            query_errors.append(f"{model_name}: {exc}")
            continue

        model_collection_count = 0
        for collection_name in _collection_candidates(course_id, model_name, model):
            collection = _get_collection_if_exists(collection_name)
            if collection is None:
                continue
            model_collection_count += 1
            lexical_collections[collection_name] = collection

            try:
                result = collection.query(query_embeddings=[query_emb], n_results=query_n)
            except Exception as exc:
                query_errors.append(f"{collection_name} ({model_name}): {exc}")
                continue

            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            total = max(len(docs), 1)
            for rank, (doc, meta) in enumerate(zip(docs, metas)):
                if not doc or is_noisy_chunk(doc):
                    continue
                distance = None
                if isinstance(distances, list) and rank < len(distances):
                    try:
                        distance = float(distances[rank])
                    except Exception:
                        distance = None

                candidate_meta = dict(meta or {})
                candidate_meta.setdefault("course_id", course_id)
                semantic_candidates.append(
                    {
                        "doc": doc,
                        "meta": candidate_meta,
                        "rank": rank,
                        "total": total,
                        "distance": distance,
                    }
                )

        # Primary model already has usable collections for this course:
        # keep latency low and avoid unnecessary fallback-model loading.
        if model_collection_count > 0:
            break

    if not semantic_candidates and query_errors and not lexical_collections:
        raise EmbeddingModelUnavailable("; ".join(query_errors) or "embedding query failed")

    deduped_semantic: dict[tuple[Any, Any, str], dict[str, Any]] = {}
    for row in semantic_candidates:
        key = _candidate_key(row["doc"], row["meta"])
        old = deduped_semantic.get(key)
        if old is None or _prefer_candidate(row, old):
            deduped_semantic[key] = row

    scored_semantic: list[dict[str, Any]] = []
    for row in deduped_semantic.values():
        lexical_hits, lexical_norm = _lexical_match_stats(normalized_question, row["doc"])
        semantic_norm = _semantic_score(row.get("distance"), row.get("rank", 0), row.get("total", 1))
        blended = SEMANTIC_WEIGHT * semantic_norm + LEXICAL_WEIGHT * lexical_norm
        if lexical_hits < RETRIEVE_MIN_LEXICAL_HITS or blended < RETRIEVE_MIN_SCORE:
            continue
        scored_semantic.append(
            {
                **row,
                "lexical_hits": lexical_hits,
                "lexical_norm": lexical_norm,
                "semantic_norm": semantic_norm,
                "blended": blended,
            }
        )

    scored_semantic.sort(
        key=lambda item: (item["blended"], item["lexical_hits"], item["semantic_norm"]),
        reverse=True,
    )
    diverse_semantic = _apply_material_diversity(scored_semantic, top_k)
    if diverse_semantic:
        return [(row["doc"], row["meta"]) for row in diverse_semantic]

    # Fallback: semantic filtering failed, do lexical-only scan in current course collections.
    lexical_candidates: list[dict[str, Any]] = []
    for collection in lexical_collections.values():
        try:
            all_rows = collection.get(include=["documents", "metadatas"])
            all_docs = all_rows.get("documents", [])
            all_metas = all_rows.get("metadatas", [])
            for doc, meta in zip(all_docs, all_metas):
                if not doc or is_noisy_chunk(doc):
                    continue
                lexical_hits, lexical_norm = _lexical_match_stats(normalized_question, doc)
                if lexical_hits < RETRIEVE_MIN_LEXICAL_HITS:
                    continue
                candidate_meta = dict(meta or {})
                candidate_meta.setdefault("course_id", course_id)
                lexical_candidates.append(
                    {
                        "doc": doc,
                        "meta": candidate_meta,
                        "lexical_hits": lexical_hits,
                        "lexical_norm": lexical_norm,
                        "semantic_norm": 0.0,
                        "blended": lexical_norm,
                    }
                )
        except Exception:
            continue

    deduped_lexical: dict[tuple[Any, Any, str], dict[str, Any]] = {}
    for row in lexical_candidates:
        key = _candidate_key(row["doc"], row["meta"])
        old = deduped_lexical.get(key)
        if old is None:
            deduped_lexical[key] = row
            continue
        if row["blended"] > old["blended"]:
            deduped_lexical[key] = row
            continue
        if row["blended"] == old["blended"] and row["lexical_hits"] > old["lexical_hits"]:
            deduped_lexical[key] = row

    ranked_lexical = sorted(
        deduped_lexical.values(),
        key=lambda item: (item["blended"], item["lexical_hits"]),
        reverse=True,
    )
    diverse_lexical = _apply_material_diversity(ranked_lexical, top_k)
    return [(row["doc"], row["meta"]) for row in diverse_lexical]
