from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import itertools
import logging
from pathlib import Path
import re
import unicodedata

import httpx
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import settings

try:
    import jieba
    import jieba.analyse as jieba_analyse
    import jieba.posseg as jieba_posseg
except Exception:  # pragma: no cover - 可选依赖回退
    jieba = None
    jieba_analyse = None
    jieba_posseg = None

try:
    from paddlenlp import Taskflow
except Exception:  # pragma: no cover - 可选依赖回退
    Taskflow = None

try:
    import hanlp
except Exception:  # pragma: no cover - 可选依赖回退
    hanlp = None

logger = logging.getLogger(__name__)

_NOISE_WORDS = {
    "appdata",
    "users",
    "user",
    "typora",
    "image",
    "images",
    "img",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "bmp",
    "webp",
    "tmp",
    "temp",
    "cache",
    "desktop",
    "downloads",
    "documents",
    "windows",
    "program",
    "files",
    "roaming",
}

_GENERIC_STRUCTURAL_WORDS = {
    "位于",
    "用于",
    "包括",
    "实现",
    "进行",
    "通过",
    "相关",
    "其中",
    "以及",
    "并且",
    "可以",
    "通常",
    "主要",
    "一个",
    "一种",
    "部分",
    "负责",
}

_GENERIC_WEAK_TERMS = {
    "网络",
    "协议",
    "系统",
    "模型",
    "方法",
    "技术",
    "过程",
    "结构",
    "功能",
    "模块",
}

_CLAUSE_CUE_PARTS = (
    "位于",
    "负责",
    "用于",
    "通过",
    "实现",
    "包括",
    "属于",
    "依赖",
    "组成",
    "包含",
)

_DEFAULT_STOPWORDS = {
    "一个",
    "一种",
    "一些",
    "以及",
    "进行",
    "可以",
    "如果",
    "就是",
    "这个",
    "那个",
    "我们",
    "你们",
    "他们",
    "问题",
    "内容",
    "课程",
    "学习",
    "知识",
    "信息",
    "说明",
    "部分",
    "相关",
    "包括",
    "通过",
    "使用",
    "实现",
    "方法",
    "原理",
    "系统",
    "过程",
    "功能",
}

_DEFAULT_DOMAIN_LEXICON = {
    "数据结构",
    "线性表",
    "链表",
    "栈",
    "队列",
    "树",
    "二叉树",
    "图",
    "排序",
    "查找",
    "数据库",
    "关系模型",
    "事务",
    "索引",
    "范式",
    "sql",
    "操作系统",
    "进程",
    "线程",
    "调度",
    "死锁",
    "内存管理",
    "文件系统",
    "计算机网络",
    "应用层",
    "运输层",
    "网络层",
    "链路层",
    "物理层",
    "tcp",
    "udp",
    "http",
    "ip",
    "路由",
}

_SENTENCE_SPLIT_RE = re.compile(r"[。！？；;，,、\n\r]+")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_PATH_RE = re.compile(r"(?:[a-zA-Z]:\\|/|\\).+")
_EXT_RE = re.compile(r"\.(?:png|jpe?g|gif|bmp|webp|svg|ico|pdf|docx?|pptx?|txt|md)$", re.IGNORECASE)
_TOKEN_CLEAN_RE = re.compile(r"^[\W_]+|[\W_]+$")
_PHRASE_NOISE_PARTS = ("属于", "用于", "可以", "包括", "进行", "以及", "实现", "有助于")
_SYNONYM_MAP = {
    "传输层": "运输层",
    "transportlayer": "运输层",
    "networklayer": "网络层",
    "applicationlayer": "应用层",
    "datalinklayer": "链路层",
}


@dataclass
class KeywordExtractResult:
    candidates: list[str]
    filtered_noise: int = 0
    extractor: str = "hybrid"
    fallback_used: bool = False


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _resolve_data_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return path

    backend_root = Path(__file__).resolve().parents[2]
    for candidate in (path, backend_root / path_value):
        if candidate.exists():
            return candidate
    return None


def _normalize_term(term: str) -> str:
    normalized = unicodedata.normalize("NFKC", term or "")
    normalized = _TOKEN_CLEAN_RE.sub("", normalized).strip().lower()
    normalized = re.sub(r"\s+", "", normalized)
    # 先做精确映射，再做子串映射，统一同义表达。
    if normalized in _SYNONYM_MAP:
        normalized = _SYNONYM_MAP[normalized]
    for alias, canonical in sorted(_SYNONYM_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        if alias and alias in normalized:
            normalized = normalized.replace(alias, canonical)
    return normalized


@lru_cache(maxsize=1)
def _load_stopwords() -> set[str]:
    words = set(_DEFAULT_STOPWORDS)
    file_path = _resolve_data_path(settings.kg_stopwords_path)
    if not file_path:
        return words

    try:
        for line in file_path.read_text(encoding="utf-8").splitlines():
            token = line.strip()
            if token and not token.startswith("#"):
                words.add(_normalize_term(token))
    except Exception as exc:  # pragma: no cover - 防御性处理
        logger.warning("Failed to load kg stopwords from %s: %s", file_path, exc)
    return words


@lru_cache(maxsize=1)
def _load_domain_lexicon() -> set[str]:
    terms = {_normalize_term(item) for item in _DEFAULT_DOMAIN_LEXICON}
    file_path = _resolve_data_path(settings.kg_domain_lexicon_path)
    if not file_path:
        return terms

    try:
        for line in file_path.read_text(encoding="utf-8").splitlines():
            token = line.strip()
            if token and not token.startswith("#"):
                terms.add(_normalize_term(token))
    except Exception as exc:  # pragma: no cover - 防御性处理
        logger.warning("Failed to load kg domain lexicon from %s: %s", file_path, exc)
    return terms


def _clean_text(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _looks_like_noise(raw_token: str, min_len: int, stopwords: set[str]) -> bool:
    token = _normalize_term(raw_token)
    domain_lexicon = _load_domain_lexicon()
    if not token:
        return True
    if len(token) < max(1, min_len) or len(token) > 10:
        return True
    if not settings.kg_enable_noise_filter:
        return token.isdigit()
    if token in stopwords:
        return True
    if token in _NOISE_WORDS:
        return True
    if token in _GENERIC_STRUCTURAL_WORDS:
        return True
    if _EXT_RE.search(token):
        return True
    if _PATH_RE.match(raw_token):
        return True
    if token.isdigit():
        return True
    if len(token) <= 2 and token not in domain_lexicon:
        return True
    if re.fullmatch(r"[a-z]{1,2}\d*", token):
        return True
    if any(part in token for part in _PHRASE_NOISE_PARTS):
        return True
    if len(token) >= 4 and any(cue in token for cue in _CLAUSE_CUE_PARTS):
        return True
    if "是" in token and len(token) >= 5:
        return True
    return False


def _prune_sub_terms(candidates: list[str], top_k: int) -> list[str]:
    if not candidates:
        return []
    domain_lexicon = _load_domain_lexicon()
    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)

    output: list[str] = []
    for term in deduped:
        if term in domain_lexicon:
            output.append(term)
            if len(output) >= max(1, top_k):
                break
            continue
        # 若已有更长术语包含当前词，抑制过泛短词。
        has_longer_cover = any(
            other != term and len(other) >= len(term) + 1 and term in other
            for other in deduped
        )
        if has_longer_cover:
            continue
        output.append(term)
        if len(output) >= max(1, top_k):
            break
    return output


def _term_sentence_coverage(text: str, terms: list[str]) -> dict[str, int]:
    if not text or not terms:
        return {}
    normalized_terms = [term for term in {normalize_keyword(item) for item in terms if normalize_keyword(item)}]
    if not normalized_terms:
        return {}

    coverage = {term: 0 for term in normalized_terms}
    for sentence_raw in split_sentences(text):
        sentence = normalize_keyword(sentence_raw)
        if not sentence:
            continue
        for term in normalized_terms:
            if term in sentence:
                coverage[term] = coverage.get(term, 0) + 1
    return coverage


def _jieba_tokens(text: str) -> list[str]:
    if jieba_posseg is None:
        return []

    accepted_prefix = ("n", "vn", "nz", "eng")
    tokens: list[str] = []
    for pair in jieba_posseg.cut(text):
        word = pair.word.strip()
        flag = pair.flag or ""
        if not word:
            continue
        if flag.startswith(accepted_prefix):
            tokens.append(word)
    return tokens


def _jieba_ranked_words(text: str, top_k: int) -> tuple[list[str], list[str]]:
    if jieba_analyse is None:
        return [], []
    tfidf_words = jieba_analyse.extract_tags(text, topK=max(10, top_k * 4), withWeight=False)
    textrank_words = jieba_analyse.textrank(text, topK=max(10, top_k * 4), withWeight=False)
    return tfidf_words, textrank_words


def _sklearn_tfidf_words(text: str, top_k: int) -> list[str]:
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return []

    try:
        vectorizer = TfidfVectorizer(max_features=3000, token_pattern=r"(?u)\b\w+\b")
        matrix = vectorizer.fit_transform(sentences)
    except Exception:
        return []

    scores = matrix.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    pairs = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    return [term for term, _ in pairs[: max(20, top_k * 4)] if term]


class HybridExtractor:
    name = "hybrid"

    def extract(self, text: str, top_k: int) -> KeywordExtractResult:
        cleaned = _clean_text(text)
        if not cleaned:
            return KeywordExtractResult(candidates=[], extractor=self.name)

        stopwords = _load_stopwords()
        domain_lexicon = _load_domain_lexicon()
        min_len = max(1, settings.kg_min_term_len)

        scores: dict[str, float] = {}
        filtered_noise = 0

        def add_score(token: str, weight: float):
            nonlocal filtered_noise
            if _looks_like_noise(token, min_len=min_len, stopwords=stopwords):
                filtered_noise += 1
                return
            key = _normalize_term(token)
            if not key:
                return
            scores[key] = scores.get(key, 0.0) + weight

        for token in _jieba_tokens(cleaned):
            add_score(token, 1.6)

        tfidf_words, textrank_words = _jieba_ranked_words(cleaned, top_k)
        for token in tfidf_words:
            add_score(token, 2.2)
        for token in textrank_words:
            add_score(token, 2.0)

        for token in _sklearn_tfidf_words(cleaned, top_k):
            add_score(token, 1.4)

        normalized_cleaned = _normalize_term(cleaned)
        for term in domain_lexicon:
            if term and term in normalized_cleaned:
                scores[term] = scores.get(term, 0.0) + 3.5

        # 术语跨句多次出现时，提高其可信度。
        coverage = _term_sentence_coverage(cleaned, list(scores.keys()))
        for term, sent_hits in coverage.items():
            if sent_hits <= 0:
                continue
            scores[term] = scores.get(term, 0.0) + min(1.6, sent_hits * 0.32)

            # 仅出现一次的超短词通常噪声大，降低其分数。
            if sent_hits == 1 and len(term) <= 2 and term not in domain_lexicon:
                scores[term] = scores.get(term, 0.0) - 0.75
            if term in _GENERIC_WEAK_TERMS and term not in domain_lexicon:
                scores[term] = scores.get(term, 0.0) - (0.55 if sent_hits >= 2 else 1.0)

        ranked = sorted(scores.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
        ranked_terms = [token for token, _ in ranked]
        candidates = _prune_sub_terms(ranked_terms, top_k=top_k)
        return KeywordExtractResult(
            candidates=candidates,
            filtered_noise=filtered_noise,
            extractor=self.name,
            fallback_used=False,
        )


@lru_cache(maxsize=1)
def _get_uie_taskflow(model_path: str | None):
    if Taskflow is None:
        raise RuntimeError("paddlenlp is not installed")

    kwargs = {"schema": ["知识点"]}
    if model_path:
        kwargs["model"] = model_path
    return Taskflow("information_extraction", **kwargs)


class UIEExtractor:
    name = "uie"

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path

    def extract(self, text: str, top_k: int) -> KeywordExtractResult:
        cleaned = _clean_text(text)
        if not cleaned:
            return KeywordExtractResult(candidates=[], extractor=self.name)

        stopwords = _load_stopwords()
        min_len = max(1, settings.kg_min_term_len)
        taskflow = _get_uie_taskflow(self.model_path)

        outputs = taskflow(cleaned[:8000])
        if not isinstance(outputs, list):
            outputs = [outputs]

        candidates: list[str] = []
        filtered_noise = 0
        seen: set[str] = set()
        for item in outputs:
            for span in (item or {}).get("知识点", []):
                raw_text = (span or {}).get("text", "")
                if _looks_like_noise(raw_text, min_len=min_len, stopwords=stopwords):
                    filtered_noise += 1
                    continue
                normalized = _normalize_term(raw_text)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                candidates.append(normalized)
                if len(candidates) >= top_k:
                    break
            if len(candidates) >= top_k:
                break

        return KeywordExtractResult(
            candidates=candidates,
            filtered_noise=filtered_noise,
            extractor=self.name,
            fallback_used=False,
        )


def _rank_terms(
    raw_terms: list[str],
    *,
    top_k: int,
) -> tuple[list[str], int]:
    stopwords = _load_stopwords()
    domain_lexicon = _load_domain_lexicon()
    min_len = max(1, settings.kg_min_term_len)

    filtered_noise = 0
    term_counter = Counter()
    for term in raw_terms:
        if _looks_like_noise(term, min_len=min_len, stopwords=stopwords):
            filtered_noise += 1
            continue
        normalized = _normalize_term(term)
        if not normalized:
            continue
        term_counter[normalized] += 1

    if not term_counter:
        return [], filtered_noise

    ranked: list[tuple[str, float]] = []
    for term, count in term_counter.items():
        score = float(count)
        if term in domain_lexicon:
            score += 1.4
        # 优先保留“短语型术语”，弱化过短 token。
        score += min(0.8, max(0.0, (len(term) - 2) * 0.1))
        ranked.append((term, score))

    ranked.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    ranked_terms = [term for term, _ in ranked]
    return _prune_sub_terms(ranked_terms, top_k=top_k), filtered_noise


def _append_term_from_value(value, output: list[str]):
    if value is None:
        return
    if isinstance(value, str):
        output.append(value)
        return
    if isinstance(value, dict):
        for key in (
            "text",
            "word",
            "entity",
            "mention",
            "name",
            "term",
            "head",
            "tail",
            "subject",
            "object",
        ):
            if key in value:
                _append_term_from_value(value.get(key), output)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _append_term_from_value(item, output)


@lru_cache(maxsize=1)
def _get_hanlp_model(model_ref: str):
    if hanlp is None:
        raise RuntimeError("hanlp is not installed")
    if not model_ref:
        raise RuntimeError("KG_HANLP_MODEL is empty")
    return hanlp.load(model_ref)


class HanLPExtractor:
    name = "hanlp"

    def __init__(self, model_ref: str | None = None):
        self.model_ref = (model_ref or "").strip()

    def _collect_terms(self, payload) -> list[str]:
        terms: list[str] = []
        if isinstance(payload, dict):
            for key, value in payload.items():
                lowered = key.lower()
                if "ner" in lowered or "tok" in lowered:
                    _append_term_from_value(value, terms)
            if not terms:
                _append_term_from_value(payload, terms)
            return terms
        _append_term_from_value(payload, terms)
        return terms

    def extract(self, text: str, top_k: int) -> KeywordExtractResult:
        cleaned = _clean_text(text)
        if not cleaned:
            return KeywordExtractResult(candidates=[], extractor=self.name)

        model = _get_hanlp_model(self.model_ref)
        payload = model(cleaned[:8000])
        candidates, filtered_noise = _rank_terms(self._collect_terms(payload), top_k=top_k)
        return KeywordExtractResult(
            candidates=candidates,
            filtered_noise=filtered_noise,
            extractor=self.name,
            fallback_used=False,
        )


class DeepKEExtractor:
    name = "deepke"

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        timeout_seconds: int = 20,
        api_key: str | None = None,
    ):
        self.endpoint = (endpoint or "").strip()
        self.timeout_seconds = max(3, int(timeout_seconds))
        self.api_key = (api_key or "").strip()

    def _collect_terms(self, payload) -> list[str]:
        terms: list[str] = []
        if isinstance(payload, dict):
            for key in (
                "candidates",
                "keywords",
                "terms",
                "entities",
                "entity_list",
                "triples",
                "relations",
                "result",
                "results",
                "data",
            ):
                if key in payload:
                    _append_term_from_value(payload.get(key), terms)
            if not terms:
                _append_term_from_value(payload, terms)
            return terms
        _append_term_from_value(payload, terms)
        return terms

    def extract(self, text: str, top_k: int) -> KeywordExtractResult:
        cleaned = _clean_text(text)
        if not cleaned:
            return KeywordExtractResult(candidates=[], extractor=self.name)
        if not self.endpoint:
            raise RuntimeError("KG_DEEPKE_ENDPOINT is empty")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request_payload = {"text": cleaned[:8000], "top_k": top_k}
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(self.endpoint, json=request_payload, headers=headers)
            response.raise_for_status()
            payload = response.json()

        candidates, filtered_noise = _rank_terms(self._collect_terms(payload), top_k=top_k)
        return KeywordExtractResult(
            candidates=candidates,
            filtered_noise=filtered_noise,
            extractor=self.name,
            fallback_used=False,
        )


def _fallback_hybrid(text: str, top_k: int, *, source: str, error: Exception) -> KeywordExtractResult:
    logger.warning("%s extractor failed, fallback to hybrid: %s", source, error)
    fallback = HybridExtractor().extract(text, top_k)
    fallback.fallback_used = True
    return fallback


def extract_keywords_with_meta(text: str, top_k: int | None = None) -> KeywordExtractResult:
    effective_top_k = max(1, top_k if top_k is not None else settings.kg_top_k)
    extractor_name = (settings.kg_extractor or "hybrid").strip().lower()

    if extractor_name == "uie":
        try:
            uie = UIEExtractor(model_path=settings.kg_uie_model_path)
            result = uie.extract(text, effective_top_k)
            # UIE 作为主链路：有有效候选时直接返回。
            if result.candidates:
                return result
        except Exception as exc:
            return _fallback_hybrid(text, effective_top_k, source="uie", error=exc)

        # UIE 无候选时自动回退 hybrid，保证候选生成稳定。
        fallback = HybridExtractor().extract(text, effective_top_k)
        fallback.fallback_used = True
        return fallback

    if extractor_name == "hanlp":
        try:
            hanlp_result = HanLPExtractor(model_ref=settings.kg_hanlp_model).extract(text, effective_top_k)
            if hanlp_result.candidates:
                return hanlp_result
        except Exception as exc:
            return _fallback_hybrid(text, effective_top_k, source="hanlp", error=exc)
        fallback = HybridExtractor().extract(text, effective_top_k)
        fallback.fallback_used = True
        return fallback

    if extractor_name == "deepke":
        try:
            deepke_result = DeepKEExtractor(
                endpoint=settings.kg_deepke_endpoint,
                timeout_seconds=settings.kg_deepke_timeout_seconds,
                api_key=settings.kg_deepke_api_key,
            ).extract(text, effective_top_k)
            if deepke_result.candidates:
                return deepke_result
        except Exception as exc:
            return _fallback_hybrid(text, effective_top_k, source="deepke", error=exc)
        fallback = HybridExtractor().extract(text, effective_top_k)
        fallback.fallback_used = True
        return fallback

    return HybridExtractor().extract(text, effective_top_k)


def extract_keywords(text: str, top_k: int = 10) -> list[str]:
    return extract_keywords_with_meta(text=text, top_k=top_k).candidates


def build_sentence_cooccurrence(text: str, terms: list[str]) -> dict[tuple[str, str], float]:
    if not text or not terms:
        return {}

    normalized_terms = []
    seen = set()
    for term in terms:
        normalized = _normalize_term(term)
        if normalized and normalized not in seen:
            seen.add(normalized)
            normalized_terms.append(normalized)

    counts: dict[tuple[str, str], float] = {}
    sentence_hits: list[set[str]] = []
    max_distance = max(8, settings.kg_max_term_distance)
    cross_sentence_penalty = max(0.0, settings.kg_cross_sentence_penalty)

    for sentence_raw in split_sentences(text):
        sentence = _normalize_term(sentence_raw)
        if not sentence:
            sentence_hits.append(set())
            continue

        position_map: dict[str, list[int]] = {}
        for term in normalized_terms:
            start = sentence.find(term)
            if start < 0:
                continue
            positions: list[int] = []
            idx = start
            while idx >= 0:
                positions.append(idx)
                idx = sentence.find(term, idx + 1)
            position_map[term] = positions

        unique_hits = sorted(position_map.keys())
        sentence_hits.append(set(unique_hits))
        if len(unique_hits) < 2:
            continue

        for left, right in itertools.combinations(unique_hits, 2):
            min_dist = min(abs(i - j) for i in position_map[left] for j in position_map[right])
            # 同句且词距更近的术语对，关系强度更高。
            score = 1.0 if min_dist <= max_distance else 0.5
            key = tuple(sorted((left, right)))
            counts[key] = counts.get(key, 0.0) + score

    if cross_sentence_penalty > 0:
        # 相邻句共现按弱证据处理，并通过惩罚系数控制权重。
        for idx in range(len(sentence_hits) - 1):
            current_hits = sentence_hits[idx]
            next_hits = sentence_hits[idx + 1]
            if not current_hits or not next_hits:
                continue
            for left in current_hits:
                for right in next_hits:
                    if left == right:
                        continue
                    key = tuple(sorted((left, right)))
                    counts[key] = counts.get(key, 0.0) + cross_sentence_penalty

    return counts


def normalize_keyword(term: str) -> str:
    return _normalize_term(term)


def find_source_sentence(text: str, term: str) -> str:
    normalized_term = _normalize_term(term)
    if not normalized_term:
        return ""
    for sentence in split_sentences(text):
        if normalized_term in _normalize_term(sentence):
            return sentence.strip()[:200]
    return ""
