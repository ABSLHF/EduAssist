from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

CURRENT_FILE = Path(__file__).resolve()
BACKEND_ROOT = CURRENT_FILE.parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
os.chdir(BACKEND_ROOT)

from app.llm.client import call_llm
from app.services.keywords import HybridExtractor, build_sentence_cooccurrence, normalize_keyword

app = FastAPI(title="EduAssist DeepKE Compatible Server", version="0.1.0")


class ExtractRequest(BaseModel):
    text: str = Field(default="", min_length=1)
    top_k: int = Field(default=12, ge=1, le=60)


def _strip_json_code_block(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _try_parse_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    raw = _strip_json_code_block(text)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Fallback: try to parse first JSON object fragment.
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        frag = raw[start : end + 1]
        try:
            parsed = json.loads(frag)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def _collect_terms(value: Any, output: list[str]):
    if value is None:
        return
    if isinstance(value, str):
        output.append(value)
        return
    if isinstance(value, dict):
        for key in ("text", "word", "entity", "term", "name", "head", "tail", "subject", "object"):
            if key in value:
                _collect_terms(value.get(key), output)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_terms(item, output)


def _normalize_unique(items: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        norm = normalize_keyword(item)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        output.append(norm)
        if len(output) >= limit:
            break
    return output


async def _extract_terms_with_llm(text: str, top_k: int) -> tuple[list[str], list[dict[str, str]]]:
    prompt = f"""你是中文知识抽取器。请从文本中抽取课程知识点和关系。
只返回 JSON，不要输出任何解释或代码块。
JSON格式：
{{
  "candidates": ["术语1","术语2"],
  "triples": [{{"head":"术语1","relation":"relates_to","tail":"术语2"}}]
}}
要求：
1) candidates 最多 {top_k} 个，优先名词短语与专业术语。
2) relation 统一用 "relates_to"。
3) 不要输出空字段。

文本：
{text[:4000]}
"""
    response = await call_llm(prompt)
    if response.startswith("[") and "接口错误" in response:
        return [], []

    data = _try_parse_json(response)
    if not data:
        return [], []

    raw_terms: list[str] = []
    for key in ("candidates", "keywords", "terms", "entities"):
        if key in data:
            _collect_terms(data.get(key), raw_terms)

    triples: list[dict[str, str]] = []
    for row in data.get("triples", []) if isinstance(data.get("triples"), list) else []:
        if not isinstance(row, dict):
            continue
        head = normalize_keyword(str(row.get("head", "")).strip())
        tail = normalize_keyword(str(row.get("tail", "")).strip())
        if not head or not tail or head == tail:
            continue
        triples.append({"head": head, "relation": "relates_to", "tail": tail})

    return _normalize_unique(raw_terms, top_k), triples


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/extract")
async def extract(req: ExtractRequest):
    text = (req.text or "").strip()
    top_k = max(1, min(60, int(req.top_k)))

    # Baseline extraction.
    hybrid = HybridExtractor().extract(text, top_k=top_k)
    merged_terms = list(hybrid.candidates)

    # Optional LLM enhancement.
    llm_terms, llm_triples = await _extract_terms_with_llm(text, top_k)
    for term in llm_terms:
        if term not in merged_terms:
            merged_terms.append(term)
    merged_terms = _normalize_unique(merged_terms, top_k)

    triples: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()
    for row in llm_triples:
        h = row["head"]
        t = row["tail"]
        key = tuple(sorted((h, t)))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        triples.append(row)

    # Fallback relation generation by cooccurrence.
    if len(triples) < 3 and len(merged_terms) >= 2:
        pair_scores = build_sentence_cooccurrence(text=text, terms=merged_terms)
        sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)
        for (left, right), score in sorted_pairs:
            if score <= 0:
                continue
            key = tuple(sorted((left, right)))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            triples.append({"head": left, "relation": "relates_to", "tail": right})
            if len(triples) >= max(4, top_k // 2):
                break

    return {
        "candidates": merged_terms,
        "entities": [{"text": term} for term in merged_terms],
        "triples": triples,
        "meta": {
            "source": "deepke_compat",
            "llm_terms": len(llm_terms),
            "hybrid_terms": len(hybrid.candidates),
        },
    }


if __name__ == "__main__":
    uvicorn.run("scripts.deepke_compat_server:app", host="127.0.0.1", port=8001, reload=False)
