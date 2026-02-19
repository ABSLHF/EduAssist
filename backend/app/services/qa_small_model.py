from functools import lru_cache


@lru_cache(maxsize=2)
def _load_qa_pipeline(model_path: str):
    try:
        from transformers import pipeline  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency 'transformers'. Install: pip install transformers torch") from exc
    return pipeline("question-answering", model=model_path, tokenizer=model_path)


def predict_answer(question: str, context: str, model_path: str) -> tuple[str, float | None]:
    qa_pipe = _load_qa_pipeline(model_path)
    result = qa_pipe(question=question, context=context)
    answer = str(result.get("answer", "")).strip()
    score = result.get("score")
    confidence = float(score) if isinstance(score, (float, int)) else None
    return answer, confidence
