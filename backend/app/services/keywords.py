from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    if not text or not text.strip():
        return []

    vectorizer = TfidfVectorizer(max_features=2000)
    tfidf = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf.toarray()[0]

    pairs = list(zip(feature_names, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [w for w, s in pairs[:top_k] if w]
