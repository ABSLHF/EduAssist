from typing import List, Tuple


def grade_text_submission(content: str, keywords: List[str]) -> Tuple[int, str]:
    if not content:
        return 0, "未提交内容"
    if not keywords:
        return 60, "未设置关键词，给出基础分"

    content_lower = content.lower()
    hit = 0
    for kw in keywords:
        if kw and kw.lower() in content_lower:
            hit += 1

    ratio = hit / max(1, len(keywords))
    score = int(60 + ratio * 40)
    feedback = f"关键词命中 {hit}/{len(keywords)}。建议补充未覆盖的知识点。"
    return score, feedback
