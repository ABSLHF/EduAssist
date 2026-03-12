import asyncio
import json

import httpx

from app.config import settings


def _extract_text(data: dict) -> str:
    answer = data.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()

    nested_answer = data.get("data", {}).get("answer")
    if isinstance(nested_answer, str) and nested_answer.strip():
        return nested_answer.strip()

    outputs = data.get("data", {}).get("outputs", {})
    if isinstance(outputs, dict):
        for key in ("answer", "text", "result", "output", "content"):
            value = outputs.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in outputs.values():
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


async def _consume_dify_sse(resp: httpx.Response) -> tuple[str, str | None]:
    answer_parts: list[str] = []
    parse_error: str | None = None

    async for raw_line in resp.aiter_lines():
        line = (raw_line or "").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue

        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue

        try:
            data = json.loads(payload)
        except Exception:
            parse_error = f"invalid sse payload: {payload[:160]}"
            continue

        event = str(data.get("event", "")).strip().lower()
        if event == "error":
            err_msg = data.get("message") or data.get("error") or str(data)
            return "", f"sse error event: {err_msg}"

        piece = _extract_text(data)
        if piece:
            answer_parts.append(piece)

        if event in {"message_end", "workflow_finished"}:
            break

    answer = "".join(answer_parts).strip()
    if answer:
        return answer, None
    if parse_error:
        return "", parse_error
    return "", "empty streaming answer"


async def call_dify(prompt: str) -> str:
    api_key = (settings.dify_api_key or "").strip()
    base_url = (settings.dify_base_url or "").strip().rstrip("/")
    if not api_key or not base_url:
        return "[Dify占位答复] " + prompt[:100]

    endpoint = f"{base_url}/chat-messages"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": {},
        "query": prompt,
        "response_mode": "streaming",
        "user": "eduassist-demo",
    }

    try:
        timeout = httpx.Timeout(60.0, connect=15.0)
        # Only Dify ignores system proxy to avoid invalid local proxy settings.
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            async with client.stream("POST", endpoint, headers=headers, json=payload) as resp:
                if resp.status_code >= 400:
                    body_text = await resp.aread()
                    try:
                        body = json.loads(body_text.decode("utf-8", errors="ignore"))
                    except Exception:
                        body = body_text.decode("utf-8", errors="ignore")
                    brief = str(body)
                    if len(brief) > 260:
                        brief = brief[:260] + "..."
                    return f"[Dify接口错误]{endpoint}: {resp.status_code} {brief}"

                answer, error = await _consume_dify_sse(resp)
                if error:
                    return f"[Dify接口错误]{endpoint}: {error}"
                return answer
    except Exception as exc:
        return f"[Dify接口错误]{endpoint}: {exc}"


async def call_glm(prompt: str) -> str:
    if not settings.glm_api_key:
        return "[GLM占位答复] " + prompt[:100]
    headers = {"Authorization": f"Bearer {settings.glm_api_key}"}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                headers=headers,
                json={"messages": [{"role": "user", "content": prompt}]},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        return f"[GLM接口错误]{e}"


async def call_ernie(prompt: str) -> str:
    if not settings.ernie_api_key:
        return "[文心占位答复] " + prompt[:100]
    headers = {"Authorization": f"Bearer {settings.ernie_api_key}"}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://aip.baidubce.com/rpc/2.0/ernie/v1/chat",
                headers=headers,
                json={"messages": [{"role": "user", "content": prompt}]},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("result", "")
    except Exception as e:
        return f"[文心接口错误]{e}"


async def call_llm(prompt: str) -> str:
    provider = (settings.model_provider or "glm").strip().lower()
    timeout_seconds = max(5, int(getattr(settings, "llm_timeout_seconds", 70)))

    label = "文心"
    runner = call_ernie(prompt)
    if provider == "dify":
        label = "Dify"
        runner = call_dify(prompt)
    elif provider == "glm":
        label = "GLM"
        runner = call_glm(prompt)

    try:
        return await asyncio.wait_for(runner, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return f"[{label}接口错误]timeout after {timeout_seconds}s"
