import httpx
from app.config import settings

async def call_glm(prompt: str) -> str:
    if not settings.glm_api_key:
        return "[GLM占位答复] " + prompt[:100]
    # Placeholder: adapt to actual GLM API
    headers = {"Authorization": f"Bearer {settings.glm_api_key}"}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post("https://open.bigmodel.cn/api/paas/v4/chat/completions", headers=headers, json={"messages": [{"role": "user", "content": prompt}]})
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        return f"[GLM接口错误]{e}"

async def call_ernie(prompt: str) -> str:
    if not settings.ernie_api_key:
        return "[文心占位答复] " + prompt[:100]
    # Placeholder endpoint
    headers = {"Authorization": f"Bearer {settings.ernie_api_key}"}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post("https://aip.baidubce.com/rpc/2.0/ernie/v1/chat", headers=headers, json={"messages": [{"role": "user", "content": prompt}]})
            resp.raise_for_status()
            data = resp.json()
            return data.get("result", "")
    except Exception as e:
        return f"[文心接口错误]{e}"

async def call_llm(prompt: str) -> str:
    provider = settings.model_provider.lower()
    if provider == "glm":
        return await call_glm(prompt)
    return await call_ernie(prompt)
