from __future__ import annotations

import base64
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover - 运行时依赖检查
    httpx = None  # type: ignore[assignment]


QWEN3_VL_API_URL = "http://10.200.109.61:30004/v1/chat/completions"
QWEN3_VL_MODEL = "/models/Qwen3-VL-8B-Instruct"
FALLBACK_TEXT = "[图片识别失败]"


def _require_httpx() -> None:
    if httpx is None:
        raise RuntimeError("缺少依赖 httpx，请先安装后再启用图片识别功能。")


def _build_payload(image_b64: str) -> dict[str, Any]:
    return {
        "model": QWEN3_VL_MODEL,
        "messages": [
            {"role": "system", "content": "请详细描述这张图片的内容"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请详细描述这张图片的内容"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ],
        "temperature": 0.2,
    }


def describe_image(image_bytes: bytes) -> str:
    """调用 Qwen3-VL 识别图片内容，失败时返回占位文本。"""
    if not image_bytes:
        return FALLBACK_TEXT

    try:
        _require_httpx()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = _build_payload(image_b64)
        with httpx.Client(timeout=60.0) as client:
            response = client.post(QWEN3_VL_API_URL, json=payload)
            response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return FALLBACK_TEXT
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        return FALLBACK_TEXT
    except Exception:
        return FALLBACK_TEXT
