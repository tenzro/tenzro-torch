"""nn.Modules whose forward pass is an HTTP call to a Tenzro node.

They hold no weights: the model runs on the node (or whichever provider the node
routes to). Only the OpenAI-shaped routes the node serves are used:
``POST /v1/embeddings`` and ``POST /v1/chat/completions``.
"""
from __future__ import annotations

from typing import Any

import httpx
import torch
from torch import nn


class TenzroApiError(RuntimeError):
    """The node answered with an error body ``{"error": {message, type, code}}``."""

    def __init__(self, status: int, message: str, type: str | None = None, code: str | None = None):
        super().__init__(f"HTTP {status} {code or ''}: {message}".replace("  ", " "))
        self.status, self.message, self.type, self.code = status, message, type, code


def _post(client: httpx.Client, url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    r = client.post(url, json=body, headers=headers)
    if r.status_code >= 400:
        try:
            e = r.json()["error"]
            raise TenzroApiError(r.status_code, e.get("message", ""), e.get("type"), e.get("code"))
        except (ValueError, KeyError, TypeError):
            raise TenzroApiError(r.status_code, r.text[:200]) from None
    return r.json()


class _Remote(nn.Module):
    def __init__(self, base_url: str, model: str, api_key: str | None, timeout: float):
        super().__init__()
        self.base_url = base_url.rstrip("/")  # e.g. http://127.0.0.1:8545/v1
        self.model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(timeout=timeout)


class RemoteEmbedding(_Remote):
    """``forward(texts) -> Tensor[n, dim]`` (float32) from ``/v1/embeddings``.

    The result carries no autograd graph: the node returns vectors, not gradients.
    """

    def __init__(self, base_url: str, model: str, *, api_key: str | None = None, timeout: float = 60.0,
                 normalize: bool = False, **extra: Any):
        super().__init__(base_url, model, api_key, timeout)
        self.normalize, self.extra = normalize, extra

    def forward(self, texts: str | list[str]) -> torch.Tensor:
        body = {"model": self.model, "input": texts, "encoding_format": "float", **self.extra}
        if self.normalize:
            body["normalize"] = True
        data = _post(self._http, f"{self.base_url}/embeddings", body, self._headers)["data"]
        rows = [d["embedding"] for d in sorted(data, key=lambda d: d.get("index", 0))]
        return torch.tensor(rows, dtype=torch.float32)


class RemoteChat(_Remote):
    """``forward(messages) -> str``: the assistant text from ``/v1/chat/completions`` (non-streaming)."""

    def __init__(self, base_url: str, model: str, *, api_key: str | None = None, timeout: float = 120.0, **params: Any):
        super().__init__(base_url, model, api_key, timeout)
        self.params = params

    def forward(self, messages: str | list[dict[str, str]]) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        body = {"model": self.model, "messages": messages, "stream": False, **self.params}
        out = _post(self._http, f"{self.base_url}/chat/completions", body, self._headers)
        return out["choices"][0]["message"]["content"]
