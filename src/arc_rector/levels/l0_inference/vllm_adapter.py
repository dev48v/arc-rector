"""L0 alternative: a self-hosted vLLM server. The production answer, not the laptop one.

vLLM is what you move to when Ollama's throughput stops being enough. PagedAttention
plus continuous batching means it serves many concurrent requests off one set of
weights at a fraction of the memory waste, which is the difference between a demo
and a service. Same open weights, same OpenAI-compatible surface, no key.

It is not the default for one blunt reason: vLLM needs an NVIDIA CUDA GPU. It does
not run on Windows, and it does not run on Apple Silicon. Ollama runs on all three.
So the default has to be the one that works on the machine you are reading this on,
and vLLM is the one line you change in config.yaml when you deploy to a GPU box:

    vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

Then `ARC_L0_INFERENCE=vllm`. Nothing else in the codebase moves.

The server is unauthenticated by default. If you started it with `--api-key`, set
that value in the env var named by `api_key_env` -- never inline in config.yaml.
"""

from __future__ import annotations

import os
from typing import Any

from ...interfaces import Inference
from ...registry import require


class VllmInference(Inference):
    name = "vllm"

    def __init__(
        self,
        *,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        temperature: float = 0.1,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        timeout: int = 180,
        api_key_env: str = "VLLM_API_KEY",
        **_: Any,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        # Optional: only used if you ran `vllm serve --api-key ...`.
        self.api_key_env = api_key_env
        self._requests: Any = None

    @property
    def _http(self) -> Any:
        if self._requests is None:
            self._requests = require("requests", self.name)
        return self._requests

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env, "").strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def complete(self, prompt: str, system: str = "", **kwargs: Any) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False,
        }
        response = self._http.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"vLLM chat failed ({response.status_code}) at {self.base_url}: "
                f"{response.text[:300]}\n"
                f"404 on the model id is the usual cause: vLLM serves exactly the model "
                f"it was launched with. Run `curl {self.base_url}/models` to see its real name."
            )

        choices = response.json().get("choices") or []
        if not choices:
            raise RuntimeError(f"vLLM returned no choices: {response.text[:300]}")
        return str(choices[0].get("message", {}).get("content", "")).strip()

    def available(self) -> bool:
        """Self-hosted, so a real probe is free: list what the server is serving."""
        try:
            response = self._http.get(
                f"{self.base_url}/models", headers=self._headers(), timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def served_models(self) -> list[str]:
        """The ids vLLM will actually accept, for when `model` does not match."""
        try:
            response = self._http.get(
                f"{self.base_url}/models", headers=self._headers(), timeout=5
            )
            return [m.get("id", "") for m in response.json().get("data", [])]
        except Exception:
            return []
