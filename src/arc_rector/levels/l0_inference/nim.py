"""L0 alternative: NVIDIA NIM, hosted open weights on someone else's GPUs.

NIM is the escape hatch for machines that cannot run a model locally. It serves
the same open-weights checkpoints you would run under Ollama -- Llama, Mistral,
Nemotron, Qwen, Gemma -- behind an OpenAI-compatible REST API at
`https://integrate.api.nvidia.com/v1`, on free-tier credits from build.nvidia.com.
No GPU, no download, an 8B model answering in a second.

The tradeoff is that it is the only adapter in this level that needs a key and a
network round trip, which breaks the project's zero-key default and puts your
corpus text on a third party's servers. Use it for a demo on a thin laptop, or
to sanity-check whether a bad answer is your pipeline's fault or your local
model's. Keep Ollama as the default for anything you would not paste into a
web form.

Because it is OpenAI-shaped, this adapter is also a working template for any
other OpenAI-compatible provider: change `base_url` and the env var name.

The key is read from the environment only. `Config.load()` populates it from a
gitignored `.env` at the repo root. It is never a config.yaml setting, never a
constructor argument, and never appears in an error message.
"""

from __future__ import annotations

import os
from typing import Any

from ...interfaces import Inference
from ...registry import require

API_KEY_ENV = "NVIDIA_API_KEY"


class NimInference(Inference):
    name = "nim"

    def __init__(
        self,
        *,
        model: str = "meta/llama-3.1-8b-instruct",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        temperature: float = 0.1,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        timeout: int = 120,
        api_key_env: str = API_KEY_ENV,
        **_: Any,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        # The name of the variable, never the value of it.
        self.api_key_env = api_key_env
        self._requests: Any = None

    @property
    def _http(self) -> Any:
        if self._requests is None:
            self._requests = require("requests", self.name)
        return self._requests

    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"{self.api_key_env} is not set, so the 'nim' adapter cannot run.\n"
                f"    1. Get a free key at https://build.nvidia.com (Get API Key).\n"
                f"    2. Add this line to the gitignored .env in the repo root:\n"
                f"           {self.api_key_env}=nvapi-...\n"
                f"    3. Or run entirely offline with:  ARC_L0_INFERENCE=ollama\n"
                f"The key is read from the environment only -- do not put it in config.yaml."
            )
        return key

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
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        response = self._http.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            # response.text is safe to show; the key lives only in the request header.
            raise RuntimeError(
                f"NVIDIA NIM chat failed ({response.status_code}) for model "
                f"{payload['model']}: {response.text[:300]}\n"
                f"401 means the key in {self.api_key_env} is wrong or expired; "
                f"404 usually means that model id does not exist on build.nvidia.com."
            )

        choices = response.json().get("choices") or []
        if not choices:
            raise RuntimeError(f"NVIDIA NIM returned no choices: {response.text[:300]}")
        return str(choices[0].get("message", {}).get("content", "")).strip()

    def available(self) -> bool:
        """Key presence only. A live probe would spend free-tier credits on a health check."""
        return bool(os.environ.get(self.api_key_env, "").strip())
