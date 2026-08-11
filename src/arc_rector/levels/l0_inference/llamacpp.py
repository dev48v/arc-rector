"""L0 alternative: llama.cpp's `llama-server`. The bottom of the hardware ladder.

This is the option that runs where nothing else will: no GPU, 8 GB of RAM, an old
laptop, a Raspberry Pi, a locked-down work machine where you cannot install a
service. llama.cpp is a single C++ binary with no Python and no runtime deps, and
GGUF quantisation is what makes it fit -- an 8B model at Q4_K_M is roughly 4.5 GB
instead of 16 GB at full precision, for a small and usually acceptable quality
loss. Q8 if you have the RAM, Q4 if you do not.

    llama-server -m models/llama-3.1-8b-instruct-Q4_K_M.gguf --port 8080

Then `ARC_L0_INFERENCE=llamacpp`. It speaks the same OpenAI-compatible
`/v1/chat/completions` as vLLM and NIM, so this adapter is the same shape as both.

Ollama is still the default because it fetches and manages the weights for you --
llama.cpp expects you to find the right GGUF on the Hub yourself. Reach for this
when you want no daemon, no model store, and full control over exactly which
quantisation is loaded. Note that `model` is advisory: llama-server serves the one
file you launched it with, whatever you put in the request.
"""

from __future__ import annotations

from typing import Any

from ...interfaces import Inference
from ...registry import require


class LlamaCppInference(Inference):
    name = "llamacpp"

    def __init__(
        self,
        *,
        model: str = "local-model",
        base_url: str = "http://localhost:8080/v1",
        temperature: float = 0.1,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        timeout: int = 300,
        **_: Any,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        # Generous by default: CPU-only generation is slow, and a timeout mid-answer
        # looks exactly like a broken pipeline.
        self.timeout = timeout
        self._requests: Any = None

    @property
    def _http(self) -> Any:
        if self._requests is None:
            self._requests = require("requests", self.name)
        return self._requests

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
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"llama.cpp chat failed ({response.status_code}) at {self.base_url}: "
                f"{response.text[:300]}\n"
                f"Is llama-server running? Start it with:\n"
                f"    llama-server -m <model>.gguf --port 8080"
            )

        choices = response.json().get("choices") or []
        if not choices:
            raise RuntimeError(f"llama.cpp returned no choices: {response.text[:300]}")
        return str(choices[0].get("message", {}).get("content", "")).strip()

    def available(self) -> bool:
        """Local and free to probe. llama-server also exposes /health off the server root."""
        try:
            response = self._http.get(f"{self.base_url}/models", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
