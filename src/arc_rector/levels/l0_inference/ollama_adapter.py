"""L0 default: Ollama (MIT runtime, serving open-weights models).

Ollama is the default because it is the shortest path from "nothing installed"
to "a model is answering": one installer, `ollama pull <model>`, done. It
handles GGUF quantisation, GPU offload and model lifecycle without any Python ML
dependency -- this adapter is plain HTTP.

Note the licence split that trips people up: the Ollama *runtime* is MIT, but the
*weights* you pull through it are not necessarily open source. Llama 3.x ships
under the Llama Community Licence (acceptable-use policy plus a 700M-MAU clause),
which is open *weights*, not OSI open *source*. See the licence table in the
README before shipping anything commercial.
"""

from __future__ import annotations

from typing import Any

from ...interfaces import Inference
from ...registry import require


class OllamaInference(Inference):
    name = "ollama"

    def __init__(
        self,
        *,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        num_ctx: int = 8192,
        timeout: int = 180,
        num_thread: int = 0,
        num_predict: int = 0,
        **_: Any,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout
        # On a CPU-only box Ollama's auto-detected thread count is often far
        # below the core count, which is the difference between a 20-second and
        # a two-minute answer. 0 keeps Ollama's own choice.
        self.num_thread = num_thread
        # Caps runaway generations; 0 leaves the model's default.
        self.num_predict = num_predict
        self._requests: Any = None

    @property
    def _http(self) -> Any:
        if self._requests is None:
            self._requests = require("requests", "ollama")
        return self._requests

    def complete(self, prompt: str, system: str = "", **kwargs: Any) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options: dict[str, Any] = {
            "temperature": kwargs.get("temperature", self.temperature),
            "num_ctx": kwargs.get("num_ctx", self.num_ctx),
        }
        num_thread = int(kwargs.get("num_thread", self.num_thread) or 0)
        if num_thread > 0:
            options["num_thread"] = num_thread
        num_predict = int(kwargs.get("num_predict", self.num_predict) or 0)
        if num_predict > 0:
            options["num_predict"] = num_predict

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        response = self._http.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama chat failed ({response.status_code}) at {self.base_url}: "
                f"{response.text[:300]}\n"
                f"Is Ollama running, and have you run `ollama pull {self.model}`?"
            )
        return str(response.json().get("message", {}).get("content", "")).strip()

    def available(self) -> bool:
        """True only when the server is up AND this model is actually pulled."""
        try:
            response = self._http.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            return any(m.get("name") == self.model for m in response.json().get("models", []))
        except Exception:
            return False

    def installed_models(self) -> list[str]:
        try:
            response = self._http.get(f"{self.base_url}/api/tags", timeout=5)
            return [m.get("name", "") for m in response.json().get("models", [])]
        except Exception:
            return []
