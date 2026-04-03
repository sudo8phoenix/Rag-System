"""Ollama LLM wrapper with health checks and streaming support."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Callable, Sequence
from urllib import error, request

from src.config.settings import AppConfig, LLMConfig

from .base import LLMConnectionError, LLMProviderError, LLMResponse, LLMStreamToken
from .prompting import build_user_prompt

JSONDict = dict[str, Any]
HttpGetJson = Callable[[str, float], JSONDict]
HttpPostJson = Callable[[str, JSONDict, float], JSONDict]
HttpPostJsonStream = Callable[[str, JSONDict, float], Iterator[JSONDict]]


def _http_get_json(url: str, timeout: float) -> JSONDict:
    req = request.Request(url=url, method="GET")
    with request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - trusted local endpoint
        payload = response.read().decode("utf-8")
    return json.loads(payload) if payload else {}


def _http_post_json(url: str, payload: JSONDict, timeout: float) -> JSONDict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - trusted local endpoint
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def _http_post_json_stream(url: str, payload: JSONDict, timeout: float) -> Iterator[JSONDict]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - trusted local endpoint
        for line in response:
            if not line:
                continue
            decoded = line.decode("utf-8").strip()
            if decoded:
                yield json.loads(decoded)


@dataclass(frozen=True)
class OllamaStatus:
    """Connectivity and model readiness status for Ollama."""

    api_reachable: bool
    model_available: bool
    available_models: list[str]


class OllamaLLM:
    """Thin wrapper around Ollama's `/api/generate` endpoint."""

    DEFAULT_TIMEOUT_SECONDS = 120.0

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        http_get_json: HttpGetJson | None = None,
        http_post_json: HttpPostJson | None = None,
        http_post_json_stream: HttpPostJsonStream | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.config = config or LLMConfig()
        self.base_url = self.config.base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._http_get_json = http_get_json or _http_get_json
        self._http_post_json = http_post_json or _http_post_json
        self._http_post_json_stream = http_post_json_stream or _http_post_json_stream

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "OllamaLLM":
        return cls(config=config.llm)

    def _get(self, endpoint: str) -> JSONDict:
        url = f"{self.base_url}{endpoint}"
        try:
            return self._http_get_json(url, self.timeout_seconds)
        except (error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise LLMConnectionError(
                f"Unable to connect to Ollama at {self.base_url}. "
                "Ensure Ollama is running (`ollama serve`)."
            ) from exc

    def _post(self, endpoint: str, payload: JSONDict) -> JSONDict:
        url = f"{self.base_url}{endpoint}"
        try:
            return self._http_post_json(url, payload, self.timeout_seconds)
        except (error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise LLMProviderError(f"Ollama request failed for {endpoint}: {exc}") from exc

    def _post_stream(self, endpoint: str, payload: JSONDict) -> Iterator[JSONDict]:
        url = f"{self.base_url}{endpoint}"
        try:
            yield from self._http_post_json_stream(url, payload, self.timeout_seconds)
        except (error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise LLMProviderError(f"Ollama streaming request failed for {endpoint}: {exc}") from exc

    def list_models(self) -> list[str]:
        """Return locally available Ollama model names."""

        response = self._get("/api/tags")
        models = response.get("models", [])
        names: list[str] = []
        for model in models:
            if isinstance(model, dict) and isinstance(model.get("name"), str):
                names.append(model["name"])
        return names

    def pull_model(self, model_name: str | None = None) -> None:
        """Pull a model into local Ollama cache."""

        self._post("/api/pull", {"name": model_name or self.config.model, "stream": False})

    def check_status(self) -> OllamaStatus:
        """Check endpoint reachability and whether configured model is available."""

        names = self.list_models()
        requested = self.config.model
        requested_with_latest = f"{requested}:latest"
        return OllamaStatus(
            api_reachable=True,
            model_available=requested in names or requested_with_latest in names,
            available_models=names,
        )

    def verify_ready(self, *, auto_pull: bool = False) -> OllamaStatus:
        """Ensure Ollama endpoint is reachable and target model exists."""

        status = self.check_status()
        if status.model_available:
            return status

        if not auto_pull:
            raise LLMConnectionError(
                "Configured model is not available in Ollama. "
                f"Run `ollama pull {self.config.model}` and retry."
            )

        self.pull_model(self.config.model)
        return self.check_status()

    def _build_payload(
        self,
        query: str,
        *,
        context_items: Sequence[str | dict[str, object]] | None = None,
        system_prompt: str | None = None,
        stream: bool = False,
    ) -> JSONDict:
        return {
            "model": self.config.model,
            "system": system_prompt or self.config.system_prompt,
            "prompt": build_user_prompt(query, context_items),
            "stream": stream,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

    def generate(
        self,
        query: str,
        *,
        context_items: Sequence[str | dict[str, object]] | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Return a single complete response from Ollama."""

        payload = self._build_payload(
            query,
            context_items=context_items,
            system_prompt=system_prompt,
            stream=False,
        )
        result = self._post("/api/generate", payload)
        text = str(result.get("response", "")).strip()
        return LLMResponse(
            text=text,
            model=str(result.get("model", self.config.model)),
            prompt=payload["prompt"],
            done_reason=str(result.get("done_reason")) if result.get("done_reason") else None,
            raw=result,
        )

    def generate_stream(
        self,
        query: str,
        *,
        context_items: Sequence[str | dict[str, object]] | None = None,
        system_prompt: str | None = None,
    ) -> Iterator[LLMStreamToken]:
        """Yield response chunks token-by-token from Ollama streaming API."""

        payload = self._build_payload(
            query,
            context_items=context_items,
            system_prompt=system_prompt,
            stream=True,
        )
        for event in self._post_stream("/api/generate", payload):
            token = str(event.get("response", ""))
            done = bool(event.get("done", False))
            if token or done:
                yield LLMStreamToken(token=token, done=done)
