"""Groq LLM wrapper with robust response parsing and streaming support."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib import error, request

from src.config.settings import AppConfig, LLMConfig

from .base import LLMConnectionError, LLMProviderError, LLMResponse, LLMStreamToken
from .prompting import build_user_prompt

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency at import time
    OpenAI = None  # type: ignore[assignment]

JSONDict = dict[str, Any]
HttpGetJson = Callable[[str, dict[str, str], float], JSONDict]
HttpPostJson = Callable[[str, JSONDict, dict[str, str], float], JSONDict]
HttpPostJsonStream = Callable[
    [str, JSONDict, dict[str, str], float], Iterator[JSONDict]
]

EMPTY_RESPONSE_FALLBACK = (
    "I could not generate a response for that request. "
    "Please try again with more context or a more specific question."
)


def _read_env_file_value(key: str) -> str | None:
    """Read a single key from the project .env file if present."""

    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    if not env_path.exists():
        return None

    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            if name.strip() == key:
                return value.strip().strip('"').strip("'")
    except OSError:
        return None

    return None


def _http_get_json(url: str, headers: dict[str, str], timeout: float) -> JSONDict:
    req = request.Request(url=url, method="GET", headers=headers)
    with request.urlopen(
        req, timeout=timeout
    ) as response:  # noqa: S310 - trusted API URL
        payload = response.read().decode("utf-8")
    return json.loads(payload) if payload else {}


def _http_post_json(
    url: str,
    payload: JSONDict,
    headers: dict[str, str],
    timeout: float,
) -> JSONDict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, data=data, method="POST", headers=headers)
    with request.urlopen(
        req, timeout=timeout
    ) as response:  # noqa: S310 - trusted API URL
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def _http_post_json_stream(
    url: str,
    payload: JSONDict,
    headers: dict[str, str],
    timeout: float,
) -> Iterator[JSONDict]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, data=data, method="POST", headers=headers)
    with request.urlopen(
        req, timeout=timeout
    ) as response:  # noqa: S310 - trusted API URL
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue

            data_part = line[len("data:") :].strip()
            if data_part == "[DONE]":
                break
            if data_part:
                yield json.loads(data_part)


@dataclass(frozen=True)
class GroqStatus:
    """Connectivity and model readiness status for Groq."""

    api_reachable: bool
    model_available: bool
    available_models: list[str]


class GroqLLM:
    """Thin wrapper around Groq's OpenAI-compatible endpoints."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        http_get_json: HttpGetJson | None = None,
        http_post_json: HttpPostJson | None = None,
        http_post_json_stream: HttpPostJsonStream | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.config = config or LLMConfig()
        self.base_url = self.config.base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._http_get_json = http_get_json or _http_get_json
        self._http_post_json = http_post_json or _http_post_json
        self._http_post_json_stream = http_post_json_stream or _http_post_json_stream
        self._has_custom_http = any(
            item is not None
            for item in (http_get_json, http_post_json, http_post_json_stream)
        )
        self._openai_client: OpenAI | None = None

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "GroqLLM":
        return cls(config=config.llm)

    def _resolve_api_key(self) -> str:
        api_key = (
            self.config.api_key
            or os.getenv("GROQ_API_KEY")
            or _read_env_file_value("GROQ_API_KEY")
        )
        if not api_key:
            raise LLMConnectionError(
                "Missing API key for Groq. Set llm.api_key in config or export GROQ_API_KEY."
            )
        return api_key

    def _headers(self) -> dict[str, str]:
        api_key = self._resolve_api_key()
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _use_openai_sdk(self) -> bool:
        return OpenAI is not None and not self._has_custom_http

    def _client(self) -> OpenAI:
        if not self._use_openai_sdk() or OpenAI is None:
            raise LLMProviderError(
                "OpenAI SDK is unavailable for Groq client mode. "
                "Install package openai or use HTTP fallback."
            )

        if self._openai_client is None:
            self._openai_client = OpenAI(
                api_key=self._resolve_api_key(),
                base_url=self.base_url,
                timeout=self.timeout_seconds,
            )
        return self._openai_client

    def _format_openai_error(self, exc: Exception, endpoint: str) -> str:
        status_code = getattr(exc, "status_code", None)
        message = str(exc).strip()

        if status_code in {401, 403}:
            guidance = (
                "Groq authentication/authorization failed. "
                "Check GROQ_API_KEY, model access permissions, and account limits."
            )
            return (
                f"Groq request failed for {endpoint}: HTTP {status_code}. {guidance}"
                + (f" Detail: {message}" if message else "")
            )

        if status_code is not None:
            return f"Groq request failed for {endpoint}: HTTP {status_code}" + (
                f". Detail: {message}" if message else ""
            )

        return f"Groq request failed for {endpoint}: {message or type(exc).__name__}"

    def _extract_http_error_detail(self, exc: error.HTTPError) -> str:
        detail = ""
        try:
            body = exc.read().decode("utf-8")
            if not body:
                return ""
            parsed = json.loads(body)
            if not isinstance(parsed, dict):
                return ""
            if isinstance(parsed.get("error"), dict):
                detail = str(parsed["error"].get("message", "")).strip()
            elif parsed.get("error"):
                detail = str(parsed["error"]).strip()
            elif parsed.get("message"):
                detail = str(parsed["message"]).strip()
        except Exception:
            return ""
        return detail

    def _format_http_error(self, exc: error.HTTPError, endpoint: str) -> str:
        detail = self._extract_http_error_detail(exc)

        if exc.code in {401, 403}:
            guidance = (
                "Groq authentication/authorization failed. "
                "Check GROQ_API_KEY, model access permissions, and account limits."
            )
            return (
                f"Groq request failed for {endpoint}: HTTP {exc.code}. {guidance}"
                + (f" Detail: {detail}" if detail else "")
            )

        return f"Groq request failed for {endpoint}: HTTP {exc.code}" + (
            f". Detail: {detail}" if detail else ""
        )

    def _get(self, endpoint: str) -> JSONDict:
        url = f"{self.base_url}{endpoint}"
        try:
            return self._http_get_json(url, self._headers(), self.timeout_seconds)
        except error.HTTPError as exc:
            raise LLMConnectionError(self._format_http_error(exc, endpoint)) from exc
        except (error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise LLMConnectionError(
                f"Unable to connect to Groq at {self.base_url}. "
                "Verify network access and API key."
            ) from exc

    def _post(self, endpoint: str, payload: JSONDict) -> JSONDict:
        url = f"{self.base_url}{endpoint}"
        try:
            result = self._http_post_json(
                url, payload, self._headers(), self.timeout_seconds
            )
        except error.HTTPError as exc:
            raise LLMProviderError(self._format_http_error(exc, endpoint)) from exc
        except (error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise LLMProviderError(
                f"Groq request failed for {endpoint}: {exc}"
            ) from exc

        if "error" in result:
            raise LLMProviderError(str(result["error"]))
        return result

    def _post_stream(self, endpoint: str, payload: JSONDict) -> Iterator[JSONDict]:
        url = f"{self.base_url}{endpoint}"
        try:
            yield from self._http_post_json_stream(
                url,
                payload,
                self._headers(),
                self.timeout_seconds,
            )
        except error.HTTPError as exc:
            raise LLMProviderError(self._format_http_error(exc, endpoint)) from exc
        except (error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise LLMProviderError(
                f"Groq streaming request failed for {endpoint}: {exc}"
            ) from exc

    def _build_payload(
        self,
        query: str,
        *,
        context_items: Sequence[str | dict[str, object]] | None = None,
        system_prompt: str | None = None,
        stream: bool = False,
    ) -> JSONDict:
        user_prompt = build_user_prompt(query, context_items)
        return {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or self.config.system_prompt,
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream,
        }

    def _extract_text_from_message_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
            return "".join(parts).strip()

        return ""

    def _normalize_response_text(self, text: str) -> str:
        cleaned = text.strip()
        return cleaned if cleaned else EMPTY_RESPONSE_FALLBACK

    def list_models(self) -> list[str]:
        """Return available Groq model names."""

        if self._use_openai_sdk():
            try:
                response = self._client().models.list()
            except Exception as exc:
                raise LLMConnectionError(
                    self._format_openai_error(exc, "/models")
                ) from exc

            model_names: list[str] = []
            for model in getattr(response, "data", []):
                model_id = getattr(model, "id", None)
                if isinstance(model_id, str):
                    model_names.append(model_id)
            return model_names

        response = self._get("/models")
        model_names: list[str] = []
        for model in response.get("data", []):
            if isinstance(model, dict) and isinstance(model.get("id"), str):
                model_names.append(model["id"])
        return model_names

    def check_status(self) -> GroqStatus:
        """Check endpoint reachability and whether configured model is available."""

        model_names = self.list_models()
        return GroqStatus(
            api_reachable=True,
            model_available=self.config.model in model_names,
            available_models=model_names,
        )

    def verify_ready(self) -> GroqStatus:
        """Ensure Groq endpoint is reachable and target model is available."""

        status = self.check_status()
        if status.model_available:
            return status

        raise LLMConnectionError(
            "Configured model is not available on Groq account. "
            f"Requested model: {self.config.model}."
        )

    def generate(
        self,
        query: str,
        *,
        context_items: Sequence[str | dict[str, object]] | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Return a single complete response from Groq."""

        payload = self._build_payload(
            query,
            context_items=context_items,
            system_prompt=system_prompt,
            stream=False,
        )

        if self._use_openai_sdk():
            try:
                result = self._client().chat.completions.create(
                    model=self.config.model,
                    messages=payload["messages"],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
            except TypeError:
                result = self._client().chat.completions.create(
                    model=self.config.model,
                    messages=payload["messages"],
                )
            except Exception as exc:
                raise LLMProviderError(
                    self._format_openai_error(exc, "/chat/completions")
                ) from exc

            choices = getattr(result, "choices", [])
            first_choice = choices[0] if choices else None
            message = getattr(first_choice, "message", None)
            content = getattr(message, "content", "")
            text = self._extract_text_from_message_content(content)

            raw = result.model_dump() if hasattr(result, "model_dump") else {}
            return LLMResponse(
                text=self._normalize_response_text(text),
                model=str(getattr(result, "model", self.config.model)),
                prompt=str(payload["messages"][1]["content"]),
                done_reason=str(getattr(first_choice, "finish_reason", "") or "")
                or None,
                raw=raw,
            )

        result = self._post("/chat/completions", payload)

        choices = result.get("choices", [])
        first_choice = choices[0] if choices else {}
        message = (
            first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
        )
        text = self._extract_text_from_message_content(message.get("content", ""))

        return LLMResponse(
            text=self._normalize_response_text(text),
            model=str(result.get("model", self.config.model)),
            prompt=str(payload["messages"][1]["content"]),
            done_reason=(
                str(first_choice.get("finish_reason"))
                if isinstance(first_choice, dict) and first_choice.get("finish_reason")
                else None
            ),
            raw=result,
        )

    def generate_stream(
        self,
        query: str,
        *,
        context_items: Sequence[str | dict[str, object]] | None = None,
        system_prompt: str | None = None,
    ) -> Iterator[LLMStreamToken]:
        """Yield response chunks token-by-token from Groq streaming API."""

        payload = self._build_payload(
            query,
            context_items=context_items,
            system_prompt=system_prompt,
            stream=True,
        )

        emitted_token = False
        for event in self._post_stream("/chat/completions", payload):
            choices = event.get("choices", [])
            first_choice = choices[0] if choices else {}

            delta = (
                first_choice.get("delta", {}) if isinstance(first_choice, dict) else {}
            )
            token = str(delta.get("content", "")) if isinstance(delta, dict) else ""
            done_reason = (
                first_choice.get("finish_reason")
                if isinstance(first_choice, dict)
                else None
            )
            done = done_reason is not None

            if token:
                emitted_token = True
                yield LLMStreamToken(token=token, done=False)
            elif done and emitted_token:
                yield LLMStreamToken(token="", done=True)

        if not emitted_token:
            # Ensure downstream UI receives at least one visible chunk.
            yield LLMStreamToken(token=EMPTY_RESPONSE_FALLBACK, done=False)
            yield LLMStreamToken(token="", done=True)
