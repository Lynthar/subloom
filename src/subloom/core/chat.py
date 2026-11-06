"""The LLM chat backends the LLM translator sends its prompts through, and their factory."""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import httpx
from ollama import Client, ResponseError

from ..models.config import (
    OPENAI_API_KEY_ENV,
    AppConfig,
    OllamaConfig,
    OpenAIConfig,
    TranslationConfig,
)
from .model_manager import DownloadProgress, OllamaModelManager
from .service import ServiceError, TranslationService, json_or_raise

logger = logging.getLogger(__name__)


class ChatBackend(TranslationService, Protocol):
    """A service that answers a prompt; check_model_available() looks ``model`` up in
    list_models(). ``last_prompt_tokens`` is how many tokens of the last prompt the server says
    it read, None when it did not say."""

    last_prompt_tokens: Optional[int]

    def chat(
        self,
        prompt: str,
        *,
        json_keys: Optional[Sequence[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Send ``prompt`` as one user message and return the reply text.

        Args:
            prompt: The whole prompt.
            json_keys: For a prompt that asks for ``{"translations": {key: text}}``, the keys
                it lists; the backend constrains the reply to that shape where it can.
            max_tokens: The most tokens the model may generate, thinking included, where the
                backend can enforce it. A reply cut there comes back truncated, not as an error.
            temperature: Sampling temperature for this request; None means translation.temperature.

        Raises:
            ServiceError: The server rejected the request.
            httpx.TransportError: The server could not be reached or dropped the connection.
        """
        ...


class OllamaChat:
    """Ollama's native chat API."""

    label = "Ollama"

    def __init__(self, cfg: OllamaConfig, settings: TranslationConfig):
        self.model = cfg.model
        self.endpoint = cfg.host
        self._temperature = settings.temperature
        self._timeout = settings.request_timeout
        self._client: Optional[Client] = None
        self.model_manager = OllamaModelManager(host=cfg.host)
        self.last_prompt_tokens: Optional[int] = None

    @property
    def client(self) -> Client:
        if self._client is None:
            timeout = httpx.Timeout(self._timeout, connect=10.0)
            self._client = Client(host=self.endpoint, timeout=timeout)
        return self._client

    def chat(
        self,
        prompt: str,
        *,
        json_keys: Optional[Sequence[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        options: Dict[str, Any] = {
            "temperature": self._temperature if temperature is None else temperature
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "options": options,
        }
        if json_keys is not None:
            kwargs["format"] = "json"
        try:
            response = self.client.chat(**kwargs)
        except ResponseError as e:
            raise ServiceError(str(e), e.status_code) from e
        self.last_prompt_tokens = response.get("prompt_eval_count")
        if response.get("done_reason") == "length":
            logger.warning(
                f"{self.model} hit the {max_tokens}-token cap (thinking included) "
                f"before finishing its reply"
            )
        return str(response["message"]["content"] or "")

    def check_connection(self) -> bool:
        return self.model_manager.check_connection()

    def list_models(self) -> List[str]:
        try:
            return self.model_manager.list_models()
        except ResponseError as e:
            raise ServiceError(str(e), e.status_code) from e

    def check_model_available(self) -> bool:
        try:
            return self.model_manager.is_model_available(self.model)
        except ResponseError as e:
            raise ServiceError(str(e), e.status_code) from e

    def ensure_model_ready(
        self,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        auto_pull: bool = True,
    ) -> bool:
        """Download the model if missing (when ``auto_pull``); True when it is ready."""
        return self.model_manager.ensure_model(
            self.model, progress_callback=progress_callback, auto_pull=auto_pull
        )


# Reply constraints in the order tried. A server that rejects one is asked with the next for
# the rest of the run: LM Studio takes only json_schema, DeepSeek only json_object.
_RESPONSE_FORMATS = ("json_schema", "json_object", None)


class OpenAIChat:
    """An OpenAI-compatible ``/chat/completions`` endpoint, local or cloud."""

    label = "OpenAI-compatible server"

    def __init__(self, cfg: OpenAIConfig, settings: TranslationConfig, api_key: Optional[str]):
        self.model = cfg.model
        self.endpoint = cfg.base_url
        self._temperature = settings.temperature
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(
            base_url=cfg.base_url,
            headers=headers,
            timeout=httpx.Timeout(settings.request_timeout, connect=10.0),
        )
        self._format_index = 0
        self.last_prompt_tokens: Optional[int] = None

    def chat(
        self,
        prompt: str,
        *,
        json_keys: Optional[Sequence[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        # max_tokens is not sent: OpenAI's reasoning models reject that name with a 400, and
        # not every compatible server knows max_completion_tokens.
        while True:
            kind = _RESPONSE_FORMATS[self._format_index] if json_keys is not None else None
            body: Dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self._temperature if temperature is None else temperature,
            }
            if kind is not None and json_keys is not None:
                body["response_format"] = _response_format(kind, json_keys)
            response = self._http.post("chat/completions", json=body)
            # Only a complaint that names the parameter: a 400 for a missing "json" in the
            # prompt or a bad model must surface, not silently drop the constraint.
            if (
                kind is not None
                and response.status_code in (400, 422)
                and "response_format" in response.text
            ):
                self._format_index += 1
                logger.warning(
                    f"{self.endpoint} does not accept response_format {kind}; "
                    f"trying {_RESPONSE_FORMATS[self._format_index] or 'none'}"
                )
                continue
            data = json_or_raise(response)
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                raise ServiceError(f"Unexpected reply shape: {str(data)[:200]}") from e
            usage = data.get("usage")
            self.last_prompt_tokens = (
                usage.get("prompt_tokens") if isinstance(usage, dict) else None
            )
            return content or ""

    def check_connection(self) -> bool:
        try:
            self._http.get("models")
        except httpx.TransportError as e:
            logger.error(f"Cannot connect to {self.endpoint}: {e}")
            return False
        return True

    def list_models(self) -> List[str]:
        data = json_or_raise(self._http.get("models"))
        try:
            return [m["id"] for m in data["data"]]
        except (KeyError, TypeError) as e:
            raise ServiceError(f"Unexpected model list: {str(data)[:200]}") from e

    def check_model_available(self) -> bool:
        return self.model in self.list_models()


def _response_format(kind: str, keys: Sequence[str]) -> Dict[str, Any]:
    if kind == "json_object":
        return {"type": "json_object"}
    # Every key required: a schema-constrained model cannot skip a segment.
    translations = {
        "type": "object",
        "properties": {key: {"type": "string"} for key in keys},
        "required": list(keys),
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {"translations": translations},
        "required": ["translations"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {"name": "subtitle_translations", "strict": True, "schema": schema},
    }


def create_chat_backend(config: AppConfig) -> ChatBackend:
    """Build the backend ``config.translation.backend`` names, from its config section.

    Args:
        config: The application config.

    Raises:
        ValueError: translation.backend names no LLM backend.
    """
    backend = config.translation.backend
    if backend == "ollama":
        return OllamaChat(config.ollama, config.translation)
    if backend == "openai":
        api_key = os.environ.get(OPENAI_API_KEY_ENV) or config.openai.api_key
        return OpenAIChat(config.openai, config.translation, api_key)
    raise ValueError(f"Unknown translation backend {backend!r}")
