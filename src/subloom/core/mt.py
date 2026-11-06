"""The machine translation branch: DeepL and Google take a batch of texts and answer each one,
with no prompt to build and no reply to parse. MachineTranslator drives them."""

import logging
import os
from typing import Callable, Dict, List, Optional, Protocol, Sequence

import httpx

from ..exceptions import TranslationError
from ..models.config import (
    DEEPL_API_KEY_ENV,
    GOOGLE_API_KEY_ENV,
    AppConfig,
    TranslationConfig,
)
from ..models.degradation import TRANSLATION_INCOMPLETE, Degradation
from ..models.subtitle import SubtitleSegment
from .service import ServiceError, TranslationService, call_with_retries, json_or_raise
from .translator import LANGUAGE_NAMES

logger = logging.getLogger(__name__)


class MtBackend(TranslationService, Protocol):
    """A machine translation service; check_model_available() checks the key, as there is
    no model to choose."""

    max_texts: int  # Texts one request may carry

    def translate_texts(
        self, texts: Sequence[str], source: str, target: str, context: str
    ) -> List[str]:
        """Translate each text; the answer has one entry per text, in order.

        Args:
            texts: The texts to translate.
            source: Source language code, as the pipeline uses them ("zh", "zh-TW", "en").
            target: Target language code, likewise.
            context: Surrounding dialogue that guides but is not translated; a service
                without such a parameter ignores it.

        Raises:
            ServiceError: The service rejected the request.
            httpx.TransportError: The service could not be reached or dropped the connection.
        """
        ...


# Our codes DeepL spells differently. A target is sent with its variant, since older API
# versions reject a bare EN or PT; a source never carries one.
_DEEPL_TARGETS = {"zh": "ZH-HANS", "zh-tw": "ZH-HANT", "en": "EN-US", "pt": "PT-BR"}


class DeepLBackend:
    """DeepL's v2 translate API."""

    label = "DeepL"
    model = ""
    # The current API bounds a request by size only (128 KiB); older versions capped it at 50.
    max_texts = 50

    def __init__(self, api_key: str, settings: TranslationConfig):
        # DeepL's documented convention: Free keys end in ":fx" and have their own host.
        free = api_key.endswith(":fx")
        self.endpoint = "https://api-free.deepl.com" if free else "https://api.deepl.com"
        self._http = httpx.Client(
            base_url=self.endpoint,
            headers={"Authorization": f"DeepL-Auth-Key {api_key}"},
            timeout=httpx.Timeout(settings.request_timeout, connect=10.0),
        )

    def translate_texts(
        self, texts: Sequence[str], source: str, target: str, context: str
    ) -> List[str]:
        body = {
            "text": list(texts),
            "source_lang": source.split("-")[0].upper(),
            "target_lang": _DEEPL_TARGETS.get(target.lower(), target.upper()),
        }
        if context:
            body["context"] = context
        data = self._json(self._http.post("v2/translate", json=body))
        try:
            return [t["text"] for t in data["translations"]]
        except (KeyError, TypeError) as e:
            raise ServiceError(f"Unexpected DeepL reply: {str(data)[:200]}") from e

    def check_connection(self) -> bool:
        return _reachable(self._http, "v2/usage", self.endpoint)

    def list_models(self) -> List[str]:
        return []

    def check_model_available(self) -> bool:
        self._json(self._http.get("v2/usage"))
        return True

    @staticmethod
    def _json(response: httpx.Response):
        if response.status_code == 456:
            raise ServiceError("DeepL character quota exhausted (HTTP 456)", 456)
        return json_or_raise(response)


# Our codes Google spells differently; the rest pass through.
_GOOGLE_CODES = {"zh": "zh-CN", "zh-tw": "zh-TW"}


class GoogleBackend:
    """Google Cloud Translation Basic (v2), authenticated with an API key."""

    label = "Google Cloud Translation"
    model = ""
    max_texts = 128  # The v2 cap on strings per request

    def __init__(self, api_key: str, settings: TranslationConfig):
        self.endpoint = "https://translation.googleapis.com/language/translate/v2"
        # The header, not the documented ?key= parameter, keeps the key out of URLs and logs.
        self._http = httpx.Client(
            base_url="https://translation.googleapis.com/language/translate/",
            headers={"x-goog-api-key": api_key},
            timeout=httpx.Timeout(settings.request_timeout, connect=10.0),
        )

    def translate_texts(
        self, texts: Sequence[str], source: str, target: str, context: str
    ) -> List[str]:
        body = {
            "q": list(texts),
            "source": _GOOGLE_CODES.get(source.lower(), source),
            "target": _GOOGLE_CODES.get(target.lower(), target),
            # The default, html, returns entities: "don't" comes back as "don&#39;t".
            "format": "text",
        }
        data = json_or_raise(self._http.post("v2", json=body))
        try:
            return [t["translatedText"] for t in data["data"]["translations"]]
        except (KeyError, TypeError) as e:
            raise ServiceError(f"Unexpected Google reply: {str(data)[:200]}") from e

    def check_connection(self) -> bool:
        return _reachable(self._http, "v2/languages", self.endpoint)

    def list_models(self) -> List[str]:
        return []

    def check_model_available(self) -> bool:
        json_or_raise(self._http.get("v2/languages"))
        return True


def _reachable(http: httpx.Client, path: str, endpoint: str) -> bool:
    try:
        http.get(path)
    except httpx.TransportError as e:
        logger.error(f"Cannot connect to {endpoint}: {e}")
        return False
    return True


def create_mt_backend(config: AppConfig) -> MtBackend:
    """Build the machine translation service ``config.translation.backend`` names.

    Raises:
        ValueError: No API key is set for it, or it names no machine translation service.
    """
    backend = config.translation.backend
    if backend == "deepl":
        key = os.environ.get(DEEPL_API_KEY_ENV) or config.deepl.api_key
        if not key:
            raise ValueError(
                f"translation.backend deepl needs {DEEPL_API_KEY_ENV} or deepl.api_key"
            )
        return DeepLBackend(key, config.translation)
    if backend == "google":
        key = os.environ.get(GOOGLE_API_KEY_ENV) or config.google.api_key
        if not key:
            raise ValueError(
                f"translation.backend google needs {GOOGLE_API_KEY_ENV} or google.api_key"
            )
        return GoogleBackend(key, config.translation)
    raise ValueError(f"Unknown machine translation backend {backend!r}")


class MachineTranslator:
    """Translates a batch per request and takes the service's answers as they come: no batch
    clamp and no per-segment retry, since an engine answers the same text the same way."""

    LANGUAGE_NAMES: Dict[str, str] = LANGUAGE_NAMES

    def __init__(self, settings: TranslationConfig, backend: MtBackend):
        self.settings = settings
        self.backend = backend
        self._untranslated = 0

    def effective_batch_size(self) -> int:
        return self.backend.max_texts

    def check_connection(self) -> bool:
        return self.backend.check_connection()

    def check_model_available(self) -> bool:
        return self.backend.check_model_available()

    def translate(
        self,
        segments: List[SubtitleSegment],
        source_lang: str,
        target_lang: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        context_window: int = 3,
    ) -> List[SubtitleSegment]:
        """Translate every segment; same arguments as SubtitleTranslator.translate.

        Raises:
            TranslationError: The service refused (a bad key, an exhausted quota) or kept
                failing, or answered a different number of texts than it was sent.
        """
        if source_lang == target_lang:
            logger.warning("Source and target languages are the same, skipping translation")
            return segments
        self._untranslated = 0
        size = self.backend.max_texts
        translated: List[SubtitleSegment] = []
        for start in range(0, len(segments), size):
            batch = segments[start : start + size]
            # Each text is translated on its own, so the whole stretch of dialogue around it,
            # this batch included, is the context.
            nearby = segments[max(0, start - context_window) : start + size + context_window]
            context = "\n".join(seg.text for seg in nearby)
            texts = [seg.text for seg in batch]
            answers = call_with_retries(
                lambda: self.backend.translate_texts(texts, source_lang, target_lang, context),
                self.settings.max_retries,
            )
            if len(answers) != len(batch):
                raise TranslationError(
                    f"{self.backend.label} answered {len(answers)} texts for {len(batch)} sent"
                )
            for seg, answer in zip(batch, answers):
                text = answer.strip()
                if not text:
                    self._untranslated += 1
                translated.append(
                    SubtitleSegment(
                        index=seg.index, start=seg.start, end=seg.end, text=text or seg.text
                    )
                )
            if progress_callback:
                progress_callback(len(translated), len(segments))
        logger.info(f"Translation complete: {len(translated)} segments ({self.backend.label})")
        return translated

    def names_degradation(self) -> Optional[Degradation]:
        """Machine translation takes no name list, so it never runs a name pass."""
        return None

    def untranslated_degradation(self, target_lang: str, total: int) -> Optional[Degradation]:
        """What the last translate() call left untranslated: texts the service answered empty."""
        if not self._untranslated:
            return None
        return Degradation(
            TRANSLATION_INCOMPLETE,
            f"{target_lang}: {self._untranslated} of {total} segments left in the source language",
        )
