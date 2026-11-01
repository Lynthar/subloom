"""Subtitle translation: the LLM branch (SubtitleTranslator), and create_translator, which
picks it or the machine translation branch in core.mt."""

from typing import Callable, Dict, List, Optional, Protocol
import json
import re
import logging

import httpx

from ..exceptions import TranslationError
from ..models.config import MT_BACKENDS, AppConfig, TranslationConfig
from ..models.degradation import TRANSLATION_INCOMPLETE, Degradation
from ..models.subtitle import SubtitleSegment
from .chat import ChatBackend, create_chat_backend
from .service import ServiceError, TranslationService, call_with_retries

logger = logging.getLogger(__name__)

# Six or more copies in a row of a unit of up to 20 non-digit characters ("あ" x400 from ASR):
# sent whole, the reply repeats one token past Ollama's abort limit (over 100 in a row) and the
# batch fails. Digits never form a unit, so "1000000" stays a number.
_REPEAT_RUN = re.compile(r"(\D{1,20}?)\1{5,}")
MAX_REPEATS = 5

# Generation cap per request, thinking included (Ollama counts it), so a model looping on its own
# output stops here instead of at request_timeout on every retry. qwen3.8:27b used at most 1.6k
# tokens a request on real subtitles, thinking and all; the base leaves room for wordier thinkers.
REPLY_TOKENS_BASE = 4096
REPLY_TOKENS_PER_SEGMENT = 256


def squeeze_repeats(text: str) -> str:
    """``text`` with every run of more than MAX_REPEATS copies of a unit cut to MAX_REPEATS."""
    return _REPEAT_RUN.sub(lambda m: m.group(1) * MAX_REPEATS, text)


def reply_token_cap(segments: int) -> int:
    """The generation cap for a request translating ``segments`` segments."""
    return REPLY_TOKENS_BASE + REPLY_TOKENS_PER_SEGMENT * segments


def _failed_on_content(error: TranslationError) -> bool:
    """Whether a batch's exhausted retries point at what it asked (a 5xx, a reply that never
    finished) rather than at the service, so its segments are worth sending one by one."""
    cause = error.__cause__
    if isinstance(cause, httpx.ReadTimeout):
        return True
    return isinstance(cause, ServiceError) and (cause.status_code or 0) >= 500


# Names the prompts and progress messages use; any other code is passed as is.
LANGUAGE_NAMES: Dict[str, str] = {
    "zh": "Simplified Chinese",
    "zh-TW": "Traditional Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
    "ar": "Arabic",
    "vi": "Vietnamese",
    "th": "Thai",
}


class SubtitleTranslator:
    """Subtitle translator over the chat backend translation.backend selects."""

    # Default JSON prompt template, used when no library template is picked — the backend
    # constrains the reply to JSON for reliable parsing. A selected template falls back to the
    # legacy [N]-line text path (see _is_json_mode / _parse_response).
    DEFAULT_PROMPT_TEMPLATE = """You are a professional subtitle translator. Translate dialogue from {source_lang} into {target_lang}.

GUIDELINES
- Preserve natural conversational tone, emotional nuance, and register.
- Keep translations concise for subtitle readability.
- For very short utterances (interjections, "uh", "嗯", "ah"), still translate them — do NOT drop them.
- Treat the context blocks as background only; never include them in your output.
{context_before}
SUBTITLES TO TRANSLATE — every key in the JSON below MUST appear in your output:
{segments}
{context_after}
OUTPUT — return STRICTLY a JSON object of the form:
{{"translations": {{"<index>": "<translated text>", ...}}}}

Do not include any text outside the JSON object. Do not omit any indices. Do not renumber.
"""

    LANGUAGE_NAMES = LANGUAGE_NAMES

    # Approximate VRAM requirements for Ollama models (MB)
    OLLAMA_MODEL_VRAM = {
        "qwen2.5:7b": 5000,  # ~5GB
        "qwen2.5:14b": 9000,  # ~9GB
        "qwen2.5:32b": 20000,  # ~20GB
        "qwen2.5:72b": 45000,  # ~45GB
    }

    @classmethod
    def select_optimal_model(cls, prefer_quality: bool = True) -> str:
        """
        Select optimal translation model based on available VRAM.

        Args:
            prefer_quality: Prefer larger/higher quality models if VRAM allows.

        Returns:
            Recommended model name.
        """
        from ..utils.gpu import get_available_vram

        available_vram = get_available_vram()

        if available_vram <= 0:
            logger.warning("Cannot detect GPU VRAM, using default model")
            return "qwen2.5:14b"

        # Sort models by VRAM requirement (larger first if prefer_quality)
        sorted_models = sorted(
            cls.OLLAMA_MODEL_VRAM.items(),
            key=lambda x: x[1],
            reverse=prefer_quality,
        )

        for model_name, required_vram in sorted_models:
            if available_vram >= required_vram * 1.1:  # 10% headroom
                logger.info(
                    f"Selected translation model: {model_name} "
                    f"(requires {required_vram}MB, available {available_vram}MB)"
                )
                return model_name

        return "qwen2.5:7b"  # Fallback to smallest model

    def __init__(
        self,
        settings: TranslationConfig,
        backend: ChatBackend,
        *,
        save_failed_log: bool = False,
        failed_log_path: Optional[str] = None,
    ):
        self.settings = settings
        self.backend = backend
        self.save_failed_log = save_failed_log
        self.failed_log_path = failed_log_path
        self._failed_translations: List[dict] = []  # Track failed translations
        self._batch_failure_indices: dict = {}  # Track failed indices per reason for compressed logging

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        save_failed_log: bool = False,
        failed_log_path: Optional[str] = None,
    ) -> "SubtitleTranslator":
        """Build a translator over the chat backend ``config.translation.backend`` names.

        Args:
            config: The application config.
            save_failed_log: Write translation failures to ``failed_log_path`` (or a
                timestamped file in the working directory).
            failed_log_path: Where that log goes.

        Raises:
            ValueError: translation.backend names no known backend.
        """
        return cls(
            config.translation,
            create_chat_backend(config),
            save_failed_log=save_failed_log,
            failed_log_path=failed_log_path,
        )

    def _is_json_mode(self) -> bool:
        """
        Whether to drive the LLM in JSON mode + parse JSON output.

        We only enable JSON mode for the default prompt — library templates
        were authored against the legacy [N]-line format and would misbehave
        under a JSON constraint. Decided by the template actually in effect,
        so a dangling library id that falls back to the default prompt stays
        in JSON mode.
        """
        return self._selected_template() is None

    def _effective_batch_size(self) -> int:
        """
        Cap batch size based on model capability.

        Larger batches degrade translation reliability for smaller models
        (the LLM "forgets" segments mid-batch). The parameter count is read off
        the model name ("qwen2.5:14b", "Qwen2.5-14B-Instruct").
        """
        cfg_max = max(1, self.settings.max_batch_size)
        match = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)b(?![a-z0-9])", self.backend.model.lower())
        if match is None:
            # A sizeless name on an OpenAI-compatible server is usually a large cloud model;
            # on Ollama it is a default tag, typically 8B or smaller.
            return cfg_max if self.settings.backend == "openai" else min(cfg_max, 4)
        size = float(match.group(1))
        if size >= 32:
            return cfg_max
        if size >= 13:
            return min(cfg_max, 6)
        return min(cfg_max, 4)

    def effective_batch_size(self) -> int:
        """Public accessor for the model-clamped batch size (for progress UI)."""
        return self._effective_batch_size()

    def check_connection(self) -> bool:
        """Whether the backend's server answers."""
        return self.backend.check_connection()

    def check_model_available(self) -> bool:
        """Whether the server offers the model; raises when it cannot be asked."""
        return self.backend.check_model_available()

    def get_prompt_template(self) -> str:
        """The selected library template, or the default when none is (or it is gone)."""
        return effective_prompt_template(self.settings.prompt_template_id)

    def _selected_template(self) -> Optional[str]:
        """The library template body in effect; None means the default applies."""
        return library_template(self.settings.prompt_template_id)

    def _build_translation_prompt(
        self,
        segments: List[SubtitleSegment],
        source_lang: str,
        target_lang: str,
        context_before: Optional[List[SubtitleSegment]] = None,
        context_after: Optional[List[SubtitleSegment]] = None,
    ) -> str:
        """
        Build translation prompt with context for natural translation.

        In JSON mode (default), context is rendered as prose so the LLM cannot
        confuse it with the indexed segments-to-translate. In legacy mode the
        original [N] format is preserved for backward compatibility with
        prompt-library templates.
        """
        source_name = self.LANGUAGE_NAMES.get(source_lang, source_lang)
        target_name = self.LANGUAGE_NAMES.get(target_lang, target_lang)
        json_mode = self._is_json_mode()

        if json_mode:
            # Prose context — visually distinct from the JSON-shaped segments
            # block, so the LLM can't accidentally include context indices in
            # its output.
            context_before_text = ""
            if context_before:
                snippet = " ".join(squeeze_repeats(seg.text.strip()) for seg in context_before[-3:])
                context_before_text = (
                    f'\nEARLIER DIALOGUE (background only, do NOT include in output): "{snippet}"\n'
                )

            context_after_text = ""
            if context_after:
                snippet = " ".join(squeeze_repeats(seg.text.strip()) for seg in context_after[:2])
                context_after_text = (
                    f'\nLATER DIALOGUE (background only, do NOT include in output): "{snippet}"\n'
                )

            # Segments rendered as a JSON object — primes the model to
            # respond with a JSON object of the same shape.
            seg_obj = {str(seg.index): squeeze_repeats(seg.text) for seg in segments}
            segments_text = json.dumps(seg_obj, ensure_ascii=False, indent=2)
        else:
            context_before_text = ""
            if context_before:
                context_lines = [
                    f"  [{seg.index}] {squeeze_repeats(seg.text)}" for seg in context_before[-3:]
                ]
                context_before_text = f"""
PREVIOUS DIALOGUE (for context, DO NOT translate):
{chr(10).join(context_lines)}"""

            context_after_text = ""
            if context_after:
                context_lines = [
                    f"  [{seg.index}] {squeeze_repeats(seg.text)}" for seg in context_after[:2]
                ]
                context_after_text = f"""
FOLLOWING DIALOGUE (for context, DO NOT translate):
{chr(10).join(context_lines)}"""

            lines = [f"[{seg.index}] {squeeze_repeats(seg.text)}" for seg in segments]
            segments_text = chr(10).join(lines)

        template = self.get_prompt_template()

        return template.format(
            source_lang=source_name,
            target_lang=target_name,
            context_before=context_before_text,
            segments=segments_text,
            context_after=context_after_text,
        )

    def _parse_translation_response(
        self,
        response: str,
        original_segments: List[SubtitleSegment],
    ) -> List[SubtitleSegment]:
        """Parse translation response. Tries JSON first, falls back to regex."""
        translated = []

        # Log raw response for debugging (truncated)
        response_preview = response[:500] + "..." if len(response) > 500 else response
        logger.debug(f"Raw LLM response:\n{response_preview}")

        index_to_translation: dict = {}
        # Each fallback runs until every requested index is present. Counting entries is
        # not enough: a reply renumbered from 1 has the right count and none of our indices.
        expected = {seg.index for seg in original_segments}

        # Strategy 1: JSON parsing (the new default path).
        # The LLM may wrap the JSON in markdown fences or add a stray prefix
        # despite a JSON constraint; pull out the first {...} block and parse it.
        if not index_to_translation:
            json_blob = self._extract_json_object(response)
            if json_blob is not None:
                try:
                    parsed = json.loads(json_blob)
                    translations = parsed.get("translations") if isinstance(parsed, dict) else None
                    if isinstance(translations, dict):
                        for key, value in translations.items():
                            try:
                                idx = int(key)
                            except (TypeError, ValueError):
                                continue
                            if isinstance(value, str) and value.strip():
                                index_to_translation[idx] = value.strip()
                    elif isinstance(parsed, dict):
                        # Some models return {"5": "...", "6": "..."} directly
                        for key, value in parsed.items():
                            try:
                                idx = int(key)
                            except (TypeError, ValueError):
                                continue
                            if isinstance(value, str) and value.strip():
                                index_to_translation[idx] = value.strip()
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON parse failed, falling back to regex: {e}")

        # Strategy 2: [N] prefixed lines (legacy format, also catches stray
        # markdown like **[1]** text)
        if not expected.issubset(index_to_translation):
            pattern1 = r"\[(\d+)\]\s*[:：]?\s*(.+?)(?=\[\d+\]|$)"
            matches = re.findall(pattern1, response, re.DOTALL)
            for idx_str, text in matches:
                idx = int(idx_str)
                text = text.strip()
                if text and idx not in index_to_translation:
                    index_to_translation[idx] = text

        # Strategy 3: bare numbered lines — "1.", "1:", "1)", "(1)"
        if not expected.issubset(index_to_translation):
            pattern2 = r"(?:^|\n)\s*[\(\[]?(\d+)[\]\)]?\s*[.:：\)]\s*(.+?)(?=\n\s*[\(\[]?\d+[\]\)]?\s*[.:：\)]|$)"
            matches = re.findall(pattern2, response, re.DOTALL)
            for idx_str, text in matches:
                idx = int(idx_str)
                text = text.strip()
                if text and idx not in index_to_translation:
                    index_to_translation[idx] = text

        # Strategy 4: positional fallback — same number of non-empty lines
        if not expected.issubset(index_to_translation):
            lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
            lines = [
                line
                for line in lines
                if not line.startswith(("#", "-", "*", "翻译", "Translation"))
            ]

            if len(lines) == len(original_segments):
                for seg, line in zip(original_segments, lines):
                    if seg.index not in index_to_translation:
                        # Strip a leading index marker, but ONLY unambiguous forms: [3], (3), 3. /
                        # 3: / 3). A bare number plus space or a pure number line is legitimate
                        # content.
                        cleaned = re.sub(
                            r"^(?:[\[\(]\d+[\]\)][.:：]?|\d+[.:：)])\s*", "", line
                        ).strip()
                        if cleaned:
                            index_to_translation[seg.index] = cleaned

        # Build translated segments
        for seg in original_segments:
            if seg.index in index_to_translation:
                trans_text = index_to_translation[seg.index]
                # Clean up common issues
                trans_text = self._clean_translation(trans_text, seg.text)
                translated.append(
                    SubtitleSegment(
                        index=seg.index,
                        start=seg.start,
                        end=seg.end,
                        text=trans_text,
                    )
                )
            else:
                # Detailed failure analysis
                failure_reason = self._analyze_translation_failure(
                    seg, response, index_to_translation
                )

                # Track failed indices by reason (for compressed batch logging)
                if failure_reason not in self._batch_failure_indices:
                    self._batch_failure_indices[failure_reason] = []
                self._batch_failure_indices[failure_reason].append(seg.index)

                # Track failed translation for later analysis/log file
                self._failed_translations.append(
                    {
                        "index": seg.index,
                        "original": seg.text,
                        "reason": failure_reason,
                        "response_snippet": response[:200] if response else "(empty)",
                    }
                )

                translated.append(seg)

        return translated

    def _analyze_translation_failure(
        self,
        segment: SubtitleSegment,
        response: str,
        found_translations: dict,
    ) -> str:
        """Analyze why a translation failed and return a descriptive reason."""
        original_text = segment.text

        # Check for common refusal patterns
        refusal_patterns = [
            ("I cannot", "model_refusal"),
            ("I can't", "model_refusal"),
            ("I'm sorry", "model_refusal"),
            ("不能翻译", "model_refusal"),
            ("无法翻译", "model_refusal"),
            ("不适合", "content_filter"),
            ("inappropriate", "content_filter"),
            ("sensitive", "content_filter"),
            ("违规", "content_filter"),
        ]

        response_lower = response.lower()
        for pattern, reason_type in refusal_patterns:
            if pattern.lower() in response_lower:
                if reason_type == "model_refusal":
                    return "模型拒绝翻译"
                else:
                    return "内容被过滤"

        # Check if response is empty or too short
        if not response.strip():
            return "LLM返回空响应"

        if len(response.strip()) < 10:
            return "LLM响应过短"

        # Check if the segment index pattern is missing
        if f"[{segment.index}]" not in response and str(segment.index) not in response:
            return "响应中未找到段落索引"

        # Check if original text might be problematic
        if len(original_text) < 3:
            return "原文过短"

        # Check for interjections that models often skip
        interjection_patterns = ["あぁ", "うん", "ああ", "えっ", "んん", "はぁ"]
        if any(p in original_text for p in interjection_patterns):
            return "可能是语气词被跳过"

        # Default - parsing issue
        return "解析失败"

    def _clean_translation(self, translated: str, original: str) -> str:
        """Clean up translation text."""
        translated = translated.strip()

        # If translation is suspiciously short compared to original, might be an error
        if len(translated) < 2 and len(original) > 10:
            return original

        # Strip a leftover index marker only in its bracketed form ([5], (5)): the old pattern also
        # matched bare digits and corrupted translations beginning with a number. JSON mode skips
        # this.
        if not self._is_json_mode():
            translated = re.sub(r"^[\[\(]\d+[\]\)]\s*", "", translated)

        return translated.strip()

    @staticmethod
    def _extract_json_object(response: str) -> Optional[str]:
        """
        Extract the first balanced JSON object from a response string.

        Handles markdown fences (```json ... ```) and stray prefixes like
        "Here is the translation:" that some models emit despite a JSON constraint.
        Returns None if no balanced object is found.
        """
        if not response:
            return None

        # Strip markdown code fences if present
        text = response.strip()
        if text.startswith("```"):
            # Drop the opening fence (and optional language tag)
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1 :]
            # Drop trailing fence
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]

        # Walk the string finding a balanced { ... } block, respecting strings
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def translate_batch(
        self,
        segments: List[SubtitleSegment],
        source_lang: str,
        target_lang: str,
        context_before: Optional[List[SubtitleSegment]] = None,
        context_after: Optional[List[SubtitleSegment]] = None,
    ) -> List[SubtitleSegment]:
        """
        Translate a batch of subtitles with context.

        Args:
            segments: Subtitle segments to translate.
            source_lang: Source language code.
            target_lang: Target language code.
            context_before: Previous segments for context.
            context_after: Following segments for context.

        Returns:
            Translated subtitle segments.

        Raises:
            TranslationError: See translate().
        """
        if not segments:
            return []

        # Reset batch failure indices for this batch
        self._batch_failure_indices = {}

        prompt = self._build_translation_prompt(
            segments,
            source_lang,
            target_lang,
            context_before=context_before,
            context_after=context_after,
        )

        # The keys the default prompt lists, so the backend can constrain the reply to them.
        json_keys = [str(seg.index) for seg in segments] if self._is_json_mode() else None
        cap = reply_token_cap(len(segments))

        try:
            reply = call_with_retries(
                lambda: self.backend.chat(prompt, json_keys=json_keys, max_tokens=cap),
                self.settings.max_retries,
            )
        except TranslationError as e:
            if not _failed_on_content(e):
                raise
            return self._translate_one_by_one(segments, source_lang, target_lang, e)

        translated = self._parse_translation_response(reply, segments)

        # Compressed per-batch failure summary at INFO, not WARNING: the individual retry
        # that runs immediately after usually fixes these. The final WARNING, if any, comes
        # from translate() once all batches have settled.
        if self._batch_failure_indices:
            summary_parts = []
            for reason, indices in self._batch_failure_indices.items():
                if len(indices) <= 5:
                    idx_str = ", ".join(str(i) for i in indices)
                else:
                    # Show first 3 and last 2 with ellipsis
                    idx_str = (
                        f"{indices[0]}, {indices[1]}, {indices[2]}...{indices[-2]}, {indices[-1]}"
                    )
                summary_parts.append(f"{reason} [{idx_str}]")
            logger.info(f"Batch parse issues, retrying individually: {'; '.join(summary_parts)}")

        # Failure detection: the translated text is identical to the original AND source !=
        # target (same language makes identical text a legitimate outcome).
        failed_segments = [
            (i, seg)
            for i, seg in enumerate(translated)
            if seg.text == segments[i].text and source_lang != target_lang
        ]

        # Retry ANY failure individually — the old "only if >50% failed" threshold left
        # small batches with one or two untranslated segments unfixed, the most common
        # failure in practice.
        if failed_segments:
            logger.debug(f"Batch had {len(failed_segments)} failure(s), retrying individually")
            translated = self._retry_failed_individually(
                translated, failed_segments, source_lang, target_lang
            )

        return translated

    def _translate_one_by_one(
        self,
        segments: List[SubtitleSegment],
        source_lang: str,
        target_lang: str,
        error: TranslationError,
    ) -> List[SubtitleSegment]:
        """Send a failed batch's segments one at a time; those still failing keep the source.

        Raises:
            TranslationError: ``error``, when not one segment got through either: then the
                service is failing, not this batch, and every further batch would do the same.
        """
        logger.warning(
            f"Batch {segments[0].index}-{segments[-1].index} failed ({error.__cause__}); "
            f"translating its {len(segments)} segments one by one"
        )
        for seg in segments:
            self._failed_translations.append(
                {
                    "index": seg.index,
                    "original": seg.text,
                    "reason": "批次请求失败",
                    "response_snippet": str(error.__cause__)[:200],
                }
            )
        translated = self._retry_failed_individually(
            list(segments), list(enumerate(segments)), source_lang, target_lang
        )
        batch = {seg.index for seg in segments}
        still_failing = [f for f in self._failed_translations if f["index"] in batch]
        if len(still_failing) == len(segments):
            raise error
        return translated

    def _retry_failed_individually(
        self,
        translated: List[SubtitleSegment],
        failed: List[tuple],
        source_lang: str,
        target_lang: str,
    ) -> List[SubtitleSegment]:
        """Retry translating failed segments one by one."""
        result = list(translated)
        source_name = self.LANGUAGE_NAMES.get(source_lang, source_lang)
        target_name = self.LANGUAGE_NAMES.get(target_lang, target_lang)

        for idx, seg in failed:
            try:
                # Simple single-segment prompt
                simple_prompt = (
                    f"Translate this subtitle from {source_name} to {target_name}. "
                    f"Output ONLY the translation, nothing else:\n\n{squeeze_repeats(seg.text)}"
                )

                reply = self.backend.chat(simple_prompt, max_tokens=reply_token_cap(1)).strip()
                trans_text = self._clean_translation(reply, seg.text)

                # An answer equal to the source (a name, a kanji both languages share) is a
                # translation; _clean_translation handing back the source for a stub is not.
                if trans_text and (trans_text != seg.text or reply == seg.text):
                    result[idx] = SubtitleSegment(
                        index=seg.index,
                        start=seg.start,
                        end=seg.end,
                        text=trans_text,
                    )
                    # The retry fixed it — drop the stale record, or the final
                    # warning and translation_failures.json report a failure
                    # that no longer exists. Indices are unique file-wide.
                    self._failed_translations = [
                        f for f in self._failed_translations if f["index"] != seg.index
                    ]
                    logger.debug(f"Successfully retried segment {seg.index}")

            except Exception as e:
                logger.debug(f"Individual retry failed for segment {seg.index}: {e}")
                # Keep original on failure

        return result

    def translate(
        self,
        segments: List[SubtitleSegment],
        source_lang: str,
        target_lang: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        context_window: int = 3,
    ) -> List[SubtitleSegment]:
        """
        Translate all subtitles with context awareness.

        Args:
            segments: Subtitle segments to translate.
            source_lang: Source language code.
            target_lang: Target language code.
            progress_callback: Progress callback function (completed, total).
            context_window: Number of surrounding segments to include as context.

        Returns:
            Translated subtitle segments.

        Raises:
            TranslationError: The service refused a request (a 4xx other than 429, so an
                exhausted quota ends the run) or kept failing past max_retries. The one
                exception is a batch lost to a 5xx or a read timeout: its segments are sent
                one by one, and those still failing keep the source text and are counted by
                untranslated_degradation. It raises only if none of them got through.
            ConnectionError: Ollama could not be reached (its client raises this, unretried).
        """
        if source_lang == target_lang:
            logger.warning("Source and target languages are the same, skipping translation")
            return segments

        # Reset failure tracking per call: one SubtitleTranslator is reused across every target
        # language in a `process` run, so without this the second language's failures file and "N
        # failed" warning inherit the first language's records.
        self.clear_failed_translations()
        self._batch_failure_indices = {}

        batch_size = self._effective_batch_size()
        if batch_size != self.settings.max_batch_size:
            logger.info(
                f"Auto-clamped translation batch size: {self.settings.max_batch_size} -> "
                f"{batch_size} (model: {self.backend.model})"
            )

        logger.info(
            f"Starting translation: {len(segments)} segments ({source_lang} -> {target_lang}), "
            f"batch_size={batch_size}"
        )

        translated_segments = []

        for i in range(0, len(segments), batch_size):
            batch = segments[i : i + batch_size]

            # Get context from surrounding segments
            context_before = segments[max(0, i - context_window) : i]
            context_after = segments[i + batch_size : i + batch_size + context_window]

            translated_batch = self.translate_batch(
                batch,
                source_lang,
                target_lang,
                context_before=context_before,
                context_after=context_after,
            )
            translated_segments.extend(translated_batch)

            if progress_callback:
                progress_callback(len(translated_segments), len(segments))

        # The count untranslated_degradation reports: an answer identical to the source (a name,
        # a number) survived its own retry, so it is a translation, not a failure.
        failed_count = len(self._failed_translations)
        if failed_count > 0:
            logger.warning(
                f"Translation complete: {len(translated_segments)} segments, "
                f"{failed_count} left untranslated ({failed_count * 100 // len(segments)}%)"
            )

            # Save failed translations log if enabled
            if self.save_failed_log:
                self._save_failed_log()
        else:
            logger.info(f"Translation complete: {len(translated_segments)} segments")

        return translated_segments

    def _save_failed_log(self) -> None:
        """Save failed translations to a log file for analysis."""
        import json
        from pathlib import Path
        from datetime import datetime

        # Determine log path
        if self.failed_log_path:
            log_path = Path(self.failed_log_path)
        else:
            log_path = Path.cwd() / f"translation_failures_{datetime.now():%Y%m%d_%H%M%S}.json"

        try:
            # Categorize failures
            categorized: Dict[str, List[dict]] = {
                "model_refusal": [],
                "content_filter": [],
                "parsing_error": [],
                "other": [],
            }

            for failure in self._failed_translations:
                reason = failure.get("reason", "")
                if "拒绝" in reason:
                    categorized["model_refusal"].append(failure)
                elif "过滤" in reason or "content" in reason.lower():
                    categorized["content_filter"].append(failure)
                elif "解析" in reason or "parsing" in reason.lower():
                    categorized["parsing_error"].append(failure)
                else:
                    categorized["other"].append(failure)

            report = {
                "timestamp": datetime.now().isoformat(),
                "model": self.backend.model,
                "total_failures": len(self._failed_translations),
                "summary": {
                    "model_refusal": len(categorized["model_refusal"]),
                    "content_filter": len(categorized["content_filter"]),
                    "parsing_error": len(categorized["parsing_error"]),
                    "other": len(categorized["other"]),
                },
                "failures": self._failed_translations,
            }

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            logger.info(f"Failed translations log saved: {log_path}")

        except Exception as e:
            logger.error(f"Failed to save translation log: {e}")

    def get_failed_translations(self) -> List[dict]:
        """Get list of failed translations for external analysis."""
        return self._failed_translations.copy()

    def clear_failed_translations(self) -> None:
        """Clear the failed translations list."""
        self._failed_translations = []

    def untranslated_degradation(self, target_lang: str, total: int) -> Optional[Degradation]:
        """What the last translate() call left untranslated, or None when nothing.

        Counts segments the model never answered; an identical answer (a name) is a translation."""
        missing = len(self._failed_translations)
        if not missing:
            return None
        return Degradation(
            TRANSLATION_INCOMPLETE,
            f"{target_lang}: {missing} of {total} segments left in the source language",
        )


def library_template(template_id: Optional[str]) -> Optional[str]:
    """The body of the library template ``template_id`` names; None when unset or gone."""
    if not template_id:
        return None
    from .prompt_library import get_prompt_library

    template = get_prompt_library().get_template(template_id)
    return template.template if template else None


def effective_prompt_template(template_id: Optional[str]) -> str:
    """The prompt the LLM branch runs: the selected library template, else the default."""
    selected = library_template(template_id)
    if selected is not None:
        return selected
    if template_id:
        logger.warning(f"Prompt template '{template_id}' not found, using default")
    return SubtitleTranslator.DEFAULT_PROMPT_TEMPLATE


class Translator(Protocol):
    """What the pipeline and the entry paths use, whichever branch built it."""

    LANGUAGE_NAMES: Dict[str, str]

    @property
    def backend(self) -> TranslationService: ...

    def translate(
        self,
        segments: List[SubtitleSegment],
        source_lang: str,
        target_lang: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        context_window: int = 3,
    ) -> List[SubtitleSegment]: ...

    def untranslated_degradation(self, target_lang: str, total: int) -> Optional[Degradation]: ...

    def effective_batch_size(self) -> int: ...

    def check_connection(self) -> bool: ...

    def check_model_available(self) -> bool: ...


def create_translator(
    config: AppConfig,
    *,
    save_failed_log: bool = False,
    failed_log_path: Optional[str] = None,
) -> Translator:
    """Build the translator ``config.translation.backend`` selects: the LLM branch for an
    LLM backend, the machine translation branch for MT_BACKENDS. Only that backend is ever
    built; there is no fallback to another.

    Args:
        config: The application config.
        save_failed_log: Write the LLM branch's parse failures to ``failed_log_path`` (or a
            timestamped file in the working directory); the MT branch has none to write.
        failed_log_path: Where that log goes.

    Raises:
        ValueError: translation.backend names no known backend, or its API key is unset.
    """
    if config.translation.backend in MT_BACKENDS:
        from .mt import MachineTranslator, create_mt_backend

        return MachineTranslator(config.translation, create_mt_backend(config))
    return SubtitleTranslator.from_config(
        config, save_failed_log=save_failed_log, failed_log_path=failed_log_path
    )
