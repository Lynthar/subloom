"""Degradations: a run that finished, but worse than configured.

The codes are part of the HTTP API (each job lists its degradations), so never rename one.

| code | meaning |
|---|---|
| whisperx_missing | WhisperX is not installed; plain faster-whisper timing was used |
| alignment_failed | forced alignment failed; segment-level timestamps were used |
| alignment_unsupported_language | the aligner is not trained on the spoken language |
| cpu_fallback | CUDA was unavailable; the model ran on the CPU |
| flash_attention_missing | Qwen3-ASR ran without FlashAttention 2; long segments lose accuracy |
| auto_model_fallback | --auto-model could not read free GPU memory and fell back to small |
| translation_incomplete | some segments were left in the source language |
| names_not_found | the name pass gave no usable answer; names were translated without it |
| names_partial | the transcript did not fit the model's context; the name pass read only part |
"""

from dataclasses import dataclass

WHISPERX_MISSING = "whisperx_missing"
ALIGNMENT_FAILED = "alignment_failed"
ALIGNMENT_UNSUPPORTED_LANGUAGE = "alignment_unsupported_language"
CPU_FALLBACK = "cpu_fallback"
FLASH_ATTENTION_MISSING = "flash_attention_missing"
AUTO_MODEL_FALLBACK = "auto_model_fallback"
TRANSLATION_INCOMPLETE = "translation_incomplete"
NAMES_NOT_FOUND = "names_not_found"
NAMES_PARTIAL = "names_partial"


@dataclass(frozen=True)
class Degradation:
    """One way a run fell short of its configuration; reported with the result, never fatal."""

    code: str  # One of the codes above
    message: str  # One line for people
