"""Model download manager with resume support for Ollama."""

from dataclasses import dataclass
from typing import Optional, Callable, Generator, List
import logging

from ollama import Client, ResponseError

logger = logging.getLogger(__name__)


@dataclass
class DownloadProgress:
    """Download progress information."""

    status: str  # "pulling manifest", "downloading", "verifying", "done", "error"
    total_bytes: int
    completed_bytes: int
    digest: str

    @property
    def percent(self) -> float:
        """Get download percentage."""
        if self.total_bytes == 0:
            return 0.0
        return self.completed_bytes / self.total_bytes * 100

    @property
    def is_downloading(self) -> bool:
        """Check if actively downloading."""
        return self.status in ("downloading", "pulling")

    @property
    def is_complete(self) -> bool:
        """Check if download is complete."""
        return self.status == "success"


class OllamaModelManager:
    """Manage Ollama model downloads with progress tracking and resume support."""

    def __init__(self, host: str = "http://localhost:11434"):
        """
        Initialize model manager.

        Args:
            host: Ollama API host URL.
        """
        self.host = host
        self._client: Optional[Client] = None

    @property
    def client(self) -> Client:
        """Get or create Ollama client."""
        if self._client is None:
            self._client = Client(host=self.host)
        return self._client

    def list_models(self) -> List[str]:
        """
        List available local models.

        Returns:
            List of model names.

        Raises:
            ollama.ResponseError, or the transport error underneath it, when the
            daemon cannot be asked. Do not fold that into []: an empty list means
            Ollama answered and holds no models.

        Note:
            Ollama API returns ListResponse with models: Sequence[Model]
            where Model.model is the model name (Optional[str]).
            See: https://github.com/ollama/ollama-python/blob/main/ollama/_types.py
        """
        response = self.client.list()
        # ListResponse has 'models' attribute containing Model objects
        # Each Model has 'model' field (not 'name') as the model identifier
        models = getattr(response, "models", []) or []

        result = []
        for m in models:
            # Model.model is the actual field name in Ollama's API
            name = getattr(m, "model", None)
            if name:
                result.append(name)
        return result

    def is_model_available(self, model: str) -> bool:
        """
        Check if a model is downloaded locally.

        Args:
            model: Model name (e.g., "qwen2.5:14b").

        Returns:
            True if model is available.

        Raises:
            Whatever list_models raises: "cannot ask Ollama" is not "not installed",
            and a caller that reads it so offers a download that cannot work.
        """
        available_models = self.list_models()

        # Exact, tag-aware match. The old `model_base in available` substring test reported
        # "qwen2.5:32b" as installed when only "qwen2.5:7b" was — the download was skipped and
        # translation died on model-not-found after transcription had already run.
        def _normalize(name: str) -> str:
            return name if ":" in name else f"{name}:latest"

        target = _normalize(model)
        return any(_normalize(a) == target for a in available_models)

    def pull_model(
        self,
        model: str,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> Generator[DownloadProgress, None, None]:
        """
        Pull (download) a model with streaming progress.

        Ollama automatically supports resume - if download was interrupted,
        running pull again will continue from where it left off.

        Args:
            model: Model name to download.
            progress_callback: Optional callback for progress updates.

        Yields:
            DownloadProgress objects with current download state.

        Raises:
            RuntimeError: If download fails, including a stream that ends without
                the model showing up in the local list.
        """
        logger.info(f"Starting model pull: {model}")

        try:
            for progress in self.client.pull(model, stream=True):
                # Ollama's ProgressResponse carries status/total/completed/digest, and total and
                # completed are Optional — they can be None during initialization.
                status = getattr(progress, "status", "unknown") or "unknown"
                total = getattr(progress, "total", None)
                completed = getattr(progress, "completed", None)
                digest = getattr(progress, "digest", None) or ""

                # Ensure numeric values are not None (API returns None before download starts)
                total = total if total is not None else 0
                completed = completed if completed is not None else 0

                dp = DownloadProgress(
                    status=status,
                    total_bytes=total,
                    completed_bytes=completed,
                    digest=digest,
                )

                if progress_callback:
                    progress_callback(dp)

                yield dp

                if status == "success":
                    logger.info(f"Model {model} downloaded successfully")

            installed = self.is_model_available(model)
        except ResponseError as e:
            logger.error(f"Model pull failed: {e}")
            raise RuntimeError(f"Failed to download model {model}: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during model pull: {e}")
            raise RuntimeError(f"Download error: {e}") from e

        # A stream can end cleanly without the model landing, success event or not.
        if not installed:
            raise RuntimeError(f"Download of model {model} ended but it is not installed")

    def ensure_model(
        self,
        model: str,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        auto_pull: bool = True,
    ) -> bool:
        """
        Ensure a model is available, optionally downloading it.

        Args:
            model: Model name to ensure.
            progress_callback: Optional callback for download progress.
            auto_pull: If True, automatically download missing model.

        Returns:
            True if model is ready to use; a failed download is False, not an exception.

        Raises:
            Whatever list_models raises when the daemon cannot be asked.
        """
        if self.is_model_available(model):
            logger.info(f"Model {model} is already available")
            return True

        if not auto_pull:
            logger.warning(f"Model {model} not available and auto_pull is disabled")
            return False

        logger.info(f"Model {model} not found, starting download...")

        try:
            # Consume the generator to complete the download
            for _ in self.pull_model(model, progress_callback):
                pass

            return self.is_model_available(model)

        except RuntimeError as e:
            logger.error(f"Failed to ensure model: {e}")
            return False

    def check_connection(self) -> bool:
        """
        Check if Ollama service is reachable.

        Returns:
            True if connection is successful.
        """
        try:
            self.client.list()
            return True
        except Exception as e:
            logger.error(f"Cannot connect to Ollama at {self.host}: {e}")
            return False


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human-readable string."""
    if bytes_value < 1024:
        return f"{bytes_value}B"
    elif bytes_value < 1024 * 1024:
        return f"{bytes_value / 1024:.1f}KB"
    elif bytes_value < 1024 * 1024 * 1024:
        return f"{bytes_value / (1024 * 1024):.1f}MB"
    else:
        return f"{bytes_value / (1024 * 1024 * 1024):.2f}GB"
