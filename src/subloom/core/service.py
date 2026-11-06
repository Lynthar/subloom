"""What every translation service shares: the error it raises, the checks the entry paths run
before any work, and the retry policy around each request."""

import logging
import time
from typing import Any, Callable, List, Optional, Protocol, TypeVar

import httpx

from ..exceptions import TranslationError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceError(Exception):
    """The server answered with an error; ``status_code`` is None when it gave none."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class TranslationService(Protocol):
    """The probes an entry path runs before any work, whatever kind of service it is."""

    label: str  # Kind of service, for messages
    endpoint: str  # Where it is reached, for messages
    model: str  # Empty for a service without a model choice

    def check_connection(self) -> bool:
        """True when the server answers at all."""
        ...

    def list_models(self) -> List[str]:
        """Model names the server offers; [] for a service without a model choice.

        Raises:
            ServiceError, or the transport error underneath it, when the server cannot
            be asked. Never folded into []: an empty list means it answered and has none.
        """
        ...

    def check_model_available(self) -> bool:
        """True when the configured model is offered and the credentials are accepted.

        Raises:
            ServiceError: The server refused to answer (a rejected key, for one).
        """
        ...


def call_with_retries(call: Callable[[], T], max_retries: int) -> T:
    """Run ``call``, retrying transport failures, 429 and 5xx with exponential backoff.

    Raises:
        TranslationError: A 4xx other than 429 (it will not heal: a bad request, a wrong
            model or key, an exhausted quota) on the first try, anything else once
            ``max_retries`` attempts are spent.
        Exception: Whatever else ``call`` raises, unwrapped.
    """
    for attempt in range(max_retries):
        try:
            return call()
        except (ServiceError, httpx.TransportError) as e:
            status = e.status_code if isinstance(e, ServiceError) else None
            if isinstance(status, int) and 400 <= status < 500 and status != 429:
                raise TranslationError(f"Translation failed: {e}") from e
            logger.warning(
                f"Translation request failed ({type(e).__name__}) "
                f"(attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt == max_retries - 1:
                raise TranslationError(
                    f"Translation failed after {max_retries} attempts: {e}"
                ) from e
            time.sleep(min(2**attempt, 8))
    raise TranslationError("Translation was not attempted: max_retries is below 1")


def json_or_raise(response: httpx.Response) -> Any:
    """The body of a successful response.

    Raises:
        ServiceError: An HTTP error status (carried as ``status_code``), or a body that is
            not JSON.
    """
    if response.status_code >= 400:
        raise ServiceError(
            f"HTTP {response.status_code}: {response.text[:300]}", response.status_code
        )
    try:
        return response.json()
    except ValueError as e:
        raise ServiceError(f"Reply is not JSON: {response.text[:200]}") from e
