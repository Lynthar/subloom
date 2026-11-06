"""Bearer token authentication."""

import os
import secrets
from typing import Optional

from fastapi import Header, HTTPException, status

TOKEN_ENV_VAR = "SUBLOOM_TOKEN"


def get_configured_token() -> Optional[str]:
    """Read the expected bearer token from the environment.

    Returns None if the env var is unset OR set to empty string. In that case
    auth is disabled — callers should refuse to start the server in that mode
    unless --no-auth was passed explicitly.
    """
    token = os.environ.get(TOKEN_ENV_VAR, "").strip()
    return token or None


async def require_token(authorization: Optional[str] = Header(None)) -> None:
    """FastAPI dependency that enforces Bearer token auth.

    Uses constant-time comparison to avoid timing attacks. When auth is
    disabled at startup this dependency is replaced with a no-op via
    `app.dependency_overrides`, so this function is only called when a token
    is actually configured.
    """
    expected = get_configured_token()
    if expected is None:
        # Defensive: should never happen when auth is enabled. Refuse rather
        # than silently allowing through.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server token not configured",
        )

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    presented = authorization[len("Bearer ") :].strip()
    if not secrets.compare_digest(presented, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def noop_auth() -> None:
    """No-op dependency used when auth is explicitly disabled."""
    return None
