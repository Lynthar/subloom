"""HTTP server command — exposes subloom as a REST API."""

import typer


def serve(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Bind address. Loopback by default; pass 0.0.0.0 explicitly to "
        "expose the API to the network (bearer token travels in plaintext "
        "HTTP — put TLS in front for anything beyond a trusted LAN).",
    ),
    port: int = typer.Option(8765, "--port", "-p", help="Bind port"),
    workers: int = typer.Option(
        1,
        "--workers",
        "-w",
        min=1,
        max=4,
        help="Concurrent job workers. GPU-bound — leave at 1 unless you know what you're doing.",
    ),
    no_auth: bool = typer.Option(
        False,
        "--no-auth",
        help="Disable bearer-token auth. Only safe on a fully trusted host (loopback only).",
    ),
    log_level: str = typer.Option("info", "--log-level", help="uvicorn log level"),
):
    """
    Start the subloom HTTP server.

    Auth: by default the server requires a bearer token, read from the
    SUBLOOM_TOKEN environment variable. Pick a long random string and
    set it before launching:

        export SUBLOOM_TOKEN="$(openssl rand -hex 32)"
        subloom serve

    Use --no-auth ONLY when binding to 127.0.0.1 on a single-user machine.
    """
    try:
        import uvicorn
    except ImportError:
        typer.echo(
            "ERROR: serve requires the [serve] extra. Install with:\n  pip install -e '.[serve]'",
            err=True,
        )
        raise typer.Exit(1)

    from ...server.auth import TOKEN_ENV_VAR, get_configured_token
    from ...server.app import create_app
    from ..app import apply_translation_flags, get_config

    if not no_auth:
        if get_configured_token() is None:
            typer.echo(
                f"ERROR: ${TOKEN_ENV_VAR} is not set. Either set it (recommended)\n"
                f"or pass --no-auth (only safe on a trusted single-user host).",
                err=True,
            )
            raise typer.Exit(2)
    else:
        if host not in ("127.0.0.1", "localhost"):
            typer.echo(
                f"WARNING: --no-auth on a non-loopback host ({host}). Anyone on "
                f"the network can submit jobs.",
                err=True,
            )

    # get_config() (not letting create_app fall back to AppConfig.load()) so
    # the root --config flag reaches the server like every other command.
    config = get_config()
    apply_translation_flags(config, None, None)  # An unset API key fails here, not in every job
    app_instance = create_app(config=config, max_workers=workers, require_auth=not no_auth)

    uvicorn.run(
        app_instance,
        host=host,
        port=port,
        log_level=log_level,
    )
