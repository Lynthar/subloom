"""HuggingFace Hub downloads with byte-level progress, shared by the ASR backends."""

import os
import threading
from pathlib import Path
from typing import Callable, List, Optional


def download_repos(
    repo_ids: List[str],
    *,
    cache_dir: Optional[str],
    token: Optional[str],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Download each repo's snapshot into the HF cache, one after another.

    Args:
        progress_callback: Receives (bytes_on_disk, 0) about twice a second, the total
            being unknown here; size the bar from the backend's own estimate. Bytes count
            every file already in the repos' cache folders, partial downloads included.

    Raises:
        Exception: Whatever huggingface_hub raises for a failed download.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    from huggingface_hub.file_download import repo_folder_name
    from huggingface_hub.utils import (
        are_progress_bars_disabled,
        disable_progress_bars,
        enable_progress_bars,
    )

    # snapshot_download's tqdm_class only counts files, so bytes are read off the disk.
    root = Path(cache_dir or HF_HUB_CACHE)
    folders = [root / repo_folder_name(repo_id=r, repo_type="model") for r in repo_ids]
    stop = threading.Event()

    def report(callback: Callable[[int, int], None]) -> None:
        while not stop.wait(0.5):
            callback(sum(_bytes_on_disk(f) for f in folders), 0)

    reporter = (
        threading.Thread(target=report, args=(progress_callback,), daemon=True)
        if progress_callback
        else None
    )
    # Our Rich bar draws the progress; HF's own bars would fight it for the console.
    bars_were_disabled = are_progress_bars_disabled()
    disable_progress_bars()
    if reporter:
        reporter.start()
    try:
        for repo_id in repo_ids:
            snapshot_download(repo_id, cache_dir=cache_dir, token=token)
    finally:
        stop.set()
        if reporter:
            reporter.join()
        if not bars_were_disabled:
            enable_progress_bars()
    if progress_callback:
        progress_callback(sum(_bytes_on_disk(f) for f in folders), 0)


def _bytes_on_disk(folder: Path) -> int:
    # Symlinks are skipped: with them, snapshots/ links to the same bytes blobs/ holds.
    total = 0
    for dirpath, _dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(dirpath, name)
            if not os.path.islink(path):
                try:
                    total += os.path.getsize(path)
                except OSError:  # renamed from .incomplete between listing and stat
                    pass
    return total
