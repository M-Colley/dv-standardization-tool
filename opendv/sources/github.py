"""GitHub source-loader URL parsing helpers.

Parsing is split out from ``run_batch_standardization`` so it can be
unit-tested without paying the cost of the full pipeline import.
``run_batch_standardization`` re-exports ``GitHubLocation`` and
``_parse_github_location`` for backwards compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

GITHUB_HOSTS: set[str] = {"github.com", "www.github.com"}


@dataclass(frozen=True)
class GitHubLocation:
    """Parsed pieces of a GitHub URL accepted in the manifest.

    ``clone_url`` is always the canonical ``https://github.com/<owner>/<repo>.git``
    form so the orchestrator can pass it directly to ``git clone``.
    ``ref`` is set when the URL points at ``/tree/<ref>/`` or
    ``/blob/<ref>/`` paths; ``subpath`` captures any trailing path.
    """

    clone_url: str
    ref: str | None = None
    subpath: Path | None = None


def _parse_github_location(location: str) -> GitHubLocation:
    """Parse a GitHub repo / tree / blob URL into a ``GitHubLocation``.

    Accepts the bare repo URL (``https://github.com/<owner>/<repo>``) as
    well as scoped tree/blob URLs (``https://github.com/<owner>/<repo>/tree/<ref>/<sub/path>``).
    Raises ``ValueError`` for empty input, non-GitHub hosts, or malformed
    paths so the manifest validator can surface a clear message.
    """
    text = str(location or "").strip()
    if not text:
        raise ValueError("GitHub location must be a non-empty repository URL.")

    parsed = urlparse(text)
    host = parsed.netloc.lower()
    if host not in GITHUB_HOSTS:
        raise ValueError(f"Unsupported GitHub host in location: {location}")

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError(
            "GitHub location must point to a repository like https://github.com/<owner>/<repo> "
            "or a scoped tree URL like https://github.com/<owner>/<repo>/tree/<ref>/<path>."
        )

    owner = parts[0]
    repo = parts[1][:-4] if parts[1].endswith(".git") else parts[1]
    clone_url = f"https://github.com/{owner}/{repo}.git"
    ref: str | None = None
    subpath: Path | None = None

    if len(parts) > 2:
        if parts[2] not in {"tree", "blob"} or len(parts) < 4:
            raise ValueError(
                "GitHub location must be a repository root URL or a tree/blob URL with a ref."
            )
        ref = parts[3]
        if len(parts) > 4:
            subpath = Path(*parts[4:])

    return GitHubLocation(clone_url=clone_url, ref=ref, subpath=subpath)


__all__ = [
    "GITHUB_HOSTS",
    "GitHubLocation",
    "_parse_github_location",
]
