"""Shared text encoding helpers backed by charset-normalizer."""

from __future__ import annotations

from pathlib import Path

from charset_normalizer import from_bytes

_COMMON_TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


def detect_text_encoding(raw: bytes) -> str:
    """Best-effort text encoding detection for tabular text inputs."""
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if raw.startswith(b"\xfe\xff"):
        return "utf-16-be"

    if raw:
        best_match = from_bytes(raw).best()
        if best_match and best_match.encoding:
            return best_match.encoding

    for encoding in _COMMON_TEXT_ENCODINGS:
        try:
            raw.decode(encoding)
            return encoding
        except (UnicodeDecodeError, LookupError):
            continue
    return "latin-1"


def detect_file_encoding(path: Path, sample_bytes: int = 32768) -> str:
    """Detect the encoding of a text file from its leading bytes."""
    return detect_text_encoding(path.read_bytes()[:sample_bytes])
