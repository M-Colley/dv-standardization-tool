"""Shared text encoding helpers backed by charset-normalizer."""

from __future__ import annotations

from pathlib import Path

from charset_normalizer import from_bytes

_COMMON_TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
_CHARSET_SAMPLE_BYTES = 32768


def _coerce_encoding(encoding: str) -> str:
    """Upgrade plain ``ascii`` to ``utf-8``.

    A file that sniffs as ASCII over its leading bytes may still contain
    non-ASCII further in; ``utf-8`` is a strict superset, so reading ASCII
    content as UTF-8 is always safe and avoids surprising decode failures
    downstream.
    """
    return "utf-8" if encoding.lower() == "ascii" else encoding


def detect_text_encoding(raw: bytes) -> str:
    """Best-effort text encoding detection for tabular text inputs.

    This is the single shared encoding detector for the project; the
    convert_dv loader and the survey parsers delegate to it so a given file
    is classified identically regardless of the call path.
    """
    sample = raw[:_CHARSET_SAMPLE_BYTES]
    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if sample.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if sample.startswith(b"\xfe\xff"):
        return "utf-16-be"

    if not sample:
        return "utf-8"

    # ``charset-normalizer`` already ranks candidates internally; ``best()``
    # returns the highest-confidence match (or ``None``) for the sampled bytes.
    best_match = from_bytes(sample).best()
    if best_match and best_match.encoding:
        return _coerce_encoding(best_match.encoding)

    for encoding in _COMMON_TEXT_ENCODINGS:
        try:
            sample.decode(encoding)
            return _coerce_encoding(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return "latin-1"


def detect_file_encoding(path: Path, sample_bytes: int = _CHARSET_SAMPLE_BYTES) -> str:
    """Detect the encoding of a text file from its leading bytes.

    Reads at most ``sample_bytes`` from the file so detection cost stays
    constant regardless of file size.
    """
    with path.open("rb") as fh:
        return detect_text_encoding(fh.read(sample_bytes))
