"""Tabular data loaders + format registry for the DV standardization tool.

Owns the format registry that maps file extensions to per-format loader
functions, plus the encoding / delimiter / magic-byte sniffers and the
inverse ``save_output_file`` helper. Extracted from ``scripts.convert_dv``
so the loader plumbing can be unit-tested without dragging in the
column-mapping and metadata-inference machinery.

``scripts.convert_dv`` re-exports every public name (``_FORMAT_REGISTRY``,
``load_input_file``, ``save_output_file``, ``_load_text_table``, ...)
so existing imports keep working.

Supported input formats are the union of the keys registered by the
``@_register_format`` decorators below: ``.csv``, ``.tsv``, ``.txt``,
``.dat``, ``.xlsx``, ``.xls``, ``.xlsm``, ``.ods``, ``.pkl``, ``.pickle``,
``.parquet``, ``.sav``, ``.zsav``, ``.dta``, ``.feather``, ``.arrow``,
``.json``, ``.jsonl``, ``.ndjson``. Output formats supported by
``save_output_file``: ``.csv``, ``.xlsx``, ``.xls``, ``.pkl``, ``.pickle``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format registry — extension → loader function
# ---------------------------------------------------------------------------

_FORMAT_REGISTRY: dict[str, Callable[[Path], pd.DataFrame]] = {}


def _register_format(*extensions: str):
    """Decorator that registers a loader for one or more file extensions."""
    def decorator(fn: Callable[[Path], pd.DataFrame]) -> Callable[[Path], pd.DataFrame]:
        for ext in extensions:
            _FORMAT_REGISTRY[ext.lower()] = fn
        return fn
    return decorator


# Magic-byte → canonical extension mapping for content-based sniffing.
_MAGIC_BYTES: list[tuple[bytes, str | None]] = [
    (b"PAR1", ".parquet"),       # Parquet magic
    (b"\x7fELF", None),          # binary ELF, skip
    (b"$FL2", ".sav"),           # SPSS .sav
    (b"\xef\xbb\xbf", ".csv"),   # UTF-8 BOM → treat as CSV
    (b"<", ".xml"),              # XML
    (b"{", ".json"),             # JSON object
    (b"[", ".json"),             # JSON array
]


def _sniff_format(path: Path) -> str | None:
    """Return a canonical extension by inspecting the first bytes of *path*, or None."""
    try:
        raw = path.read_bytes()[:8]
    except OSError:
        return None
    for magic, ext in _MAGIC_BYTES:
        if raw.startswith(magic):
            return ext
    return None


# ---------------------------------------------------------------------------
# Encoding + delimiter detection
# ---------------------------------------------------------------------------

def _detect_encoding(path: Path) -> str:
    """Detect file encoding, trying chardet then common fallbacks."""
    raw = path.read_bytes()[:32768]

    # BOM detection first
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if raw.startswith(b"\xfe\xff"):
        return "utf-16-be"

    # Try chardet
    try:
        import chardet
        result = chardet.detect(raw)
        if result and result.get("confidence", 0) > 0.75:
            encoding = (result.get("encoding") or "utf-8").lower()
            if encoding == "ascii":
                return "utf-8"
            return encoding
    except ImportError:
        pass

    # Fallback chain
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            raw.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    return "latin-1"


def _detect_delimiter(sample: str, prefer: str | None = None) -> str:
    """Detect delimiter by counting consistent occurrences across rows.

    More robust than csv.Sniffer: counts each candidate delimiter in each
    of the first N rows, then picks the one with the lowest coefficient
    of variation (most consistent counts = most likely the real delimiter).
    """
    candidates = [",", ";", "\t", "|", " "]
    if prefer:
        candidates = [prefer] + [c for c in candidates if c != prefer]

    rows = [r for r in sample.splitlines() if r.strip()][:15]
    if not rows:
        return prefer or ","

    best_delim = prefer or ","
    best_score = float("inf")

    for delim in candidates:
        counts = [row.count(delim) for row in rows]
        if max(counts) == 0:
            continue
        mean = sum(counts) / len(counts)
        if mean == 0:
            continue
        variance = sum((c - mean) ** 2 for c in counts) / len(counts)
        cv = (variance ** 0.5) / mean  # coefficient of variation
        if cv < best_score:
            best_score = cv
            best_delim = delim

    return best_delim


# ---------------------------------------------------------------------------
# Per-format loaders
# ---------------------------------------------------------------------------

@_register_format(".csv", ".tsv", ".txt", ".dat")
def _load_text_table(path: Path) -> pd.DataFrame:
    """Load a delimited text file with robust encoding and delimiter auto-detection."""
    # For .tsv/.dat/.txt prefer tab first; for .csv prefer comma first.
    suffix = path.suffix.lower()
    prefer_delim: str | None = "\t" if suffix in {".tsv", ".dat", ".txt"} else ","

    encoding = _detect_encoding(path)
    try:
        sample = path.read_text(encoding=encoding, errors="replace")[:16384]
    except OSError:
        sample = ""

    delimiter = _detect_delimiter(sample, prefer=prefer_delim)

    try:
        return pd.read_csv(
            path,
            sep=delimiter,
            encoding=encoding,
            engine="python",
            on_bad_lines="warn",
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"File '{path}' is empty or contains no parseable data."
        ) from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Failed to parse '{path}': {exc}. The file may be malformed."
        ) from exc


@_register_format(".xlsx", ".xls", ".xlsm", ".ods")
def _load_excel(path: Path) -> pd.DataFrame:
    """Load an Excel workbook, auto-selecting the most data-rich sheet."""
    try:
        xl = pd.ExcelFile(path)
    except ImportError as exc:
        raise ImportError(
            "Reading Excel files requires openpyxl: pip install openpyxl"
        ) from exc

    if len(xl.sheet_names) == 1:
        return xl.parse(xl.sheet_names[0])

    # Multiple sheets: pick the one with the most non-empty cells
    best_sheet: str | None = None
    best_count = -1
    for name in xl.sheet_names:
        df = xl.parse(name)
        count = int(df.notna().sum().sum())
        if count > best_count:
            best_count = count
            best_sheet = name

    logger.info(
        "Multi-sheet workbook '%s': selected sheet '%s' (%d non-empty cells). "
        "Other sheets: %s",
        path.name,
        best_sheet,
        best_count,
        [n for n in xl.sheet_names if n != best_sheet],
    )
    return xl.parse(best_sheet)


@_register_format(".pkl", ".pickle")
def _load_pickle(path: Path) -> pd.DataFrame:
    """Load a pickle file and coerce its payload to a DataFrame."""
    payload = pd.read_pickle(path)
    return _coerce_pickled_payload_to_dataframe(payload)


@_register_format(".parquet")
def _load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file using pyarrow or fastparquet."""
    try:
        import pyarrow.parquet as _pq  # noqa: F401
        return pd.read_parquet(path)
    except ImportError:
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except ImportError:
            raise ImportError(
                "Reading Parquet requires pyarrow or fastparquet: pip install pyarrow"
            )


@_register_format(".sav", ".zsav")
def _load_spss(path: Path) -> pd.DataFrame:
    """Load an SPSS .sav / .zsav file via pandas.read_spss."""
    try:
        return pd.read_spss(path)
    except Exception as exc:
        raise ValueError(f"Failed to read SPSS file '{path}': {exc}") from exc


@_register_format(".dta")
def _load_stata(path: Path) -> pd.DataFrame:
    """Load a Stata .dta file."""
    return pd.read_stata(path, convert_categoricals=False)


@_register_format(".feather", ".arrow")
def _load_feather(path: Path) -> pd.DataFrame:
    """Load a Feather/Arrow file via pyarrow."""
    try:
        return pd.read_feather(path)
    except ImportError:
        raise ImportError(
            "Reading Feather/Arrow requires pyarrow: pip install pyarrow"
        )


@_register_format(".json")
def _load_json(path: Path) -> pd.DataFrame:
    """Load a JSON file — supports records array, JSON Lines, and nested JSON."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # JSON Lines: first non-empty line starts with '{'
    if lines and lines[0].startswith("{"):
        try:
            records = [json.loads(line) for line in lines]
            return pd.json_normalize(records)
        except json.JSONDecodeError:
            pass
    return pd.read_json(path)


@_register_format(".jsonl", ".ndjson")
def _load_jsonlines(path: Path) -> pd.DataFrame:
    """Load a JSON Lines / NDJSON file."""
    return pd.read_json(path, lines=True)


# ---------------------------------------------------------------------------
# Public load / save / coerce entry points
# ---------------------------------------------------------------------------

def load_input_file(input_path: str) -> pd.DataFrame:
    """Load tabular input data using the format registry.

    Format is determined first by file extension (registry lookup), then
    by content-based magic-byte sniffing if the extension is unrecognised.
    """
    path = Path(input_path)
    suffix = path.suffix.lower()

    # Try registry by extension first
    loader = _FORMAT_REGISTRY.get(suffix)
    if loader is None:
        # Fall back to content sniffing
        sniffed = _sniff_format(path)
        if sniffed is not None:
            suffix = sniffed
        loader = _FORMAT_REGISTRY.get(suffix)

    if loader is None:
        supported = ", ".join(sorted(_FORMAT_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported input format: '{path.suffix or 'no extension'}'. "
            f"Supported formats: {supported}"
        )

    return loader(path)


def _coerce_pickled_payload_to_dataframe(payload: Any) -> pd.DataFrame:
    """Normalize common pickle payload shapes into a DataFrame."""
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, pd.Series):
        return payload.to_frame()
    if isinstance(payload, dict):
        if not payload:
            return pd.DataFrame()
        try:
            return pd.DataFrame(payload)
        except ValueError:
            return pd.json_normalize(payload)
    if isinstance(payload, (list, tuple)):
        if not payload:
            return pd.DataFrame()
        return pd.DataFrame(payload)
    if hasattr(payload, "shape"):
        try:
            return pd.DataFrame(payload)
        except ValueError:
            pass

    raise ValueError(
        f"Unsupported pickle payload type '{type(payload).__name__}'. "
        "Expected a DataFrame-like object such as a pandas DataFrame, Series, dict, list, or array."
    )


def save_output_file(df: pd.DataFrame, output_path: str) -> None:
    """Save standardized data to CSV, Excel, or pickle based on file extension."""
    output_file = Path(output_path)
    suffix = output_file.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        try:
            df.to_excel(output_file, index=False)
        except ImportError as exc:
            raise ImportError(
                "Writing Excel files requires the optional dependency 'openpyxl'. "
                "Install it with: pip install openpyxl"
            ) from exc
        return
    if suffix == ".csv":
        df.to_csv(output_file, index=False)
        return
    if suffix in {".pkl", ".pickle"}:
        df.to_pickle(output_file)
        return

    raise ValueError(
        f"Unsupported output format: '{suffix or 'no extension'}'. "
        "Supported formats are .csv, .xlsx, .xls, .pkl, and .pickle."
    )


__all__ = [
    "_FORMAT_REGISTRY",
    "_MAGIC_BYTES",
    "_register_format",
    "_sniff_format",
    "_detect_encoding",
    "_detect_delimiter",
    "_load_text_table",
    "_load_excel",
    "_load_pickle",
    "_load_parquet",
    "_load_spss",
    "_load_stata",
    "_load_feather",
    "_load_json",
    "_load_jsonlines",
    "load_input_file",
    "_coerce_pickled_payload_to_dataframe",
    "save_output_file",
]
