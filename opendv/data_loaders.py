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
``.json``, ``.jsonl``, ``.ndjson``, ``.mat`` (MATLAB v4 to v7.2, read with
``scipy.io.loadmat``; the ``variable``, ``columns`` and ``dims`` table
options say which variable becomes the table). Pickles (``.pkl``, ``.pickle``) are
discovered but refused unless the caller passes ``allow_pickle=True``:
unpickling runs code from the file. Output formats supported by
``save_output_file``: ``.csv``, ``.xlsx``, ``.parquet``, ``.pkl``,
``.pickle``.
"""

from __future__ import annotations

import csv
import fnmatch
import io
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
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
    (b"{", ".json"),             # JSON object
    (b"[", ".json"),             # JSON array
]


def _sniff_format(path: Path) -> str | None:
    """Return a canonical extension by inspecting the first bytes of *path*, or None."""
    try:
        with path.open("rb") as fh:
            raw = fh.read(8)
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
    """Detect file encoding using charset-normalizer with a BOM fast path.

    The BOM short-circuit returns the canonical ``utf-8-sig`` /
    ``utf-16-le`` / ``utf-16-be`` labels — charset-normalizer would also
    detect them, but the labels can vary slightly between releases and
    pandas treats the explicit ``-sig`` variant specially when reading
    CSVs (it strips the BOM from the first column header). Keeping the
    short-circuit guarantees that behaviour.

    For everything else we delegate to the shared
    ``encoding_utils.detect_text_encoding`` helper (backed by
    ``charset_normalizer``) so a file is classified identically here and in
    the survey parsers. ``best()`` runs over a 32 KiB sample — enough for
    almost all real-world CSVs — and ``ascii`` is coerced to ``utf-8`` so
    downstream pandas reads don't surprise us when the file later contains
    non-ASCII bytes.
    """
    from opendv.encoding_utils import detect_text_encoding

    return detect_text_encoding(_read_head(path))


_ENCODING_SNIFF_BYTES = 32768
_DELIMITER_SAMPLE_CHARS = 16384


def _read_head(path: Path) -> bytes:
    with path.open("rb") as fh:
        return fh.read(_ENCODING_SNIFF_BYTES)


def _decode_text_sample(head: bytes, encoding: str, path: Path) -> str:
    """Return the delimiter-sniffing sample without reopening *path*.

    Equivalent to ``open(path, encoding=encoding, errors="replace").read(16384)``
    (text mode, universal newlines). Opening files is surprisingly expensive on
    Windows with real-time AV scanning, so the sample is decoded from the
    bytes already read for encoding detection; it only falls back to a real
    read when those bytes cannot supply a full sample (long multi-byte text).
    """
    text = head.decode(encoding, errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    # A head cut mid-file can end in a split multi-byte character or a "\r"
    # whose "\n" was not read; keep a small margin so neither reaches the sample.
    if len(head) < _ENCODING_SNIFF_BYTES or len(text) >= _DELIMITER_SAMPLE_CHARS + 4:
        return text[:_DELIMITER_SAMPLE_CHARS]
    with path.open("r", encoding=encoding, errors="replace") as fh:
        return fh.read(_DELIMITER_SAMPLE_CHARS)


def _detect_delimiter(sample: str, prefer: str | None = None) -> str:
    """Detect delimiter by counting consistent occurrences across rows.

    More robust than csv.Sniffer: counts each candidate delimiter in each
    of the first N rows, then picks the one with the lowest coefficient
    of variation (most consistent counts = most likely the real delimiter).
    """
    # Structured delimiters are evaluated first; whitespace is only a last
    # resort. Otherwise a "; "- or ", "-delimited file (e.g. ROADS / eHMI VR
    # logs) can be mis-split on the space, producing broken column names like
    # "userPositionX;", "focused", "object".
    structured = [",", ";", "\t", "|"]
    if prefer and prefer in structured:
        structured = [prefer] + [c for c in structured if c != prefer]
    whitespace_fallback = [" "]

    rows = [r for r in sample.splitlines() if r.strip()][:15]
    if not rows:
        return prefer or ","

    def _best(cands: list[str]) -> str | None:
        best_delim: str | None = None
        best_score = float("inf")
        for delim in cands:
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

    return _best(structured) or _best(whitespace_fallback) or (prefer or ",")


_DECIMAL_COMMA_NUMBER = re.compile(r"[+-]?\d*,\d+")
_DECIMAL_POINT_NUMBER = re.compile(r"[+-]?\d*\.\d+")


def _detect_decimal(sample: str, delimiter: str) -> str:
    """Return ``","`` for a file written with decimal commas, else ``"."``.

    Spreadsheets in German, French and many other locales export
    semicolon-separated CSV with decimal commas (``0,581``), which pandas
    reads as text unless told ``decimal=","``. A file counts as decimal-comma
    only when its delimiter is not a comma and the sample holds decimal-comma
    numbers but no decimal-point ones: with ``decimal=","`` pandas would turn
    ``1.5`` into text, so mixed or ambiguous files keep the default.
    """
    if delimiter == ",":
        return "."
    fields = [
        field.strip()
        for row in list(csv.reader(io.StringIO(sample), delimiter=delimiter))[1:]
        for field in row
    ]
    has_comma = any(_DECIMAL_COMMA_NUMBER.fullmatch(f) for f in fields)
    has_point = any(_DECIMAL_POINT_NUMBER.fullmatch(f) for f in fields)
    return "," if has_comma and not has_point else "."


# ---------------------------------------------------------------------------
# Per-format loaders
# ---------------------------------------------------------------------------

@_register_format(".csv", ".tsv", ".txt", ".dat")
def _load_text_table(path: Path, **read_kwargs: Any) -> pd.DataFrame:
    """Load a delimited text file with robust encoding and delimiter auto-detection.

    *read_kwargs* go to ``pandas.read_csv`` (the ``skip_rows`` table option
    passes ``skiprows``).
    """
    # For .tsv/.dat/.txt prefer tab first; for .csv prefer comma first.
    suffix = path.suffix.lower()
    prefer_delim: str | None = "\t" if suffix in {".tsv", ".dat", ".txt"} else ","

    from opendv.encoding_utils import detect_text_encoding

    head = _read_head(path)
    encoding = detect_text_encoding(head)
    try:
        sample = _decode_text_sample(head, encoding, path)
    except (OSError, LookupError):
        sample = ""

    delimiter = _detect_delimiter(sample, prefer=prefer_delim)
    decimal = _detect_decimal(sample, delimiter)

    def _read(encoding: str) -> pd.DataFrame:
        # The C engine is typically an order of magnitude faster; it handles
        # single-character separators fine. Fall back to the lenient python
        # engine only when the C parser rejects the file.
        try:
            return pd.read_csv(
                path, sep=delimiter, decimal=decimal, encoding=encoding, on_bad_lines="warn", **read_kwargs
            )
        except UnicodeDecodeError:
            raise
        except (pd.errors.ParserError, ValueError):
            return pd.read_csv(
                path, sep=delimiter, decimal=decimal, encoding=encoding, engine="python", on_bad_lines="warn",
                **read_kwargs,
            )

    try:
        try:
            return _read(encoding)
        except UnicodeDecodeError:
            # Detection sees only the first 32 KB; a byte further in (e.g. a
            # cp1252 apostrophe in a free-text answer) can still fail.
            whole_file = detect_text_encoding(path.read_bytes(), sample_bytes=None)
            logger.info("'%s' is not %s beyond its first 32 KB; reading it as %s", path.name, encoding, whole_file)
            return _read(whole_file)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"File '{path}' is empty or contains no parseable data."
        ) from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Failed to parse '{path}': {exc}. The file may be malformed."
        ) from exc


@_register_format(".xlsx", ".xls", ".xlsm", ".ods")
def _load_excel(path: Path, **parse_kwargs: Any) -> pd.DataFrame:
    """Load an Excel workbook, auto-selecting the most data-rich sheet.

    *parse_kwargs* go to ``pandas.ExcelFile.parse`` for every sheet (the
    ``skip_rows`` table option passes ``skiprows``).
    """
    try:
        xl = pd.ExcelFile(path)
    except ImportError as exc:
        raise ImportError(
            "Reading Excel files requires openpyxl: pip install openpyxl"
        ) from exc

    # Closed on exit: an open workbook stays locked on Windows.
    with xl:
        if len(xl.sheet_names) == 1:
            return xl.parse(xl.sheet_names[0], **parse_kwargs)

        # Multiple sheets: pick the one with the most non-empty cells. Keep each
        # parsed frame so the winner is returned without a second parse pass.
        # A source can name the sheet instead (manifest table_options).
        best_sheet: str | None = None
        best_count = -1
        best_df: pd.DataFrame | None = None
        for name in xl.sheet_names:
            df = xl.parse(name, **parse_kwargs)
            count = int(df.notna().sum().sum())
            if count > best_count:
                best_count = count
                best_sheet = name
                best_df = df

        logger.info(
            "Multi-sheet workbook '%s': selected sheet '%s' (%d non-empty cells). "
            "Other sheets: %s",
            path.name,
            best_sheet,
            best_count,
            [n for n in xl.sheet_names if n != best_sheet],
        )
        return best_df


# Unpickling runs code: a pickle stream names callables that ``pickle.load``
# imports and calls, so loading a downloaded .pkl can execute anything its
# author chose. Pickled tables are therefore refused unless the caller opts in
# for files it trusts (a manifest source's ``allow_pickle: true``, ``opendv
# convert --allow-pickle``). The suffixes stay registered so discovery finds
# pickles and the batch can report which files it refused.
PICKLE_SUFFIXES = frozenset({".pkl", ".pickle"})


class PickleRefusedError(ValueError):
    """A pickled table was not loaded because unpickling was not allowed."""


def pickle_refusal_message(path: Path | str) -> str:
    return (
        f"Refusing to unpickle '{Path(path).name}': loading a pickle runs code embedded in the file. "
        "Convert it to CSV or Parquet in an isolated environment (docs/reproducibility.md, "
        "'Pickled data files'), or allow pickles only for files you created yourself "
        "(manifest: allow_pickle: true; opendv convert: --allow-pickle)."
    )


@_register_format(".pkl", ".pickle")
def _load_pickle(path: Path) -> pd.DataFrame:
    """Refuse a pickled table (see ``PICKLE_SUFFIXES``).

    Only ``load_input_file(..., allow_pickle=True)`` unpickles, so code that
    calls a loader from the registry by suffix never runs a pickle.
    """
    raise PickleRefusedError(pickle_refusal_message(path))


def _unpickle_table(path: Path) -> pd.DataFrame:
    """Unpickle *path* and coerce its payload to a DataFrame. Runs code from the file."""
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
# MATLAB .mat files
# ---------------------------------------------------------------------------
#
# ``scipy.io.loadmat`` parses MAT files of versions 4 to 7.2 without running
# anything they contain: MATLAB objects (``table``, ``timetable``, classdef
# instances) and function handles come back as opaque values, which are
# refused here. A .mat file holds named variables, not one table, so the
# ``variable`` table option names the one to read (a dotted path reaches into
# structs). Without it, the file's variables are read as the fields of one
# struct when they line up, else the most data-rich variable is taken (as for
# a multi-sheet workbook). A variable becomes a table as follows:
#
# - a struct: one column per field; a field holding an n x m matrix gives m
#   columns ``field_1 .. field_m``; scalar fields (e.g. a participant id) are
#   repeated on every row; nested structs give ``outer.inner`` columns;
# - a struct array: one table per element as above, stacked, with a 1-based
#   ``<variable>_index`` column first;
# - a vector: one column; a 2-D matrix: one column per matrix column, named by
#   the ``columns`` option (a list, or the name of a variable holding the
#   names) or ``<variable>_1 ..``;
# - an array with ``dims`` (one name per axis): a long table with one 1-based
#   position column per axis and one value column named after the variable,
#   e.g. ``dims: [participant, condition]`` for a participants x conditions
#   matrix of one measure.

MATLAB_SUFFIXES = frozenset({".mat"})
MATLAB_OPTION_KEYS = ("variable", "columns", "dims")


def _matlab_namespace(path: Path) -> dict[str, Any]:
    """The variables of a MAT file (v4 to v7.2), without its ``__header__``-style entries."""
    from scipy.io import loadmat

    try:
        data = loadmat(path, simplify_cells=True)
    except NotImplementedError as exc:  # v7.3 files are HDF5 containers
        raise ValueError(
            f"'{path.name}' is a MATLAB v7.3 (HDF5) file, which scipy.io.loadmat cannot read; "
            "it has to be re-saved in MATLAB with -v7 or exported to CSV"
        ) from exc
    return {name: value for name, value in data.items() if not name.startswith("__")}


def _matlab_lookup(namespace: dict[str, Any], variable: str, file_name: str) -> Any:
    """The value at a dotted path (``results.tlx``) of a MAT file's variables."""
    value: Any = namespace
    walked: list[str] = []
    for part in str(variable).split("."):
        if not isinstance(value, dict) or part not in value:
            where = ".".join(walked) or "the file"
            available = sorted(value) if isinstance(value, dict) else []
            raise ValueError(f"MATLAB variable '{variable}' not found in '{file_name}': "
                             f"{where} has {available or 'no fields'}")
        value = value[part]
        walked.append(part)
    return value


def _matlab_refuse_opaque(name: str, value: Any) -> None:
    from scipy.io.matlab import MatlabFunction, MatlabOpaque

    if isinstance(value, (MatlabOpaque, MatlabFunction)):
        raise ValueError(
            f"MATLAB variable '{name}' is a MATLAB object or function handle (e.g. a table), "
            "which cannot be read without MATLAB; export it to CSV or save it as a struct"
        )


def _matlab_columns(name: str, value: Any) -> dict[str, Any]:
    """Columns of one struct field or vector: ``{column name: scalar or 1-D values}``."""
    _matlab_refuse_opaque(name, value)
    if isinstance(value, dict):
        columns: dict[str, Any] = {}
        for field, inner in value.items():
            columns.update(_matlab_columns(f"{name}.{field}" if name else field, inner))
        return columns
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        raise ValueError(f"MATLAB field '{name}' is a struct array; read it with variable: {name}")
    if isinstance(value, list):
        value = np.asarray(value, dtype=object)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return {name: value.item()}
        if value.ndim == 1:
            return {name: value}
        if value.ndim == 2:
            return {f"{name}_{j + 1}": value[:, j] for j in range(value.shape[1])}
        raise ValueError(f"MATLAB field '{name}' has {value.ndim} dimensions; read it with "
                         f"variable: {name} and dims: [one name per axis]")
    return {name: value}  # a scalar: number, string or bool


def _matlab_struct_frame(name: str, value: dict[str, Any], file_name: str) -> pd.DataFrame:
    """One struct as a table: equal-length fields side by side, scalars repeated."""
    columns = _matlab_columns("", value)
    lengths = {col: len(v) for col, v in columns.items() if isinstance(v, np.ndarray)}
    rows = set(lengths.values()) - {1}
    if len(rows) > 1:
        detail = ", ".join(f"{col} ({n})" for col, n in lengths.items())
        raise ValueError(f"MATLAB struct '{name or file_name}' in '{file_name}' has fields of different "
                         f"lengths: {detail}; name one with the 'variable' table option")
    n_rows = rows.pop() if rows else 1
    data = {col: (v if isinstance(v, np.ndarray) and len(v) == n_rows else
                  [v.item() if isinstance(v, np.ndarray) else v] * n_rows)
            for col, v in columns.items()}
    return pd.DataFrame(data)


def _matlab_frame(name: str, value: Any, file_name: str, columns: Any = None,
                  dims: list[str] | None = None, namespace: dict[str, Any] | None = None) -> pd.DataFrame:
    """One MATLAB variable as a table (see the section comment above)."""
    _matlab_refuse_opaque(name, value)
    leaf = name.split(".")[-1] if name else "value"
    if dims is not None:
        array = np.asarray(value)
        if not isinstance(dims, list) or array.ndim != len(dims) or array.dtype == object:
            raise ValueError(f"dims for MATLAB variable '{name}' in '{file_name}' must name each of its "
                             f"{array.ndim} axes (shape {array.shape}) of a numeric array, not {dims!r}")
        index = pd.MultiIndex.from_product([range(1, n + 1) for n in array.shape], names=[str(d) for d in dims])
        return pd.DataFrame({leaf: array.reshape(-1)}, index=index).reset_index()
    if columns is not None:
        array = np.asarray(value) if not isinstance(value, dict) else None
        if array is None or array.ndim != 2:
            raise ValueError(f"'columns' applies to a 2-D MATLAB matrix; '{name}' in '{file_name}' is not one")
        names = columns
        if isinstance(columns, str):
            names = _matlab_lookup(namespace or {}, columns, file_name)
        names = [str(n) for n in np.asarray(names, dtype=object).reshape(-1)]
        if len(names) != array.shape[1]:
            raise ValueError(f"'columns' for '{name}' in '{file_name}' names {len(names)} columns; "
                             f"the matrix has {array.shape[1]}")
        return pd.DataFrame(array, columns=names)
    if isinstance(value, dict):
        return _matlab_struct_frame(name, value, file_name)
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        parts = [_matlab_struct_frame(f"{name}({i + 1})", element, file_name)
                 for i, element in enumerate(value)]
        for i, part in enumerate(parts):
            part.insert(0, f"{leaf}_index", i + 1)
        return pd.concat(parts, ignore_index=True)
    array = np.asarray(value, dtype=object if isinstance(value, list) else None)
    if array.ndim <= 1:
        return pd.DataFrame({leaf: array.reshape(-1)})
    if array.ndim == 2:
        return pd.DataFrame(array, columns=[f"{leaf}_{j + 1}" for j in range(array.shape[1])])
    raise ValueError(f"MATLAB variable '{name}' in '{file_name}' has {array.ndim} dimensions; "
                     "name its axes with the 'dims' table option")


@_register_format(".mat")
def _load_matlab(path: Path, variable: str | None = None, columns: Any = None,
                 dims: list[str] | None = None) -> pd.DataFrame:
    """Load one table from a MATLAB .mat file (v4 to v7.2) with ``scipy.io.loadmat``."""
    namespace = _matlab_namespace(path)
    if variable is not None:
        return _matlab_frame(str(variable), _matlab_lookup(namespace, variable, path.name), path.name,
                             columns=columns, dims=dims, namespace=namespace)
    if columns is not None or dims is not None:
        raise ValueError(f"'columns' and 'dims' need the 'variable' they apply to ('{path.name}')")
    if not namespace:
        raise ValueError(f"'{path.name}' holds no MATLAB variables")
    if len(namespace) == 1:
        (name, value), = namespace.items()
        return _matlab_frame(name, value, path.name)
    try:
        return _matlab_struct_frame("", namespace, path.name)
    except ValueError:
        pass
    best: tuple[int, str, pd.DataFrame] | None = None
    for name, value in namespace.items():
        try:
            frame = _matlab_frame(name, value, path.name)
        except ValueError:
            continue
        count = int(frame.notna().sum().sum())
        if best is None or count > best[0]:
            best = (count, name, frame)
    if best is None:
        raise ValueError(f"No MATLAB variable of '{path.name}' reads as a table: {sorted(namespace)}; "
                         "name one with the 'variable' table option")
    logger.info("MAT file '%s': selected variable '%s' (%d non-empty cells). Other variables: %s",
                path.name, best[1], best[0], [n for n in namespace if n != best[1]])
    return best[2]


# ---------------------------------------------------------------------------
# Public load / save / coerce entry points
# ---------------------------------------------------------------------------

_SPREADSHEET_SUFFIXES = {".xlsx", ".xls", ".xlsm", ".ods"}
_TEXT_TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".dat"}
TABLE_OPTION_KEYS = ("sheet", "header", "skip_rows", "pivot", "drop_columns", "rename", *MATLAB_OPTION_KEYS)


def table_options_for(table_options: dict[str, Any] | None, dataset_id: str) -> dict[str, Any]:
    """The manifest's ``table_options`` entry for one file (exact path first, then globs).

    A source can declare, per file path relative to its root (or glob)::

        table_options:
          bvi-auditory-hav.xlsx: {sheet: Results}
          subjective_workload_*.xlsx:
            sheet: NASA_TLX_Data
            header: 1                # 0-based row holding the column names
            pivot: {index: [subject_id, curvature], columns: subscale, values: score}
          survey_export.xlsx:
            skip_rows: [0, 1]        # Qualtrics: question-text and import-id rows
          "**/ObservationsPerEvaluation.csv":
            drop_columns: [MentalLoad]   # a transformed copy of an answer kept elsewhere
          "**/toast_responses.csv":
            rename: {item_1: toast_1, item_2: toast_2}   # sus_responses.csv also has item_1..

    ``rename`` (``{old: new}``, applied after ``pivot`` and ``drop_columns``)
    lets two files of one source share raw column names with different
    meanings: a mapping file applies one target per column name to every file
    of its source, so the mapping names the renamed columns instead.

    ``skip_rows`` lists data rows (0-based, counted from the row below the
    header) that are not records, such as the question-text and import-id
    rows of a Qualtrics export. They are skipped while the file is parsed, so
    the columns get the types of the remaining cells.
    """
    if not table_options:
        return {}
    if dataset_id in table_options:
        return dict(table_options[dataset_id])
    for pattern, options in table_options.items():
        if fnmatch.fnmatch(dataset_id, pattern):
            return dict(options)
    return {}


def _pivot_long_table(df: pd.DataFrame, index: list[str] | str, columns: str, values: str) -> pd.DataFrame:
    """One column per value of *columns* (e.g. one per NASA-TLX subscale).

    ``DataFrame.pivot`` refuses duplicate (index, column) pairs, so a wrong
    index fails loudly instead of silently averaging rows.
    """
    wide = df.pivot(index=index, columns=columns, values=values)
    wide.columns = [str(c) for c in wide.columns]
    return wide.reset_index()


class _SkippedRows:
    """The ``skip_rows`` table option as a pandas ``skiprows`` callable.

    Skips the listed data rows (0-based, counted from the row below the
    header) while pandas parses the file, so a column whose only text cells
    sit in those rows is read as numbers. Records the rows it met; ``check``
    raises for listed rows the file does not have.
    """

    def __init__(self, rows: Any, header: Any, file_name: str) -> None:
        rows = [] if rows is None else rows
        if not isinstance(rows, list) or not all(
            isinstance(row, int) and not isinstance(row, bool) and row >= 0 for row in rows
        ):
            raise ValueError(f"skip_rows for '{file_name}' must be a list of 0-based row numbers, not {rows!r}")
        if header is None:
            first_data_row = 0
        elif isinstance(header, int) and not isinstance(header, bool):
            first_data_row = header + 1
        elif rows:
            raise ValueError(f"skip_rows for '{file_name}' needs one header row, not {header!r}")
        else:
            first_data_row = 0
        self.rows = frozenset(rows)
        self._first_data_row = first_data_row
        self._file_name = file_name
        self._met: set[int] = set()

    def __call__(self, file_row: int) -> bool:
        row = file_row - self._first_data_row
        if row in self.rows:
            self._met.add(row)
            return True
        return False

    def check(self) -> None:
        missing = sorted(self.rows - self._met)
        if missing:
            raise ValueError(f"skip_rows {missing} not found in '{self._file_name}'")


def load_input_file(
    input_path: str, options: dict[str, Any] | None = None, *, allow_pickle: bool = False
) -> pd.DataFrame:
    """Load tabular input data using the format registry.

    Format is determined first by file extension (registry lookup), then
    by content-based magic-byte sniffing if the extension is unrecognised.
    *options* are a file's ``table_options`` (``table_options_for``): for a
    spreadsheet the ``sheet`` to read (instead of the most data-rich one) and
    the 0-based ``header`` row; for a spreadsheet or delimited text file the
    ``skip_rows`` below the header that hold no records (0-based data rows,
    e.g. ``[0, 1]`` for a Qualtrics export's question texts and import ids);
    for a MATLAB .mat file the ``variable`` to read (a dotted path into
    structs), the ``columns`` of a matrix and the ``dims`` that turn an array
    into a long table (see ``_load_matlab``);
    for any table a long-to-wide ``pivot``, ``drop_columns`` (columns that
    must not be read as data, e.g. an optimizer's inverted copy of a
    questionnaire answer) and ``rename`` (``{old: new}`` column names, e.g.
    to tell apart two files' ``item_1``). Columns named by ``drop_columns``
    or ``rename``, and rows named by ``skip_rows``, must exist, so a changed
    file fails loudly instead of passing unmapped.

    ``.pkl``/``.pickle`` files raise ``PickleRefusedError`` unless
    *allow_pickle* is true: unpickling runs code from the file, so allow it
    only for files you created yourself.
    """
    path = Path(input_path)
    suffix = path.suffix.lower()
    options = dict(options or {})
    unknown = sorted(set(options) - set(TABLE_OPTION_KEYS))
    if unknown:
        raise ValueError(f"Unknown table option(s) {unknown} for '{path.name}'; allowed: {list(TABLE_OPTION_KEYS)}")
    matlab = {key: options[key] for key in MATLAB_OPTION_KEYS if key in options}
    if matlab and suffix not in MATLAB_SUFFIXES:
        raise ValueError(f"{sorted(matlab)} apply to MATLAB .mat files, not '{path.name}'")
    header = options.get("header", 0)
    skipped = _SkippedRows(options.get("skip_rows"), header, path.name)
    read_kwargs: dict[str, Any] = {"skiprows": skipped} if skipped.rows else {}
    if "sheet" in options or "header" in options:
        if suffix not in _SPREADSHEET_SUFFIXES:
            raise ValueError(f"'sheet' and 'header' apply to spreadsheets, not '{path.name}'")
        df = pd.read_excel(path, sheet_name=options.get("sheet", 0), header=header, **read_kwargs)
    elif read_kwargs and suffix in _SPREADSHEET_SUFFIXES:
        df = _load_excel(path, **read_kwargs)
    elif read_kwargs and suffix in _TEXT_TABLE_SUFFIXES:
        df = _load_text_table(path, **read_kwargs)
    elif read_kwargs:
        raise ValueError(f"'skip_rows' applies to spreadsheets and delimited text files, not '{path.name}'")
    elif suffix in MATLAB_SUFFIXES:
        df = _load_matlab(path, **matlab)
    elif suffix in PICKLE_SUFFIXES:
        if allow_pickle is not True:  # a truthy non-boolean never opts in
            raise PickleRefusedError(pickle_refusal_message(path))
        df = _unpickle_table(path)
    else:
        df = _load_with_registry(path)
    skipped.check()
    if options.get("pivot"):
        df = _pivot_long_table(df, **options["pivot"])
    drop = [str(c) for c in options.get("drop_columns") or []]
    if drop:
        missing = [c for c in drop if c not in {str(col).strip() for col in df.columns}]
        if missing:
            raise ValueError(f"drop_columns {missing} not found in '{path.name}'")
        df = df.loc[:, [str(col).strip() not in drop for col in df.columns]]
    if options.get("rename"):
        df = _rename_columns(df, options["rename"], path.name)
    return df


def _rename_columns(df: pd.DataFrame, rename: Any, file_name: str) -> pd.DataFrame:
    """Apply a ``rename`` table option, matching headers as the batch does (stripped)."""
    if not isinstance(rename, dict):
        raise ValueError(f"rename for '{file_name}' must map old to new column names, not {type(rename).__name__}")
    mapping = {str(old).strip(): str(new).strip() for old, new in rename.items()}
    present = [str(col).strip() for col in df.columns]
    missing = [old for old in mapping if old not in present]
    if missing:
        raise ValueError(f"rename {missing} not found in '{file_name}'")
    renamed = [mapping.get(name, name) for name in present]
    clashes = sorted({name for name in renamed if renamed.count(name) > 1} & set(mapping.values()))
    if clashes:
        raise ValueError(f"rename would give '{file_name}' duplicate columns {clashes}")
    return df.set_axis([mapping.get(name, col) for name, col in zip(present, df.columns)], axis=1)


def _load_with_registry(path: Path) -> pd.DataFrame:
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
        # Formats contributed by installed plugins (opendv.loaders entry points).
        from opendv.plugins import load_plugins

        load_plugins()
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


def _write_parquet(df: pd.DataFrame, output_file: Path) -> None:
    """Write Parquet with pyarrow; object columns mixing types become strings
    (Parquet columns have one type) and are logged."""
    try:
        df.to_parquet(output_file, index=False)
        return
    except (TypeError, ValueError, ImportError) as exc:
        if isinstance(exc, ImportError):
            raise ImportError("Writing Parquet requires pyarrow: pip install pyarrow") from exc
        first_error = exc
    out = df.copy()
    mixed = [
        col for col in out.columns
        if out[col].dtype == object and out[col].dropna().map(type).nunique() > 1
    ]
    for col in mixed:
        out[col] = out[col].map(lambda v: v if pd.isna(v) else str(v)).astype("string")
    if not mixed:
        raise first_error
    logger.info("Parquet: stored mixed-type columns %s as strings", mixed)
    out.to_parquet(output_file, index=False)


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
        except ValueError as exc:
            # pandas 2.x removed the xlwt-based .xls writer.
            raise ValueError(
                f"Cannot write '{output_file.name}': pandas no longer supports the "
                "legacy .xls format. Use a .xlsx output path instead."
            ) from exc
        return
    if suffix == ".csv":
        df.to_csv(output_file, index=False)
        return
    if suffix in {".pkl", ".pickle"}:
        df.to_pickle(output_file)
        return
    if suffix == ".parquet":
        _write_parquet(df, output_file)
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
    "_detect_decimal",
    "_load_text_table",
    "_load_excel",
    "_load_pickle",
    "_unpickle_table",
    "PICKLE_SUFFIXES",
    "PickleRefusedError",
    "pickle_refusal_message",
    "_load_parquet",
    "_load_spss",
    "_load_stata",
    "_load_feather",
    "_load_json",
    "_load_jsonlines",
    "_load_matlab",
    "MATLAB_SUFFIXES",
    "MATLAB_OPTION_KEYS",
    "load_input_file",
    "table_options_for",
    "TABLE_OPTION_KEYS",
    "_coerce_pickled_payload_to_dataframe",
    "save_output_file",
]
