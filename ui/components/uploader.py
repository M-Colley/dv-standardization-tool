"""Reusable upload helpers for the OpenDV-HCI Streamlit interface."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st


def upload_tabular_file(
    label: str,
    key: str,
    help_text: str | None = None,
):
    return st.file_uploader(
        label,
        type=["csv", "xlsx", "xls"],
        key=key,
        help=help_text,
    )


def list_excel_sheets(uploaded_file) -> list[str]:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".xlsx", ".xls"}:
        return []
    workbook = pd.ExcelFile(io.BytesIO(uploaded_file.getvalue()))
    return list(workbook.sheet_names)


def load_uploaded_table(uploaded_file, sheet_name: str | int = 0) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    raw = uploaded_file.getvalue()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name)

    # Mirror the CLI loader's robustness: detect encoding and delimiter so a
    # semicolon-separated or latin-1 CSV parses the same here as in
    # scripts/convert_dv.py instead of collapsing into a single column.
    from scripts.data_loaders import _detect_delimiter
    from scripts.encoding_utils import detect_text_encoding

    encoding = detect_text_encoding(raw)
    sample = raw[:16384].decode(encoding, errors="replace")
    delimiter = _detect_delimiter(sample, prefer=",")
    return pd.read_csv(io.BytesIO(raw), sep=delimiter, encoding=encoding)


def upload_csv():
    uploaded_file = upload_tabular_file(
        "Choose a CSV file",
        key="legacy_csv_uploader",
        help_text="Legacy CSV-only uploader.",
    )
    if uploaded_file is None:
        return None
    try:
        return load_uploaded_table(uploaded_file)
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Failed to read CSV: {exc}")
        return None
