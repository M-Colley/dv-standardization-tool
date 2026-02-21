"""Streamlit UI for OpenDV-HCI standardization with mapping diagnostics."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.convert_dv import identify_unmapped_columns, standardize_columns

st.set_page_config(page_title="OpenDV-HCI Tool", layout="wide")

st.title("OpenDV-HCI: Dependent Variable Standardization Tool")


def _load_uploaded_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


def _build_mapping(schema: dict) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for dv in schema.get("dvs", []):
        canonical = dv.get("id")
        if not canonical:
            continue
        mapping[canonical] = canonical
        mapping[canonical.lower()] = canonical
        for alias in dv.get("aliases", []):
            mapping[alias] = canonical
            mapping[alias.lower()] = canonical
    return mapping


uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
if uploaded_file:
    df_raw = _load_uploaded_file(uploaded_file)
    st.subheader("Original Column Names")
    st.dataframe(pd.DataFrame(df_raw.columns, columns=["Column Names"]))

    try:
        with open("schemas/standard_dv_mapping.yaml", "r", encoding="utf-8") as f:
            schema = yaml.safe_load(f)
    except FileNotFoundError:
        st.error("Schema file not found. Please ensure schemas/standard_dv_mapping.yaml exists.")
        st.stop()

    mapping = _build_mapping(schema)
    df_clean = standardize_columns(df_raw.copy(), mapping)

    unknown_columns = identify_unmapped_columns([str(c) for c in df_raw.columns], mapping)
    total_columns = len(df_raw.columns)
    mapped_columns = total_columns - len(unknown_columns)
    mapping_rate = (mapped_columns / total_columns) if total_columns else 0.0

    st.subheader("Mapping Quality")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Mapped Columns", mapped_columns)
    metric_col2.metric("Unknown Columns", len(unknown_columns))
    metric_col3.metric("Mapping Rate", f"{mapping_rate:.0%}")

    if unknown_columns:
        st.warning("Unmapped aliases detected: " + ", ".join(unknown_columns))

    st.subheader("Standardized Column Names")
    st.dataframe(pd.DataFrame(df_clean.columns, columns=["Column Names"]))

    st.subheader("Download")
    csv_download = df_clean.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Standardized CSV",
        data=csv_download,
        file_name=f"{Path(uploaded_file.name).stem}_standardized.csv",
        mime="text/csv",
    )

    xlsx_buffer = io.BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
        df_clean.to_excel(writer, index=False, sheet_name="standardized")
    st.download_button(
        "Download Standardized Excel",
        data=xlsx_buffer.getvalue(),
        file_name=f"{Path(uploaded_file.name).stem}_standardized.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.markdown("---")
st.markdown("Upload a dataset to standardize DV aliases and inspect mapping quality before export.")
