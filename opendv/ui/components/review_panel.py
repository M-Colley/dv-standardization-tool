"""
review_panel.py

Streamlit tab for reviewing LLM-deduced and unmapped column mappings
(scripts/mapping_review.py): browse the queue, record decisions in an editable
table, and write them back into the sources' mapping YAMLs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from opendv.mapping_review import (
    DECISIONS,
    DEFAULT_DV_SCHEMA,
    DEFAULT_LOG,
    STATUS_PRIORITY,
    _dv_info,
    apply_decisions,
    build_review_queue,
    decisions_from_queue,
    review_summary,
)
from opendv._paths import REPO_ROOT

SHOWN_COLUMNS = [
    "decision", "target", "note", "status", "source_id", "raw_column", "suggestion", "suggestion_range",
    "outside_range", "n", "mean", "sd", "min", "max", "candidates", "rejection_reason", "datasets",
]


@st.cache_data(show_spinner="Building the review queue...")
def _cached_queue(batch_dir: str, meta_view_mtime: float) -> pd.DataFrame:
    return build_review_queue(Path(batch_dir))


def render_review_tab() -> None:
    st.subheader("Review column mappings")
    st.markdown(
        "LLM-deduced mappings are right only about six times in ten, so check them before relying on them. "
        "Decisions are written into each source's mapping YAML and logged with your name; the next batch run "
        "applies them like any curated mapping. `accept` keeps the LLM suggestion, `map` uses the DV in "
        "*target*, `non_dv` marks the column as not a dependent variable, `study_specific` keeps it as an "
        "outcome without a schema DV."
    )
    col1, col2, col3 = st.columns(3)
    batch_dir = col1.text_input(
        "Batch output directory", value=str(REPO_ROOT / "data" / "processed" / "catalog_meta_analysis")
    )
    manifest = col2.text_input("Manifest", value=str(REPO_ROOT / "sources_manifest_no4tu.yaml"))
    reviewer = col3.text_input("Your name (recorded with every decision)")

    meta_view = Path(batch_dir) / "meta_view.csv"
    if not meta_view.is_file():
        st.info("Point to a batch output directory that contains meta_view.csv.")
        return
    queue = _cached_queue(batch_dir, meta_view.stat().st_mtime)
    if queue.empty:
        st.success("Nothing to review.")
        return

    statuses = st.multiselect(
        "Show", options=list(STATUS_PRIORITY), default=["llm_accepted", "llm_rejected", "unmapped"],
        help="llm_accepted rows are used in the analysis, so review them first.",
    )
    view = queue[queue["status"].isin(statuses)][SHOWN_COLUMNS]
    st.caption(f"{len(view)} of {len(queue)} columns; most urgent first.")
    edited = st.data_editor(
        view,
        key="review_editor",
        width="stretch",
        hide_index=True,
        disabled=[c for c in SHOWN_COLUMNS if c not in ("decision", "target", "note")],
        column_config={
            "decision": st.column_config.SelectboxColumn("decision", options=list(DECISIONS)),
            "target": st.column_config.SelectboxColumn(
                "target", options=sorted(_dv_info(DEFAULT_DV_SCHEMA)), help="DV id for 'map'"
            ),
            "outside_range": st.column_config.CheckboxColumn(
                "outside range", help="Observed values cannot lie on the suggested DV's scale without rescaling"
            ),
        },
    )
    decisions = decisions_from_queue(edited)
    st.caption(f"{len(decisions)} decision(s) recorded in the table.")
    if st.button("Write decisions to the mapping files", disabled=not decisions or not reviewer.strip()):
        try:
            result = apply_decisions(decisions, Path(manifest), reviewer, batch_dir=Path(batch_dir))
        except (ValueError, FileNotFoundError) as exc:
            st.error(f"Nothing was written: {exc}")
        else:
            st.success(
                f"Wrote {result['decisions']} decision(s) to {', '.join(result['written'])}"
                + ("; the manifest now points at the new mapping files." if result["manifest_updated"] else ".")
                + " Rerun the batch to apply them."
            )
    summary = review_summary(DEFAULT_LOG)
    if summary.get("decisions"):
        st.markdown("**Review log**")
        st.json(summary)
