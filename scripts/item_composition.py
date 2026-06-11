"""Per-row composition of multi-item scales (NASA-TLX, SUS, TiA, ...).

After ``scripts.convert_dv.standardize_columns`` renames raw columns to canonical
DV ids, several originals (``Trust1``, ``Trust2``, ...) often collapse onto the
same canonical (``trust_rating``). The dedup pass in :mod:`scripts.convert_dv`
keeps the DataFrame indexable by suffixing duplicates as ``__dup_N``, but
statistically these are not independent observations — they are items of a
single construct measured on the same participant in the same trial. The right
thing to do is compose them into a per-row composite score (typically the mean
across items, after reverse-coding any negatively-keyed items) and treat that
composite as the one observation of the construct for that row.

This module implements Tier 1 of that strategy:

* automatic group discovery from the ``__dup_N`` suffix introduced by
  :func:`scripts.convert_dv.standardize_columns`;
* a Cronbach's α reliability estimate per group, with a sign-of-correlation
  reverse-coding heuristic applied before α is computed;
* a single composite column per group (named with the base canonical) plus the
  item columns renamed to ``<canonical>__item_N`` for traceability;
* a list of composition records that the orchestrator can persist alongside
  the run artefacts.

Schema-declared scale definitions (Tier 2 — explicit ``items``,
``reverse_items``, and ``score_transform``) are intentionally out of scope here;
this module handles the unannotated common case and emits the metadata
downstream tooling needs to flag low-reliability composites.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from scripts.convert_dv import strip_duplicate_suffix

logger = logging.getLogger(__name__)

ITEM_COLUMN_SUFFIX = "__item_"

# Dataset types where multi-item composition makes statistical sense.
# Sensor streams and detection logs have repeated "Trust1, Trust2, ..." style
# columns that are arms/channels, not Likert items, and must not be averaged.
DEFAULT_ELIGIBLE_DATASET_TYPES: tuple[str, ...] = ("questionnaire", "results_table")

# Composing requires enough rows for α to be meaningful. With <5 complete rows
# the reverse-coding heuristic is essentially noise, so we skip composition
# and leave the items as separate __dup_N columns.
MIN_COMPLETE_ROWS_FOR_COMPOSITION = 5

# Items whose Pearson correlation with the mean of the others is below this
# threshold are flipped before α is computed. -0.05 (not 0) gives a small
# margin so weakly-noisy items don't get spuriously flipped.
REVERSE_CODING_NEGATIVE_THRESHOLD = -0.05


def is_item_suffixed(column_name: Any) -> bool:
    """Return True when the column name carries the ``__item_N`` suffix."""
    if not isinstance(column_name, str):
        return False
    idx = column_name.rfind(ITEM_COLUMN_SUFFIX)
    if idx == -1:
        return False
    tail = column_name[idx + len(ITEM_COLUMN_SUFFIX):]
    return tail.isdigit()


def strip_item_suffix(column_name: Any) -> str:
    """Return the base canonical for an ``__item_N`` suffixed column."""
    if not isinstance(column_name, str):
        return column_name
    if is_item_suffixed(column_name):
        return column_name[: column_name.rfind(ITEM_COLUMN_SUFFIX)]
    return column_name


def compute_cronbach_alpha(item_matrix: pd.DataFrame) -> float:
    """Compute Cronbach's α on a numeric item DataFrame (one column per item).

    Returns NaN when there are fewer than two items or the row-sum variance is
    zero (perfectly collinear items, or a degenerate single-value column).
    """
    k = item_matrix.shape[1]
    if k < 2:
        return float("nan")
    item_vars = item_matrix.var(ddof=1, axis=0).sum()
    total_var = float(item_matrix.sum(axis=1).var(ddof=1))
    if not np.isfinite(total_var) or total_var <= 0:
        return float("nan")
    return float((k / (k - 1)) * (1.0 - item_vars / total_var))


def detect_reverse_coded_items(item_matrix: pd.DataFrame) -> list[str]:
    """Return columns that are reverse-keyed relative to the scale's common factor.

    Anchored heuristic: build the inter-item correlation matrix, pick the item
    most representative of the dominant common factor (largest total absolute
    correlation with the others), and flag every item that correlates
    negatively with that anchor.

    This replaces the earlier "correlate each item with the mean of the
    others" approach, which was contaminated when a scale contained *two or
    more* reverse-keyed items: the un-flipped reverse items pulled the
    mean-of-others toward them and diluted correlations below the detection
    threshold. Correlating against a single clean anchor avoids that coupling,
    is deterministic, and converges by construction. The anchor's own polarity
    is irrelevant — only internal consistency matters for α and the composite.
    """
    cols = list(item_matrix.columns)
    if len(cols) < 2:
        return []

    try:
        corr = item_matrix.corr()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Reverse-coding correlation matrix failed: %s", exc)
        return []

    abs_corr_sums = corr.abs().sum(axis=1)
    if abs_corr_sums.empty or bool(abs_corr_sums.isna().all()):
        return []
    anchor = abs_corr_sums.idxmax()

    reversed_cols: list[str] = []
    for col in cols:
        if col == anchor:
            continue
        r = corr.at[col, anchor]
        if pd.notna(r) and r < REVERSE_CODING_NEGATIVE_THRESHOLD:
            reversed_cols.append(col)
    return reversed_cols


def apply_reverse_coding(
    item_matrix: pd.DataFrame,
    reversed_cols: list[str],
    scale_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Flip reverse-keyed items by ``(lo + hi) - value``.

    ``scale_range`` overrides the observed (min, max) per column — pass it when
    the response scale is known from the schema so partially-observed ranges
    don't bias the flip.
    """
    if not reversed_cols:
        return item_matrix
    out = item_matrix.copy()
    for col in reversed_cols:
        series = out[col]
        if scale_range is not None:
            lo, hi = scale_range
        else:
            lo = float(series.min(skipna=True))
            hi = float(series.max(skipna=True))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            # Single observed value — flipping is a no-op; record the request
            # but leave the column alone.
            continue
        out[col] = (lo + hi) - series
    return out


def _group_columns_by_base_canonical(df: pd.DataFrame) -> dict[str, list[str]]:
    """Group DataFrame columns by their stripped base canonical name.

    Includes only groups with at least two columns (i.e. the base name plus at
    least one ``__dup_N`` sibling, or two ``__dup_N`` siblings). Columns whose
    base name does not appear with ``__dup_N`` are returned as singleton groups
    and excluded by the caller — there is nothing to compose.
    """
    groups: dict[str, list[str]] = {}
    for col in df.columns:
        base = strip_duplicate_suffix(str(col))
        groups.setdefault(base, []).append(str(col))
    return {base: cols for base, cols in groups.items() if len(cols) > 1}


def _apply_declared_scale(
    df: pd.DataFrame,
    scale: dict[str, Any],
    canonical_to_originals: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Compose a single declared scale onto *df*.

    Declared scales (Tier 2) bypass the auto-detect alpha gate — the schema
    author has asserted these items belong together. α is still recorded for
    quality monitoring, but composition proceeds even when α is negative.

    The scale dict accepts:
        canonical            — required, canonical DV id of the composite
        aliases              — required, list of original column names that are items
        reverse_aliases      — optional, items to flip before averaging
        scale_range          — optional [lo, hi] applied to reverse-coding

    Returns the modified DataFrame and a record dict, or ``(df, None)`` when
    the scale's aliases don't match the dataset.
    """
    canonical = scale.get("canonical")
    declared_aliases = scale.get("aliases") or []
    reverse_aliases = set(scale.get("reverse_aliases") or [])
    scale_range = scale.get("scale_range")
    if scale_range is not None and (
        not isinstance(scale_range, (list, tuple)) or len(scale_range) != 2
    ):
        scale_range = None
    if scale_range is not None:
        scale_range = (float(scale_range[0]), float(scale_range[1]))

    if not isinstance(canonical, str) or not declared_aliases:
        return df, None

    originals_for_canonical = canonical_to_originals.get(canonical, [])
    if not originals_for_canonical:
        return df, None

    # Match declared aliases to df columns by ordinal position within the
    # canonical group. The first occurrence keeps the bare canonical name;
    # subsequent occurrences carry the __dup_N suffix from standardize_columns.
    df_pairs: list[tuple[str, str]] = []
    reverse_df_cols: list[str] = []
    for idx, original in enumerate(originals_for_canonical):
        if original not in declared_aliases:
            continue
        df_col = canonical if idx == 0 else f"{canonical}__dup_{idx + 1}"
        if df_col not in df.columns:
            continue
        df_pairs.append((original, df_col))
        if original in reverse_aliases:
            reverse_df_cols.append(df_col)

    if len(df_pairs) < 2:
        return df, None

    item_cols = [df_col for _, df_col in df_pairs]
    item_matrix = df[item_cols].apply(pd.to_numeric, errors="coerce")
    corrected = apply_reverse_coding(item_matrix, reverse_df_cols, scale_range)

    complete_rows = corrected.dropna()
    alpha = compute_cronbach_alpha(complete_rows) if len(complete_rows) >= 2 else float("nan")

    composite = corrected.mean(axis=1, skipna=True)
    rename_map = {
        df_col: f"{canonical}{ITEM_COLUMN_SUFFIX}{i + 1}"
        for i, (_, df_col) in enumerate(df_pairs)
    }
    df_out = df.rename(columns=rename_map)
    df_out[canonical] = composite

    record = {
        "canonical": canonical,
        "items": [orig for orig, _ in df_pairs],
        "n_items": len(df_pairs),
        "n_complete_rows": int(len(complete_rows)),
        "cronbach_alpha": float(alpha) if np.isfinite(alpha) else None,
        "reverse_coded_items": list(reverse_aliases & {orig for orig, _ in df_pairs}),
        "scale_range": list(scale_range) if scale_range else None,
        "decision": "composed_declared_scale",
        "composite_column": canonical,
        "renamed_items": [rename_map[df_col] for _, df_col in df_pairs],
    }
    return df_out, record


def compose_item_groups(
    df: pd.DataFrame,
    dataset_type: str,
    eligible_dataset_types: tuple[str, ...] = DEFAULT_ELIGIBLE_DATASET_TYPES,
    min_alpha_to_compose: float = 0.0,
    declared_scales: list[dict[str, Any]] | None = None,
    canonical_to_originals: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Compose ``__dup_N`` groups in *df* into per-row composite scores.

    For each group of columns sharing a base canonical:

    1. Cast to numeric (non-numeric values become NaN).
    2. Detect reverse-keyed items via correlation with the mean of the others.
    3. Apply the flips.
    4. Compute Cronbach's α on rows where all items are present.
    5. When α is finite and ≥ ``min_alpha_to_compose`` (default 0.0, i.e. any
       non-negative α including 0 — anything positive means the items at least
       share a common direction), build a per-row mean composite, write it as
       the base canonical column, and rename the original items to
       ``<canonical>__item_N``.
    6. When α is NaN or negative (items contradict each other — likely a
       grouping mistake), leave the columns as-is so they appear as separate
       observations downstream.

    Returns ``(df_out, records)`` where *records* is a list of dicts describing
    each group's outcome (composed/skipped, α, item names, reverse-coded items,
    complete-row count).
    """
    records: list[dict[str, Any]] = []
    if dataset_type not in eligible_dataset_types:
        return df, records

    # Shallow copy: composition only renames columns (rename returns a new
    # frame) and appends composite columns — existing data blocks are never
    # written in place, so sharing them with the caller's frame is safe and
    # avoids duplicating large questionnaires in memory.
    df_out = df.copy(deep=False)
    canonicals_handled_by_declared: set[str] = set()

    # Phase 1: process declared scales (Tier 2). These bypass the alpha gate
    # because the schema author has authoritatively asserted the items belong
    # together. After application, the involved canonical is excluded from the
    # auto-detect pass below so we don't double-process.
    if declared_scales and canonical_to_originals:
        for scale in declared_scales:
            if not isinstance(scale, dict):
                continue
            df_out, scale_record = _apply_declared_scale(df_out, scale, canonical_to_originals)
            if scale_record is None:
                continue
            records.append(scale_record)
            canonicals_handled_by_declared.add(str(scale_record["canonical"]))

    groups = _group_columns_by_base_canonical(df_out)
    if not groups:
        return df_out, records

    for base_canonical, item_cols in groups.items():
        if base_canonical in canonicals_handled_by_declared:
            continue
        # Cast a defensive copy to numeric for the reliability analysis. The
        # full-row composite is computed against the same numeric coercion so
        # rows with sporadic NaN items still contribute via skipna=True.
        full_numeric = df_out[item_cols].apply(pd.to_numeric, errors="coerce")
        complete_rows = full_numeric.dropna()
        n_complete = int(len(complete_rows))

        record: dict[str, Any] = {
            "canonical": base_canonical,
            "items": list(item_cols),
            "n_items": len(item_cols),
            "n_complete_rows": n_complete,
        }

        if n_complete < MIN_COMPLETE_ROWS_FOR_COMPOSITION:
            record["decision"] = "skip_insufficient_rows"
            records.append(record)
            continue

        reversed_cols = detect_reverse_coded_items(complete_rows)
        complete_corrected = apply_reverse_coding(complete_rows, reversed_cols)
        alpha = compute_cronbach_alpha(complete_corrected)

        record["cronbach_alpha"] = (
            float(alpha) if isinstance(alpha, float) and np.isfinite(alpha) else None
        )
        record["reverse_coded_items"] = reversed_cols

        if record["cronbach_alpha"] is None or record["cronbach_alpha"] < min_alpha_to_compose:
            record["decision"] = "skip_low_alpha"
            records.append(record)
            continue

        full_corrected = apply_reverse_coding(full_numeric, reversed_cols)
        composite = full_corrected.mean(axis=1, skipna=True)

        rename_map = {
            col: f"{base_canonical}{ITEM_COLUMN_SUFFIX}{i + 1}"
            for i, col in enumerate(item_cols)
        }
        df_out = df_out.rename(columns=rename_map)
        df_out[base_canonical] = composite

        record["decision"] = "composed"
        record["composite_column"] = base_canonical
        record["renamed_items"] = [rename_map[col] for col in item_cols]
        records.append(record)

    return df_out, records


__all__ = [
    "ITEM_COLUMN_SUFFIX",
    "DEFAULT_ELIGIBLE_DATASET_TYPES",
    "MIN_COMPLETE_ROWS_FOR_COMPOSITION",
    "is_item_suffixed",
    "strip_item_suffix",
    "compute_cronbach_alpha",
    "detect_reverse_coded_items",
    "apply_reverse_coding",
    "compose_item_groups",
]
