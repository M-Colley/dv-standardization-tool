"""LLM-driven mapping augmentation coordinator.

This module owns the bridge between the batch pipeline and
``scripts.llm_utils``. It decides which unknown column aliases are worth
asking the local LLM about, ranks the candidate canonical DV names by
fuzzy similarity, calls into ``deduce_standard_name_with_local_llm``,
caches the answer per-source, and aggregates the resulting deductions
into a serializable summary the orchestrator can dump alongside the
batch artefact.

Like ``scripts.archive_utils`` and ``scripts.http_utils`` this module is
deliberately stdlib-plus-llm-utils: it imports nothing from
``run_batch_standardization`` so it can be unit tested without booting
the full pipeline. ``run_batch_standardization`` re-exports every
public name so existing tests and callers continue to work unchanged.

Design notes:

* ``_score_alias_match`` falls back from rapidfuzz to difflib when
  rapidfuzz is unavailable or raises. This keeps the pipeline functional
  on minimal CI installs without rapidfuzz wheels and is logged at
  DEBUG so post-mortem debugging can still recover the failure.
* ``_select_llm_candidate_shortlist`` keeps at most ``max_candidates``
  canonicals in the prompt. Past 8 the LLM signal-to-noise drops fast
  for short alias strings.
* ``_augment_mapping_with_llm_deductions`` accepts an optional
  ``inference_cache`` so multiple datasets within the same source share
  a single LLM round-trip per alias.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from scripts.convert_dv import identify_unmapped_columns
from scripts.llm_utils import deduce_standard_name_with_local_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alias scoring + candidate shortlist
# ---------------------------------------------------------------------------

def _score_alias_match(raw_column_name: str, alias: str) -> float:
    """Score how similar two column names are on a 0-100 scale.

    Uses rapidfuzz when available (max of three measures: ratio, WRatio,
    token_sort_ratio) and falls back to difflib.SequenceMatcher when not.
    Returns 0.0 for empty inputs and for very-short aliases (< 4 chars)
    that don't appear as a substring of the raw column name — short
    aliases like ``u1`` or ``sa`` are too noisy for fuzzy matching.
    """
    normalized_raw = re.sub(r"[^a-z0-9]+", " ", str(raw_column_name).strip().lower()).strip()
    normalized_alias = re.sub(r"[^a-z0-9]+", " ", str(alias).strip().lower()).strip()
    if not normalized_raw or not normalized_alias:
        return 0.0

    raw_compact = normalized_raw.replace(" ", "")
    alias_compact = normalized_alias.replace(" ", "")
    # Very short aliases (e.g. "u1", "sa", "eda") create noisy fuzzy matches.
    if len(alias_compact) < 4 and alias_compact not in raw_compact:
        return 0.0

    try:
        from rapidfuzz import fuzz  # type: ignore

        return float(
            max(
                fuzz.ratio(normalized_raw, normalized_alias),
                fuzz.WRatio(normalized_raw, normalized_alias),
                fuzz.token_sort_ratio(normalized_raw, normalized_alias),
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "rapidfuzz unavailable or scoring failed (%s); falling back to difflib.",
            exc,
        )
        from difflib import SequenceMatcher

        return SequenceMatcher(None, normalized_raw, normalized_alias).ratio() * 100.0


def _select_llm_candidate_shortlist(
    mapping: dict[str, str],
    raw_column_name: str,
    max_candidates: int = 8,
) -> tuple[list[str], float]:
    """Rank candidate canonical DV names by best fuzzy alias score.

    For each canonical in ``mapping``, take the best matching alias as
    its representative score, then return the top-N canonicals plus the
    very top score. Used by the augmenter to (a) skip LLM calls when no
    alias is even remotely close and (b) keep the prompt short.
    """
    candidate_scores: dict[str, float] = {}
    for alias, canonical in mapping.items():
        if not isinstance(alias, str) or not isinstance(canonical, str):
            continue
        score = _score_alias_match(raw_column_name, alias)
        if score <= candidate_scores.get(canonical, 0.0):
            continue
        candidate_scores[canonical] = score

    ranked = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))
    return [canonical for canonical, _ in ranked[:max_candidates]], (ranked[0][1] if ranked else 0.0)


# ---------------------------------------------------------------------------
# LLM augmentation entry point
# ---------------------------------------------------------------------------

def _augment_mapping_with_llm_deductions(
    mapping: dict[str, str],
    columns: list[str],
    source_root: Path,
    preferred_models: list[str] | None = None,
    inference_cache: dict[str, str | None] | None = None,
    repository_context: str | None = None,
    min_attempt_score: float = 65.0,
) -> dict[str, str]:
    """Attempt local-LLM alias deduction for unknown columns.

    This is especially useful for repositories that do not provide a
    custom mapping YAML file. ``inference_cache`` is keyed by the
    lowercased alias and is populated so subsequent datasets in the same
    source skip the LLM call.
    """
    augmented = dict(mapping)

    unknown = identify_unmapped_columns(columns, augmented)
    for alias in unknown:
        alias_key = str(alias).strip().lower()
        inferred: str | None
        if inference_cache is not None and alias_key in inference_cache:
            inferred = inference_cache[alias_key]
        else:
            candidate_shortlist, top_score = _select_llm_candidate_shortlist(mapping, str(alias))
            if not candidate_shortlist or top_score < min_attempt_score:
                inferred = None
                if inference_cache is not None:
                    inference_cache[alias_key] = inferred
                continue
            inferred = deduce_standard_name_with_local_llm(
                raw_column_name=str(alias),
                canonical_candidates=candidate_shortlist,
                source_root=source_root,
                preferred_models=preferred_models,
                repository_context=repository_context,
            )
            if inference_cache is not None:
                inference_cache[alias_key] = inferred
        if inferred:
            augmented[str(alias)] = inferred
            augmented[str(alias).lower()] = inferred

    return augmented


# ---------------------------------------------------------------------------
# Per-run deduction aggregation
# ---------------------------------------------------------------------------

def _collect_llm_deduction(
    deductions_by_key: dict[tuple[str, str], dict[str, Any]],
    source_id: str,
    dataset_id: str,
    alias: str,
    canonical_dv: str,
) -> None:
    """Record an LLM-derived (alias -> canonical_dv) mapping for the run.

    Deductions are deduped on (source_id, lowercased alias) so the same
    alias appearing in multiple datasets is collapsed into a single
    entry whose ``datasets`` list grows as we go.
    """
    alias_text = str(alias).strip()
    if not alias_text:
        return

    key = (source_id, alias_text.lower())
    entry = deductions_by_key.get(key)
    if entry is None:
        entry = {
            "source_id": source_id,
            "alias": alias_text,
            "canonical_dv": canonical_dv,
            "datasets": [],
        }
        deductions_by_key[key] = entry

    datasets = entry["datasets"]
    if dataset_id not in datasets:
        datasets.append(dataset_id)


def _finalize_llm_deductions(
    deductions_by_key: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sort the dataset list within each entry and the entries by source/alias."""
    finalized: list[dict[str, Any]] = []
    for entry in deductions_by_key.values():
        finalized.append(
            {
                **entry,
                "datasets": sorted(entry["datasets"]),
            }
        )
    return sorted(finalized, key=lambda item: (item["source_id"], item["alias"].lower()))


def _build_llm_deduction_log_lines(llm_deductions: list[dict[str, Any]]) -> list[str]:
    """Render finalized deductions as human-readable log lines."""
    if not llm_deductions:
        return ["No LLM-derived mappings were applied in this run."]

    lines = ["LLM-derived mappings applied:"]
    for item in llm_deductions:
        datasets = item.get("datasets", [])
        datasets_text = ", ".join(datasets) if datasets else "n/a"
        lines.append(
            f"[{item['source_id']}] {item['alias']} -> {item['canonical_dv']} (datasets: {datasets_text})"
        )
    return lines


__all__ = [
    "_score_alias_match",
    "_select_llm_candidate_shortlist",
    "_augment_mapping_with_llm_deductions",
    "_collect_llm_deduction",
    "_finalize_llm_deductions",
    "_build_llm_deduction_log_lines",
]
