#!/usr/bin/env python3
"""Relations between harmonized constructs across studies.

Pooling study *means* (:mod:`opendv.analysis.multi_study_analysis`) needs every
study to measure a DV on the same metric, i.e. scalar measurement invariance,
which no scale shared by the development studies reaches. Correlations need
less: they are unchanged by any linear rescaling, so they stay comparable when
anchors, ranges and units differ between studies (metric invariance at most).
This module asks what harmonization makes visible at that level: which pairs
of constructs are measured together in several studies, how strongly they are
related within studies, and whether the relation holds across studies, labs
and research areas.

Method
------
1. **Provenance per column.** Every DV column of every standardized file is
   traced back to its raw column through ``meta_view.csv`` (one row per
   standardized column, in column order) and assigned the harmonization step
   that mapped it:

   =========  ================================================================
   level      a DV column is usable when it was mapped by ...
   =========  ================================================================
   names      nothing: studies are pooled by identical (normalized) raw names
   automatic  the shared DV schema (the same raw name maps to the same DV in
              ``standard_dv_mapping.yaml``), question-text header matching, or
              the instrument registry's scoring of such items
   curated    a study's own mapping file (including value labels, reverse
              keying, missing-value codes and unit conversions, which only a
              mapping file declares)
   llm        the local-LLM fallback
   =========  ================================================================

   Each level includes the ones above it. The ``names`` level identifies DVs
   by their raw names, so it is what pooling open data without any
   harmonization would give.
2. **Participants.** DV columns that map to the same DV within a study (wide
   repeated measures, items of one construct) are averaged per row, then per
   participant (participant columns from ``schemas/study_designs.yaml``).
3. **Per study.** The Pearson correlation of every pair of construct-level
   DVs over participants (between-person r, ``scipy.stats.pearsonr``). For
   repeated-measures studies, the repeated-measures correlation of Bakdash &
   Marusich (2017) is computed on the rows as well (within-person r: the
   correlation of the participant-centered values, df = N - P - 1).
4. **Across studies.** Correlations are pooled per pair on Fisher's z with
   :func:`opendv.analysis.multi_study_analysis.pool_effects` (REML,
   Hartung-Knapp-Sidik-Jonkman interval, prediction interval) and
   back-transformed to r.
5. **Findings.** A pooled relation is a *finding* when it rests on at least
   three studies, its HKSJ interval excludes 0 and its Benjamini-Hochberg
   q < .05 among the pairs eligible as findings (k >= 3, not scored from
   shared items) of the same level and association type
   (``statsmodels.stats.multitest``). It *generalizes* when the 95%
   prediction interval excludes 0 as well, i.e. a new study is expected to
   show the same sign.

Single questionnaire items are left out (their scales are analysed instead),
NASA-TLX subscales are kept as constructs. A pair is ``shared_items`` when
one DV is scored from the other's items or both share items (e.g. a
NASA-TLX subscale and the TLX total); such pairs are reported but never count
as findings.

Example::

    python -m opendv.analysis.construct_relations \\
        --input-dir data/processed/catalog_meta_analysis/standardized \\
        --output-dir analyses/output_relations
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from opendv import instruments as instruments_registry  # noqa: E402
from opendv.analysis import contrast_meta_analysis as cma  # noqa: E402
from opendv.analysis import multi_study_analysis as msa  # noqa: E402
from opendv.batch_reporting import categorize_mapping_source  # noqa: E402
from opendv.convert_dv import strip_duplicate_suffix  # noqa: E402
from opendv.item_composition import is_item_suffixed  # noqa: E402

logger = logging.getLogger(__name__)

LEVELS = ("names", "automatic", "curated", "llm")
# Not a rung of the ladder: the automatic pipeline with the LLM standing in for
# the study mappings (answers from the leave-alias-out benchmark, E1).
REPLAY_LEVEL = "llm_replay"
LEVEL_RANK = {name: rank for rank, name in enumerate(LEVELS)}
LEVEL_SHORT = {
    "names": "identical\nnames",
    "automatic": "+ shared schema,\nheader matching",
    "curated": "+ study\nmappings",
    "llm": "+ local\nLLM",
}
LEVEL_LABELS = {
    "names": "identical raw column names",
    "automatic": "+ shared schema, header matching, instrument scoring",
    "curated": "+ study mapping files",
    "llm": "+ local-LLM fallback",
}
# Value transformations that only a study's mapping file declares (quality.json keys).
CURATED_TRANSFORMS = (
    "reverse_keyed_items",
    "unit_conversions",
    "value_labels_applied",
    "missing_value_codes",
)
# Instruments whose items are analysed as constructs in their own right.
STANDALONE_ITEM_INSTRUMENTS = frozenset({"nasa_tlx_raw"})
ASSOCIATIONS = ("between_person", "within_person")
MIN_PARTICIPANTS = 10
MIN_WITHIN_DF = 10
FINDING_MIN_K = 3
ALPHA = 0.05
NAME_PREFIX = "name:"
NAME_DV_SEPARATOR = "@"
STUDY_FILE_COLUMN = cma.SOURCE_FILE_COLUMN
# Pickled tables are never read: unpickling runs code from the data source.
PICKLE_SUFFIXES = (".pkl", ".pickle")

STUDY_RELATION_COLUMNS = [
    "level", "association", "dv_x", "dv_y", "study", "domain", "n", "n_participants",
    "r", "p_value", "ci95_low", "ci95_high", "z", "vi", "max_level_x", "max_level_y",
]
POOLED_RELATION_COLUMNS = [
    "level", "association", "dv_x", "dv_y", "pair_type", "k_studies", "studies", "domains",
    "n_total", "n_largest_study", "pooled_r", "ci95_low", "ci95_high",
    "hksj_ci95_low", "hksj_ci95_high", "prediction_interval_low", "prediction_interval_high",
    "p_hksj", "q_bh", "tau", "heterogeneity_i2_pct", "q_pvalue", "pooling_method",
    "k_significant_alone", "k_same_sign", "ci_width_ratio_vs_best_study",
    "k_automotive", "k_other_domains", "pooled_r_automotive", "pooled_r_other_domains",
    "includes_llm_columns", "finding", "generalizes",
]


# ── Column provenance ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ColumnInfo:
    """Where one standardized DV column came from."""

    dv: str          # base canonical DV (no __dup_N suffix)
    level: int       # LEVEL_RANK of the step that mapped it (1-3)
    raw_name: str    # normalized raw column name


def _norm(name: object) -> str:
    return msa._normalize_colname(str(name))


def _name_column(info: ColumnInfo) -> str:
    """Column of the ``names`` level: the raw name and the DV it is mapped to."""
    return f"{NAME_PREFIX}{info.raw_name}{NAME_DV_SEPARATOR}{info.dv}"


def _dv_of(column: str) -> str:
    """The DV a column stands for (the mapped DV of a ``names`` column)."""
    if column.startswith(NAME_PREFIX):
        return column.rsplit(NAME_DV_SEPARATOR, 1)[-1]
    return column


def _registry() -> tuple:
    return instruments_registry.load_instruments(msa.INSTRUMENTS_PATH)


def _item_level_dvs() -> frozenset[str]:
    """DV ids of single questionnaire items (left out of the relation analysis)."""
    items: set[str] = set()
    for inst in _registry():
        if inst.id in STANDALONE_ITEM_INSTRUMENTS:
            continue
        items.update(inst.items)
    for subitems in msa.CONSTRUCT_SUBITEMS.values():
        items.update(i for i in subitems if msa._SUBITEM_ID_PATTERN.match(i))
    tlx = next((i for i in _registry() if i.id in STANDALONE_ITEM_INSTRUMENTS), None)
    keep = set(tlx.items) if tlx else set()
    return frozenset(items - keep)


def _instrument_of() -> dict[str, str]:
    """DV id -> instrument id, for items and scores of the registry."""
    out: dict[str, str] = {}
    for inst in _registry():
        for item in inst.items:
            out.setdefault(item, inst.id)
        for score in inst.scores:
            out.setdefault(score.dv, inst.id)
    return out


def _is_dv(column: str, item_dvs: frozenset[str] | None = None) -> bool:
    """A column the analysis can use: a canonical DV or a registry item."""
    return msa._is_analysis_column(column)


def _row_level(row: pd.Series, schema_lookup: Mapping[str, str], transformed: set[str]) -> int | None:
    """Harmonization level of one meta_view row, or None for a non-DV column."""
    dv = str(row["canonical_dv"])
    if not _is_dv(dv):
        return None
    method = str(row.get("mapping_method") or "").lower()
    category = categorize_mapping_source(row.get("mapping_source"))
    if method in {"blocked", "unmapped"} or category == "blocked":
        return None
    if method == "llm" or category == "llm_deduced":
        return LEVEL_RANK["llm"]
    raw = _norm(row.get("original_alias"))
    if category == "header_match" or schema_lookup.get(raw) == dv:
        level = LEVEL_RANK["automatic"]
    else:
        level = LEVEL_RANK["curated"]
    if str(row.get("original_alias")) in transformed:
        level = max(level, LEVEL_RANK["curated"])
    return level


def _transformed_columns(quality_path: Path) -> set[str]:
    if not quality_path.is_file():
        return set()
    try:
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    out: set[str] = set()
    for key in CURATED_TRANSFORMS:
        block = quality.get(key)
        if isinstance(block, dict):
            out.update(str(c) for c in block)
    return out


def _aligned_files(input_dir: Path, meta_view: pd.DataFrame):
    """Yield ``(study, standardized file, columns, meta_view rows, transformed raw columns)``.

    meta_view.csv has one row per standardized column, written in column
    order, so rows and columns are matched by position within each file.
    """
    for (study, dataset_id), rows in meta_view.groupby(["source_id", "dataset_id"], sort=False):
        study_dir = input_dir / str(study)
        quality_files = list(study_dir.glob("*-quality.json"))
        target = None
        transformed: set[str] = set()
        for qpath in quality_files:
            try:
                quality = json.loads(qpath.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if quality.get("dataset_id") == dataset_id:
                target = quality.get("standardized_file")
                transformed = _transformed_columns(qpath)
                break
        if not target or not (study_dir / target).is_file():
            continue
        if target.lower().endswith(PICKLE_SUFFIXES):
            continue
        columns = [str(c) for c in msa._read_header(study_dir / target)]
        if len(columns) != len(rows):
            logger.warning(
                "%s/%s: %d columns but %d meta_view rows; provenance skipped",
                study, target, len(columns), len(rows),
            )
            continue
        yield str(study), target, columns, rows, transformed


def file_aliases(input_dir: Path, meta_view: pd.DataFrame) -> dict[tuple[str, str], dict[str, str]]:
    """``{(study, standardized file): {column: raw column name}}`` for every column."""
    return {
        (study, target): {col: str(alias) for col, alias in zip(columns, rows["original_alias"], strict=True)}
        for study, target, columns, rows, _ in _aligned_files(input_dir, meta_view)
    }


def column_provenance(
    input_dir: Path, meta_view: pd.DataFrame
) -> dict[tuple[str, str], dict[str, ColumnInfo]]:
    """``{(study, standardized file name): {column: ColumnInfo}}`` for every DV column.

    Composites built by the batch inherit the highest level of their items.
    """
    schema_lookup = msa._load_standard_dv_lookup()
    out: dict[tuple[str, str], dict[str, ColumnInfo]] = {}
    for study, target, columns, rows, transformed in _aligned_files(input_dir, meta_view):
        infos: dict[str, ColumnInfo] = {}
        item_levels: dict[str, list[int]] = {}
        for column, (_, row) in zip(columns, rows.iterrows(), strict=True):
            if is_item_suffixed(column):
                level = _row_level(row, schema_lookup, transformed)
                if level is not None and isinstance(row.get("item_of"), str):
                    item_levels.setdefault(row["item_of"], []).append(level)
                continue
            level = _row_level(row, schema_lookup, transformed)
            if level is None:
                continue
            infos[column] = ColumnInfo(
                dv=strip_duplicate_suffix(column),
                level=level,
                raw_name=_norm(row.get("original_alias")),
            )
        for column, info in list(infos.items()):
            levels = item_levels.get(info.dv)
            if levels and column == info.dv:
                infos[column] = ColumnInfo(info.dv, max(levels), info.raw_name)
        out[(study, target)] = infos
    return out


def replay_map(
    input_dir: Path,
    meta_view: pd.DataFrame,
    provenance: Mapping[tuple[str, str], Mapping[str, ColumnInfo]],
    e1_items: pd.DataFrame,
) -> tuple[dict[tuple[str, str], dict[str, str | None]], dict[str, int]]:
    """The LLM's answer for every column a study mapping decided.

    *e1_items* are the E1 rows of ``evaluate_llm_mapping`` (leave-alias-out:
    the curated alias is hidden and the pipeline's LLM step asked); the
    plausibility-checked answer (``final_checked``) is used. A curated DV
    column without an E1 item counts as unanswered (dropped).
    """
    e1 = e1_items[e1_items["experiment"] == "E1"]
    answers: dict[tuple[str, str], str | None] = {}
    for row in e1.itertuples(index=False):
        answer = row.final_checked if isinstance(row.final_checked, str) and row.final_checked else None
        answers[(str(row.source_id), str(row.alias))] = answer
    replay: dict[tuple[str, str], dict[str, str | None]] = {}
    counts = {"curated_dv_columns": 0, "answered": 0, "unanswered_or_rejected": 0,
              "same_dv_as_curated": 0, "curated_non_dv_mapped_by_llm": 0}
    for (study, target), aliases in file_aliases(input_dir, meta_view).items():
        infos = provenance.get((study, target), {})
        mapping: dict[str, str | None] = {}
        for column, alias in aliases.items():
            if is_item_suffixed(column):
                continue
            info = infos.get(column)
            key = (study, alias)
            if info is not None and info.level == LEVEL_RANK["curated"]:
                answer = answers.get(key)
                mapping[column] = answer
                counts["curated_dv_columns"] += 1
                counts["answered" if answer else "unanswered_or_rejected"] += 1
                counts["same_dv_as_curated"] += int(answer == info.dv)
            elif info is None and column.startswith("meta_") and answers.get(key):
                mapping[column] = answers[key]
                counts["curated_non_dv_mapped_by_llm"] += 1
        if mapping:
            replay[(study, target)] = mapping
    return replay, counts


def compare_conclusions(pooled: pd.DataFrame, reference: str = "curated",
                        candidates: Iterable[str] = ("automatic", REPLAY_LEVEL)) -> pd.DataFrame:
    """Findings of each candidate level against the reference level's findings.

    A candidate finding is *confirmed* when the reference has the same pair as
    a finding with the same sign, *sign_flipped* when the reference finding
    has the opposite sign, and *unsupported* otherwise.
    """
    rows = []
    keys = ["association", "dv_x", "dv_y"]
    ref = pooled[pooled["level"] == reference].set_index(keys)
    ref_findings = ref[ref["finding"]]
    for level in candidates:
        cand = pooled[pooled["level"] == level].set_index(keys)
        for association in ASSOCIATIONS:
            c = cand[cand.index.get_level_values(0) == association]
            r = ref_findings[ref_findings.index.get_level_values(0) == association]
            found = c[c["finding"]]
            both = found.index.intersection(r.index)
            same_sign = int((np.sign(found.loc[both, "pooled_r"]) == np.sign(r.loc[both, "pooled_r"])).sum())
            shared = c.index.intersection(ref.index[ref.index.get_level_values(0) == association])
            rows.append({
                "level": level, "association": association,
                "reference_findings": int(len(r)),
                "findings": int(len(found)),
                "confirmed": same_sign,
                "sign_flipped": int(len(both) - same_sign),
                "unsupported": int(len(found) - len(both)),
                "pairs_pooled_in_both": int(len(shared)),
                "median_abs_difference_r": float(
                    (c.loc[shared, "pooled_r"] - ref.loc[shared, "pooled_r"]).abs().median()
                ) if len(shared) else np.nan,
            })
    return pd.DataFrame(rows)


# ── Study frames per harmonization level ─────────────────────────────────────

def level_frames(
    input_dir: Path,
    provenance: Mapping[tuple[str, str], Mapping[str, ColumnInfo]],
    level: str,
    replay: Mapping[tuple[str, str], Mapping[str, str | None]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Per study: the non-DV columns plus the DV columns usable at *level*.

    At ``names`` the DV columns are renamed to ``name:<raw name>``; at the
    other levels registry scores are added from the items that remain.

    At ``llm_replay`` (with *replay*: ``{(study, file): {column: DV or None}}``)
    the automatic and LLM columns are kept, and every column a study mapping
    decided (a DV, or a column it locked as ``meta_*``) takes the DV the LLM
    gave it with that mapping hidden (None drops it): the pipeline as it would
    run on the same data without curated mappings.
    """
    replaying = level == REPLAY_LEVEL
    rank = LEVEL_RANK["automatic"] if replaying else LEVEL_RANK[level]
    # At ``names`` the baseline is told which columns are DVs (those the
    # deterministic pipeline maps) but pools them by raw name only.
    limit = LEVEL_RANK["curated"] if rank == LEVEL_RANK["names"] else rank
    by_study: dict[str, list[pd.DataFrame]] = {}
    for (study, file_name), infos in sorted(provenance.items()):
        if not infos:
            continue
        if file_name.lower().endswith(PICKLE_SUFFIXES):
            logger.warning("%s/%s: pickled table skipped", study, file_name)
            continue
        renames = dict((replay or {}).get((study, file_name), {})) if replaying else {}
        if replaying:
            keep_levels = {LEVEL_RANK["automatic"], LEVEL_RANK["llm"]}
            if not any(info.level in keep_levels for info in infos.values()) and not any(renames.values()):
                continue
        elif not any(info.level <= limit for info in infos.values()):
            continue
        df = msa._read_file(input_dir / study / file_name)
        df.columns = [str(c) for c in df.columns]
        if replaying:
            drop = [c for c in df.columns if c in infos and infos[c].level == LEVEL_RANK["curated"]
                    and not renames.get(c)]
            drop += [c for c, dv in renames.items() if c in df.columns and c not in infos and not dv]
            rename = {c: f"{dv}__dup_{900 + i}" for i, (c, dv) in enumerate(renames.items())
                      if dv and c in df.columns}
            df = df.drop(columns=[c for c in drop if c in df.columns]).rename(columns=rename)
            renamed = set(rename.values())
            stray = [c for c in df.columns if c not in infos and c not in renamed
                     and _is_dv(strip_duplicate_suffix(c)) and not is_item_suffixed(c)]
            df = df.drop(columns=stray)
            df[STUDY_FILE_COLUMN] = file_name
            by_study.setdefault(study, []).append(df)
            continue
        drop = [c for c in df.columns if c in infos and infos[c].level > limit]
        if rank == LEVEL_RANK["names"]:
            # Items are left out as at the other levels, and a raw name pools
            # only where it means the same DV (generous to the baseline: it
            # never pools two different measures that share a name).
            items = _item_level_dvs()
            drop += [c for c in df.columns if c in infos and c not in drop
                     and (infos[c].dv in items or msa._SUBITEM_ID_PATTERN.match(infos[c].dv))]
            rename = {c: _name_column(infos[c]) for c in df.columns if c in infos and c not in drop}
            df = df.drop(columns=drop).rename(columns=rename)
            df = df.loc[:, ~df.columns.duplicated()]
        else:
            df = df.drop(columns=drop)
            # DV-looking columns the provenance does not vouch for (e.g. a
            # non-DV column whose raw name happens to be a DV alias) stay out.
            stray = [c for c in df.columns if c not in infos and _is_dv(strip_duplicate_suffix(c))
                     and not is_item_suffixed(c)]
            df = df.drop(columns=stray)
        df[STUDY_FILE_COLUMN] = file_name
        by_study.setdefault(study, []).append(df)
    frames: dict[str, pd.DataFrame] = {}
    for study, parts in by_study.items():
        combined = pd.concat(parts, ignore_index=True, sort=False)
        if rank > LEVEL_RANK["names"]:
            combined = msa.add_derived_scale_scores(combined)
        frames[study] = combined
    return frames


def dv_matrix(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    """Numeric DV columns, one per DV: columns mapping to the same DV are
    averaged per row (wide repeated measures, items of one construct)."""
    groups: dict[str, list[str]] = {}
    for column in frame.columns:
        if column == STUDY_FILE_COLUMN or is_item_suffixed(column):
            continue
        if level == "names":
            if column.startswith(NAME_PREFIX):
                groups.setdefault(column, []).append(column)
            continue
        base = strip_duplicate_suffix(column)
        dv = msa._canonicalize_dv_name(base)
        if dv is None and msa._is_analysis_column(base):  # registry item names
            dv = base
        if dv is None:
            continue
        groups.setdefault(dv, []).append(column)
    data = {}
    for dv, columns in groups.items():
        values = frame[columns].apply(pd.to_numeric, errors="coerce")
        if values.notna().sum().sum() == 0:
            continue
        data[dv] = values.mean(axis=1, skipna=True)
    return pd.DataFrame(data, index=frame.index)


def construct_level_dvs(dvs: Iterable[str], level: str) -> list[str]:
    if level == "names":
        return sorted(dvs)
    items = _item_level_dvs()
    return sorted(dv for dv in dvs if dv not in items and not msa._SUBITEM_ID_PATTERN.match(dv))


# ── Correlations per study ───────────────────────────────────────────────────

def _fisher(r: float, n_eff: float) -> tuple[float, float]:
    r = float(np.clip(r, -0.9999, 0.9999))
    return float(np.arctanh(r)), 1.0 / (n_eff - 3.0)


def between_person(means: pd.DataFrame, x: str, y: str, min_n: int = MIN_PARTICIPANTS) -> dict | None:
    pair = means[[x, y]].dropna()
    n = len(pair)
    if n < min_n or pair[x].std() <= 0 or pair[y].std() <= 0:
        return None
    res = stats.pearsonr(pair[x], pair[y])
    ci = res.confidence_interval()
    z, vi = _fisher(res.statistic, n)
    return {"n": n, "n_participants": n, "r": float(res.statistic), "p_value": float(res.pvalue),
            "ci95_low": float(ci.low), "ci95_high": float(ci.high), "z": z, "vi": vi}


def within_person(
    rows: pd.DataFrame, participant: pd.Series, x: str, y: str, min_df: int = MIN_WITHIN_DF
) -> dict | None:
    """Repeated-measures correlation (Bakdash & Marusich 2017)."""
    frame = pd.DataFrame({"p": participant, "x": rows[x], "y": rows[y]}).dropna()
    if frame.empty:
        return None
    counts = frame.groupby("p")["x"].transform("size")
    frame = frame[counts >= 2]
    n_obs, n_part = len(frame), frame["p"].nunique()
    df = n_obs - n_part - 1
    if n_part < 3 or df < min_df:
        return None
    xc = frame["x"] - frame.groupby("p")["x"].transform("mean")
    yc = frame["y"] - frame.groupby("p")["y"].transform("mean")
    if xc.std() <= 0 or yc.std() <= 0:
        return None
    r = float(np.corrcoef(xc, yc)[0, 1])
    t = r * np.sqrt(df / max(1e-12, 1.0 - r**2))
    p = float(2 * stats.t.sf(abs(t), df))
    z, vi = _fisher(r, df + 2)
    half = 1.96 * np.sqrt(vi)
    return {"n": n_obs, "n_participants": n_part, "r": r, "p_value": p,
            "ci95_low": float(np.tanh(z - half)), "ci95_high": float(np.tanh(z + half)),
            "z": z, "vi": vi}


def study_relations(
    frames: Mapping[str, pd.DataFrame],
    registry: cma.DesignRegistry,
    level: str,
    domains: Mapping[str, str] | None = None,
    provenance_levels: Mapping[tuple[str, str], int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Correlations of every construct pair in every study with a declared participant column."""
    domains = domains or {}
    provenance_levels = provenance_levels or {}
    rows: list[dict] = []
    skipped: list[str] = []
    for study, frame in sorted(frames.items()):
        design = registry.studies.get(study)
        participant = design.participant_series(frame) if design else None
        if participant is None or participant.notna().sum() == 0:
            skipped.append(study)
            continue
        matrix = dv_matrix(frame, level)
        dvs = construct_level_dvs(matrix.columns, level)
        if len(dvs) < 2:
            continue
        matrix = matrix[dvs]
        keep = participant.notna()
        means = matrix[keep].groupby(participant[keep]).mean()
        repeated = bool((participant[keep].value_counts() > 1).any())
        for i, x in enumerate(dvs):
            for y in dvs[i + 1:]:
                results = {"between_person": between_person(means, x, y)}
                if repeated:
                    results["within_person"] = within_person(matrix[keep], participant[keep], x, y)
                for association, res in results.items():
                    if res is None:
                        continue
                    rows.append({
                        "level": level, "association": association, "dv_x": x, "dv_y": y,
                        "study": study, "domain": domains.get(study, "unknown"), **res,
                        "max_level_x": provenance_levels.get((study, x), np.nan),
                        "max_level_y": provenance_levels.get((study, y), np.nan),
                    })
    return pd.DataFrame(rows, columns=STUDY_RELATION_COLUMNS), skipped


# ── Pooling ──────────────────────────────────────────────────────────────────

def pair_type(x: str, y: str) -> str:
    """``shared_items`` (scored from shared items), ``same_instrument`` or ``cross_construct``."""
    x, y = _dv_of(x), _dv_of(y)
    items_x = set(msa.CONSTRUCT_SUBITEMS.get(x, [])) | {x}
    items_y = set(msa.CONSTRUCT_SUBITEMS.get(y, [])) | {y}
    if items_x & items_y:
        return "shared_items"
    instruments = _instrument_of()
    if instruments.get(x) and instruments.get(x) == instruments.get(y):
        return "same_instrument"
    return "cross_construct"


def _pool(group: pd.DataFrame) -> dict:
    pooled = msa.pool_effects(group["z"].to_numpy(), group["vi"].to_numpy(), "REML")
    return pooled


def _p_hksj(pooled: Mapping[str, float], k: int) -> float:
    se = pooled["hksj_se"]
    if not np.isfinite(se) or se <= 0:
        return np.nan
    return float(2 * stats.t.sf(abs(pooled["random_effects_mean"] / se), k - 1))


def pool_relations(study_rows: pd.DataFrame, llm_pairs: set[tuple[str, str, str]] | None = None) -> pd.DataFrame:
    """Pool every pair measured in >= 2 studies; flag findings per level and association."""
    if study_rows.empty:
        return pd.DataFrame(columns=POOLED_RELATION_COLUMNS)
    llm_pairs = llm_pairs or set()
    out: list[dict] = []
    keys = ["level", "association", "dv_x", "dv_y"]
    for (level, association, x, y), group in study_rows.groupby(keys, sort=True):
        k = len(group)
        if k < 2:
            continue
        pooled = _pool(group)
        tanh = np.tanh
        row = {
            "level": level, "association": association, "dv_x": x, "dv_y": y,
            "pair_type": pair_type(x, y), "k_studies": k,
            "studies": "; ".join(group["study"]), "domains": "; ".join(sorted(set(group["domain"]))),
            "n_total": int(group["n_participants"].sum()), "n_largest_study": int(group["n_participants"].max()),
            "pooled_r": float(tanh(pooled["random_effects_mean"])),
            "ci95_low": float(tanh(pooled["ci95_low"])), "ci95_high": float(tanh(pooled["ci95_high"])),
            "hksj_ci95_low": float(tanh(pooled["hksj_ci95_low"])),
            "hksj_ci95_high": float(tanh(pooled["hksj_ci95_high"])),
            "prediction_interval_low": float(tanh(pooled["prediction_interval_low"])),
            "prediction_interval_high": float(tanh(pooled["prediction_interval_high"])),
            "p_hksj": _p_hksj(pooled, k), "tau": pooled["tau"],
            "heterogeneity_i2_pct": pooled["heterogeneity_i2_pct"], "q_pvalue": pooled["q_pvalue"],
            "pooling_method": pooled["pooling_method"],
            "k_significant_alone": int((group["p_value"] < ALPHA).sum()),
            "k_same_sign": int((np.sign(group["r"]) == np.sign(pooled["random_effects_mean"])).sum()),
        }
        best_width = float((group["ci95_high"] - group["ci95_low"]).min())
        pooled_width = row["hksj_ci95_high"] - row["hksj_ci95_low"]
        row["ci_width_ratio_vs_best_study"] = pooled_width / best_width if best_width > 0 else np.nan
        for name, mask in (("automotive", group["domain"] == "automotive"),
                           ("other_domains", group["domain"] != "automotive")):
            sub = group[mask]
            row[f"k_{name}"] = int(len(sub))
            row[f"pooled_r_{name}"] = (
                float(np.tanh(_pool(sub)["random_effects_mean"])) if len(sub) >= 2 else np.nan
            )
        row["includes_llm_columns"] = bool(
            (group["max_level_x"] == LEVEL_RANK["llm"]).any() or (group["max_level_y"] == LEVEL_RANK["llm"]).any()
        )
        out.append(row)
    pooled = pd.DataFrame(out)
    # The false discovery rate is controlled over the pairs that can be
    # findings (k >= 3, not scored from shared items), per level and
    # association; pairs of two or of shared items are reported, not tested.
    pooled["q_bh"] = np.nan
    eligible = (pooled["k_studies"] >= FINDING_MIN_K) & (pooled["pair_type"] != "shared_items")
    for _, idx in pooled[eligible].groupby(["level", "association"]).groups.items():
        p = pooled.loc[idx, "p_hksj"]
        valid = p.notna()
        if valid.any():
            pooled.loc[p[valid].index, "q_bh"] = multipletests(p[valid], method="fdr_bh")[1]
    excludes_zero = (pooled["hksj_ci95_low"] > 0) | (pooled["hksj_ci95_high"] < 0)
    pooled["finding"] = (
        (pooled["k_studies"] >= FINDING_MIN_K)
        & (pooled["pair_type"] != "shared_items")
        & excludes_zero
        & (pooled["q_bh"] < ALPHA)
    )
    pi_excludes = (pooled["prediction_interval_low"] > 0) | (pooled["prediction_interval_high"] < 0)
    pooled["generalizes"] = pooled["finding"] & pi_excludes
    return pooled[POOLED_RELATION_COLUMNS].sort_values(
        ["level", "association", "finding", "k_studies"], ascending=[True, True, False, False]
    ).reset_index(drop=True)


# ── Ledger: what each harmonization step adds ────────────────────────────────

def _names_to_dvs(provenance: Mapping[tuple[str, str], Mapping[str, ColumnInfo]]) -> dict[tuple[str, str], set[str]]:
    """(study, name:<raw>) -> canonical DVs that raw name was mapped to."""
    out: dict[tuple[str, str], set[str]] = {}
    for (study, _), infos in provenance.items():
        for info in infos.values():
            out.setdefault((study, _name_column(info)), set()).add(info.dv)
    return out


def ladder(pooled: pd.DataFrame, study_rows: pd.DataFrame) -> pd.DataFrame:
    """Counts per harmonization level: DVs shared, pairs pooled, findings."""
    rows = []
    for level in LEVELS:
        for association in ASSOCIATIONS:
            sub = pooled[(pooled["level"] == level) & (pooled["association"] == association)]
            st = study_rows[(study_rows["level"] == level) & (study_rows["association"] == association)]
            cross = sub[sub["pair_type"] == "cross_construct"]
            rows.append({
                "level": level,
                "level_label": LEVEL_LABELS[level],
                "association": association,
                "studies_with_relations": int(st["study"].nunique()),
                "pairs_pooled_k2": int(len(sub)),
                "pairs_pooled_k3": int((sub["k_studies"] >= FINDING_MIN_K).sum()),
                "cross_construct_pairs_k3": int((cross["k_studies"] >= FINDING_MIN_K).sum()),
                # Participants of the studies with relations (largest n per study).
                "participants": int(st.groupby("study")["n_participants"].max().sum()) if not st.empty else 0,
                "findings": int(sub["finding"].sum()),
                "cross_construct_findings": int(cross["finding"].sum()),
                "generalizing_findings": int(sub["generalizes"].sum()),
            })
    return pd.DataFrame(rows)


def attribute_findings(
    pooled: pd.DataFrame,
    provenance: Mapping[tuple[str, str], Mapping[str, ColumnInfo]],
) -> pd.DataFrame:
    """For every pair pooled at the highest level analysed: k at each level and
    the first level at which it is pooled (k >= 2) and becomes a finding."""
    names_map = _names_to_dvs(provenance)
    names = pooled[pooled["level"] == "names"]
    k_names: dict[tuple[str, str, str], int] = {}
    finding_names: dict[tuple[str, str, str], bool] = {}
    for _, row in names.iterrows():
        studies = row["studies"].split("; ")
        xs = set.intersection(*(names_map.get((s, row["dv_x"]), set()) for s in studies)) if studies else set()
        ys = set.intersection(*(names_map.get((s, row["dv_y"]), set()) for s in studies)) if studies else set()
        for x in xs:
            for y in ys:
                key = (row["association"], *sorted((x, y)))
                if row["k_studies"] > k_names.get(key, 0):
                    k_names[key] = int(row["k_studies"])
                    finding_names[key] = bool(row["finding"])
    present = [lvl for lvl in LEVELS if (pooled["level"] == lvl).any()]
    if not present:
        return pd.DataFrame()
    top = pooled[pooled["level"] == present[-1]].copy()
    by_level = {
        level: pooled[pooled["level"] == level].set_index(["association", "dv_x", "dv_y"])
        for level in LEVELS[1:]
    }
    records = []
    for _, row in top.iterrows():
        key = (row["association"], row["dv_x"], row["dv_y"])
        rec = {"association": row["association"], "dv_x": row["dv_x"], "dv_y": row["dv_y"],
               "pair_type": row["pair_type"]}
        rec["k_names"] = k_names.get((row["association"], *sorted((row["dv_x"], row["dv_y"]))), 0)
        rec["finding_names"] = finding_names.get((row["association"], *sorted((row["dv_x"], row["dv_y"]))), False)
        for level in LEVELS[1:]:
            table = by_level[level]
            hit = table.loc[key] if key in table.index else None
            rec[f"k_{level}"] = int(hit["k_studies"]) if hit is not None else 0
            rec[f"finding_{level}"] = bool(hit["finding"]) if hit is not None else False
            rec[f"pooled_r_{level}"] = float(hit["pooled_r"]) if hit is not None else np.nan
        pooled_at = next((lvl for lvl in LEVELS if rec[f"k_{lvl}"] >= 2), None)
        finding_at = next((lvl for lvl in LEVELS if rec[f"finding_{lvl}"]), None)
        rec["first_pooled_at"] = pooled_at
        rec["first_finding_at"] = finding_at
        records.append(rec)
    return pd.DataFrame(records)


# ── Figures ──────────────────────────────────────────────────────────────────

def _label(dv: str) -> str:
    return _dv_of(dv).replace("_", " ")


# Reference palette of the data-viz guidance (categorical slots 1-3, ink, surface).
_INK, _INK_2, _MUTED, _GRID = "#0b0b0b", "#52514e", "#a9a8a2", "#e4e3df"
_SLOTS = ("#2a78d6", "#eb6834", "#1baf7a")
_SURFACE = "#fcfcfb"


def _style(ax) -> None:
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_INK_2, labelsize=8)
    ax.grid(axis="x", color=_GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_findings(pooled: pd.DataFrame, study_rows: pd.DataFrame, path: Path, level: str = "curated") -> None:
    """Cross-construct relations pooled over >= 3 studies at *level*: study r
    (dots), pooled r with its HKSJ interval and the prediction interval, one
    panel per association; findings in blue."""
    sub = pooled[(pooled["level"] == level) & (pooled["pair_type"] == "cross_construct")
                 & (pooled["k_studies"] >= FINDING_MIN_K)]
    if sub.empty:
        return
    panels = [a for a in ASSOCIATIONS if (sub["association"] == a).any()]
    heights = [max(1, int((sub["association"] == a).sum())) for a in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.2, 0.5 * sum(heights) + 1.6), sharex=True,
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    fig.patch.set_facecolor(_SURFACE)
    titles = {"between_person": "Between participants (participant means)",
              "within_person": "Within participants (repeated-measures r)"}
    for ax, association in zip(axes[:, 0], panels, strict=True):
        rows = sub[sub["association"] == association].sort_values("pooled_r")
        _style(ax)
        for i, (_, row) in enumerate(rows.iterrows()):
            color = _SLOTS[0] if row["finding"] else _INK_2
            st = study_rows[(study_rows["level"] == level) & (study_rows["association"] == association)
                            & (study_rows["dv_x"] == row["dv_x"]) & (study_rows["dv_y"] == row["dv_y"])]
            ax.scatter(st["r"], np.full(len(st), i), s=24, facecolor="none", edgecolor=_MUTED, lw=1, zorder=2)
            if np.isfinite(row["prediction_interval_low"]):
                ax.plot([row["prediction_interval_low"], row["prediction_interval_high"]], [i, i],
                        color=_GRID, lw=7, solid_capstyle="butt", zorder=1)
            ax.plot([row["hksj_ci95_low"], row["hksj_ci95_high"]], [i, i], color=color, lw=2, zorder=3)
            ax.scatter([row["pooled_r"]], [i], marker="D", s=42, color=color,
                       edgecolor=_SURFACE, linewidth=1.5, zorder=4)
        ax.axvline(0, color=_INK_2, lw=0.8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{_label(x)} × {_label(y)} (k = {k})" for x, y, k in
                            zip(rows["dv_x"], rows["dv_y"], rows["k_studies"], strict=True)], color=_INK)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_title(titles[association], loc="left", fontsize=9, color=_INK)
    axes[-1, 0].set_xlim(-1, 1)
    axes[-1, 0].set_xlabel("correlation r", color=_INK_2, fontsize=8)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="D", color=_SLOTS[0], lw=2, label="pooled r, HKSJ 95% CI: finding (FDR 5%)"),
        Line2D([], [], marker="D", color=_INK_2, lw=2, label="pooled r, HKSJ 95% CI: not a finding"),
        Line2D([], [], color=_GRID, lw=7, label="95% prediction interval"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=_MUTED, label="single study"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7, frameon=False,
               labelcolor=_INK_2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200, facecolor=_SURFACE)
    plt.close(fig)


def plot_ladder(table: pd.DataFrame, path: Path) -> None:
    """Relations measured in >= 2 and >= 3 studies, and findings, per level."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
    fig.patch.set_facecolor(_SURFACE)
    series = [("pairs_pooled_k2", "in ≥ 2 studies"), ("pairs_pooled_k3", "in ≥ 3 studies"),
              ("findings", "findings (FDR 5%)")]
    titles = {"between_person": "Between participants", "within_person": "Within participants"}
    for ax, association in zip(axes, ASSOCIATIONS, strict=True):
        sub = table[table["association"] == association]
        _style(ax)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=_GRID, lw=0.8)
        x = np.arange(len(sub))
        width = 0.26
        for j, (column, label) in enumerate(series):
            values = sub[column].to_numpy()
            bars = ax.bar(x + (j - 1) * width, values, width - 0.03, color=_SLOTS[j], label=label)
            for bar, value in zip(bars, values, strict=True):
                ax.text(bar.get_x() + bar.get_width() / 2, value, f"{int(value)}", ha="center",
                        va="bottom", fontsize=7, color=_INK_2)
        ax.set_xticks(x)
        ax.set_xticklabels([LEVEL_SHORT[lvl] for lvl in sub["level"]], fontsize=7, color=_INK)
        ax.set_title(titles[association], loc="left", fontsize=9, color=_INK)
    axes[0].set_ylabel("pairs of constructs", color=_INK_2, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, fontsize=7, frameon=False, labelcolor=_INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200, facecolor=_SURFACE)
    plt.close(fig)


# ── Orchestration ────────────────────────────────────────────────────────────

def run(
    input_dir: Path,
    output_dir: Path,
    meta_view_path: Path | None = None,
    designs_path: Path = cma.DEFAULT_DESIGNS_PATH,
    levels: Iterable[str] = LEVELS,
    llm_replay_items: Path | Sequence[Path] | None = None,
) -> dict:
    """Relations per study and pooled, per harmonization level.

    *llm_replay_items* are one or more llm_mapping_evaluation_items.csv
    files of evaluate_llm_mapping; several files (e.g. benchmark runs on
    disjoint sets of studies) are concatenated.
    """
    replay_files = [llm_replay_items] if isinstance(llm_replay_items, (str, Path)) else list(llm_replay_items or [])
    meta_view_path = meta_view_path or input_dir.parent / "meta_view.csv"
    meta_view = pd.read_csv(meta_view_path, low_memory=False)
    provenance = column_provenance(input_dir, meta_view)
    registry = cma.load_study_designs(designs_path)
    domains = {study: design.domain for study, design in registry.studies.items()}
    dv_levels: dict[tuple[str, str], int] = {}
    for (study, _), infos in provenance.items():
        for info in infos.values():
            dv_levels[(study, info.dv)] = max(dv_levels.get((study, info.dv), 0), info.level)
    # Registry scores computed in the analysis take the highest level of their items.
    for score_dv, items in msa._build_construct_subitems().items():
        for study in {study for study, _ in dv_levels}:
            found = [dv_levels[(study, i)] for i in items if (study, i) in dv_levels]
            if found and (study, score_dv) not in dv_levels:
                dv_levels[(study, score_dv)] = max(found)

    all_rows, skipped_by_level = [], {}
    for level in levels:
        frames = level_frames(input_dir, provenance, level)
        rows, skipped = study_relations(frames, registry, level, domains, dv_levels)
        all_rows.append(rows)
        skipped_by_level[level] = skipped
        logger.info("%s: %d study correlations; no participant column: %s", level, len(rows), skipped)
    replay_counts = None
    if replay_files:
        e1_items = pd.concat([pd.read_csv(path) for path in replay_files], ignore_index=True)
        replay, replay_counts = replay_map(input_dir, meta_view, provenance, e1_items)
        frames = level_frames(input_dir, provenance, REPLAY_LEVEL, replay)
        rows, _ = study_relations(frames, registry, REPLAY_LEVEL, domains, dv_levels)
        all_rows.append(rows)
        logger.info("%s: %d study correlations (%s)", REPLAY_LEVEL, len(rows), replay_counts)
    study_rows = pd.concat(all_rows, ignore_index=True)
    pooled = pool_relations(study_rows)
    table = ladder(pooled, study_rows)
    attribution = attribute_findings(pooled, provenance)

    output_dir.mkdir(parents=True, exist_ok=True)
    study_rows.to_csv(output_dir / "relations_by_study.csv", index=False)
    pooled.to_csv(output_dir / "relations_pooled.csv", index=False)
    table.to_csv(output_dir / "harmonization_ladder.csv", index=False)
    attribution.to_csv(output_dir / "relations_attribution.csv", index=False)
    plot_findings(pooled, study_rows, output_dir / "relations_findings.png", level="curated")
    plot_ladder(table, output_dir / "harmonization_ladder.png")
    conclusions = None
    if replay_counts is not None:
        conclusions = compare_conclusions(pooled)
        conclusions.to_csv(output_dir / "llm_replay_conclusions.csv", index=False)
    summary = {
        "levels": {lvl: LEVEL_LABELS[lvl] for lvl in levels},
        "studies_without_participant_column": skipped_by_level,
        "finding_rule": (
            f"k >= {FINDING_MIN_K} studies, pair not scored from shared items, HKSJ 95% CI excludes 0, "
            f"Benjamini-Hochberg q < {ALPHA} over these eligible pairs within level and association"
        ),
        "ladder": table.to_dict(orient="records"),
    }
    if replay_counts is not None:
        summary["llm_replay"] = {
            "e1_items": str(replay_files[0]) if len(replay_files) == 1 else [str(p) for p in replay_files],
            "columns": replay_counts,
            "conclusions_vs_curated": conclusions.to_dict(orient="records"),
        }
    (output_dir / "relations_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input-dir", required=True, help="Standardized study directory of a batch run")
    parser.add_argument("--output-dir", default="analyses/output_relations")
    parser.add_argument("--meta-view", default=None, help="meta_view.csv (default: <input-dir>/../meta_view.csv)")
    parser.add_argument("--designs", default=str(cma.DEFAULT_DESIGNS_PATH))
    parser.add_argument(
        "--llm-replay-items", default=None, nargs="+",
        help="llm_mapping_evaluation_items.csv of evaluate_llm_mapping (several files, e.g. runs on disjoint "
             "sets of studies, are concatenated): also analyse the pipeline with the LLM's leave-alias-out "
             "answers in place of the study mappings and compare the findings",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = run(
        Path(args.input_dir), Path(args.output_dir),
        Path(args.meta_view) if args.meta_view else None, Path(args.designs),
        llm_replay_items=[Path(p) for p in args.llm_replay_items] if args.llm_replay_items else None,
    )
    print(json.dumps(summary["ladder"], indent=2))


if __name__ == "__main__":
    main()
