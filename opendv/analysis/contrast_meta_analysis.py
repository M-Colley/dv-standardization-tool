"""Condition-contrast effect sizes and their synthesis across studies.

The per-DV meta-analysis in :mod:`analyses.multi_study_analysis` pools study
*means*. That describes where studies sit on a shared scale but not what the
experimental manipulations did. This module uses the study designs declared
in ``schemas/study_designs.yaml`` (participant column, manipulated factors
and their levels) to compute the effect of each condition against its
reference condition, per study and DV, and pools those effects across studies
for factors that share a canonical independent variable (e.g. eHMI present
vs. absent).

Method
------
1. Rows are averaged per participant and factor level (over trials and the
   other factors), so participants, not rows, are the units of analysis.
2. Each non-reference level is contrasted with the reference level (level
   minus reference, in raw-score SD units):

   * between-subjects factors: Hedges' g (metafor ``measure = "SMD"``,
     ``vtype = "LS"``);
   * within-subjects factors: standardized mean change with the average of
     the two condition SDs as standardizer (metafor ``measure = "SMCRPH"``,
     Bonett 2008), which puts paired contrasts on the same metric as g.

   The design actually used is the one observed in the data (participants
   seen at more than one level → within); a mismatch with the declared design
   is reported, not silently accepted.
3. Several contrasts from one study against the same reference are
   dependent. They are combined into one study-level effect by generalized
   least squares under a compound-symmetric correlation ``rho`` (metafor
   ``aggregate.escalc(struct = "CS")``); ``rho`` is varied as a sensitivity
   analysis.
4. Study-level effects are pooled with the same estimator code as the mean
   meta-analysis (:func:`analyses.multi_study_analysis.pool_effects`: DL or
   REML, Wald and Hartung-Knapp-Sidik-Jonkman intervals).

All effect-size formulas are checked against metafor 5.2.1 in
``tests/test_contrast_meta_analysis.py``.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import special


from opendv.analysis import multi_study_analysis as msa  # noqa: E402
from opendv._paths import SCHEMAS_DIR

logger = logging.getLogger(__name__)

DEFAULT_DESIGNS_PATH = SCHEMAS_DIR / "study_designs.yaml"
VALID_DESIGNS = ("within", "between")
DEFAULT_RHO = 0.5
SENSITIVITY_RHOS = (0.2, 0.8)
MIN_PARTICIPANTS = 3

CONTRAST_EFFECT_COLUMNS = [
    "study", "factor", "canonical_factor", "level", "canonical_level", "reference",
    "dv", "design_declared", "design_observed", "measure", "n", "n_level", "n_reference",
    "mean_level", "mean_reference", "sd_level", "sd_reference", "r",
    "yi", "vi", "se", "ci95_low", "ci95_high", "direction", "favours_level",
    "mapping_source",
]
CONTRAST_POOLED_COLUMNS = [
    "canonical_factor", "dv", "k_studies", "studies", "n_total",
    *msa.META_ANALYSIS_COLUMNS[msa.META_ANALYSIS_COLUMNS.index("pooling_method"):
                                msa.META_ANALYSIS_COLUMNS.index("h2") + 1],
    "estimator", "rho", "direction", "k_llm_deduced", "includes_llm_deduced",
    *(f"estimate_rho_{r:g}" for r in SENSITIVITY_RHOS),
]


# ── Design declarations ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class FactorSpec:
    """One manipulated independent variable of a study."""

    name: str
    design: str
    columns: Mapping[str, Mapping[str, str] | None]
    canonical: str | None = None
    levels: Mapping[str, str] = field(default_factory=dict)
    reference: tuple[str, ...] = ()

    def level_series(self, df: pd.DataFrame) -> pd.Series:
        """Study level of every row; the first listed column that has one wins."""
        out = pd.Series(np.nan, index=df.index, dtype=object)
        for column, mapping in self.columns.items():
            if column not in df.columns:
                continue
            raw = df[column].map(_normalize_value)
            values = raw if mapping is None else raw.map(mapping)
            out = out.where(out.notna(), values)
        return out


@dataclass(frozen=True)
class StudyDesign:
    study: str
    participant: tuple[str, ...]
    factors: tuple[FactorSpec, ...] = ()
    trial: tuple[str, ...] = ()
    timepoint: tuple[str, ...] = ()
    source: str = ""

    def participant_series(self, df: pd.DataFrame) -> pd.Series | None:
        """Participant id of every row; the first listed column that has one
        wins (files of one study may spell the column differently)."""
        present = [c for c in self.participant if c in df.columns]
        if not present:
            return None
        out = pd.Series(np.nan, index=df.index, dtype=object)
        for column in present:
            out = out.where(out.notna(), df[column].map(_normalize_value))
        return out


@dataclass(frozen=True)
class DesignRegistry:
    version: str
    canonical_factors: Mapping[str, dict]
    studies: Mapping[str, StudyDesign]


def _normalize_value(value: Any) -> Any:
    """Canonical string form of a raw cell value used as a factor level."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if pd.isna(value):
        return np.nan
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        # Integer codes read as floats (columns with missing cells): 3.0 -> "3"
        return str(int(value))
    text = str(value).strip()
    return text if text else np.nan


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return (str(value),)


def _parse_factor(study: str, raw: dict, canonical_factors: Mapping[str, dict]) -> FactorSpec:
    name = str(raw.get("name") or "").strip()
    if not name:
        raise ValueError(f"{study}: every factor needs a name")
    design = str(raw.get("design", "")).strip()
    if design not in VALID_DESIGNS:
        raise ValueError(f"{study}/{name}: design must be one of {VALID_DESIGNS}, got {design!r}")
    columns_raw = raw.get("columns") or {}
    if not isinstance(columns_raw, dict) or not columns_raw:
        raise ValueError(f"{study}/{name}: 'columns' must map at least one column")
    columns = {
        str(col): (None if mapping is None else {str(k).strip(): str(v).strip() for k, v in mapping.items()})
        for col, mapping in columns_raw.items()
    }
    canonical = raw.get("canonical")
    levels = {str(k).strip(): str(v).strip() for k, v in (raw.get("levels") or {}).items()}
    reference = _as_tuple(raw.get("reference"))
    if canonical is not None:
        canonical = str(canonical)
        if canonical not in canonical_factors:
            raise ValueError(f"{study}/{name}: unknown canonical factor {canonical!r}")
        allowed = set(canonical_factors[canonical].get("levels") or [])
        if not levels:
            raise ValueError(f"{study}/{name}: 'levels' is required with 'canonical'")
        bad = sorted(set(levels.values()) - allowed)
        if bad:
            raise ValueError(f"{study}/{name}: levels {bad} not in canonical levels {sorted(allowed)}")
        if not reference:
            canonical_ref = str(canonical_factors[canonical].get("reference"))
            reference = tuple(lvl for lvl, canon in levels.items() if canon == canonical_ref)
    if not reference:
        raise ValueError(f"{study}/{name}: no reference level")
    return FactorSpec(
        name=name, design=design, columns=columns, canonical=canonical,
        levels=levels, reference=reference,
    )


def load_study_designs(path: Path = DEFAULT_DESIGNS_PATH) -> DesignRegistry:
    """Parse and validate ``study_designs.yaml``."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    canonical_factors = data.get("canonical_factors") or {}
    for fid, spec in canonical_factors.items():
        levels = spec.get("levels") or []
        if spec.get("reference") not in levels:
            raise ValueError(f"canonical factor {fid}: reference must be one of its levels")
    studies: dict[str, StudyDesign] = {}
    for study, raw in (data.get("studies") or {}).items():
        raw = raw or {}
        participant = _as_tuple(raw.get("participant"))
        if not participant:
            raise ValueError(f"{study}: 'participant' column is required")
        factors = tuple(_parse_factor(study, f, canonical_factors) for f in raw.get("factors") or [])
        names = [f.name for f in factors]
        if len(names) != len(set(names)):
            raise ValueError(f"{study}: duplicate factor names {names}")
        studies[str(study)] = StudyDesign(
            study=str(study),
            participant=participant,
            factors=factors,
            trial=_as_tuple(raw.get("trial")),
            timepoint=_as_tuple(raw.get("timepoint")),
            source=str(raw.get("source", "")).strip(),
        )
    return DesignRegistry(
        version=str(data.get("version", "")),
        canonical_factors=canonical_factors,
        studies=studies,
    )


# ── Effect sizes (metafor-equivalent) ────────────────────────────────────────

def small_sample_correction(m: np.ndarray | float) -> np.ndarray | float:
    """Exact Hedges correction c(m) = Γ(m/2) / (√(m/2) Γ((m-1)/2)) (metafor ``.cmicalc``)."""
    m = np.asarray(m, dtype=float)
    return np.exp(special.gammaln(m / 2) - 0.5 * np.log(m / 2) - special.gammaln((m - 1) / 2))


def smd_between(m1, m2, sd1, sd2, n1, n2) -> tuple[float, float]:
    """Hedges' g for two independent groups (metafor SMD, vtype LS)."""
    df = n1 + n2 - 2
    sp = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / df)
    yi = float(small_sample_correction(df) * (m1 - m2) / sp)
    vi = 1.0 / n1 + 1.0 / n2 + yi**2 / (2.0 * (n1 + n2))
    return yi, float(vi)


def smcrph_within(m1, m2, sd1, sd2, r, n) -> tuple[float, float]:
    """Standardized mean change, raw-score standardization with the average
    variance of both conditions (metafor SMCRPH, vtype LS; Bonett 2008)."""
    sdp2 = (sd1**2 + sd2**2) / 2.0
    sddiff2 = sd1**2 + sd2**2 - 2.0 * r * sd1 * sd2
    mi = 2.0 * (n - 1) / (1.0 + r**2)
    yi = float(small_sample_correction(mi) * (m1 - m2) / np.sqrt(sdp2))
    vi = sddiff2 / (sdp2 * (n - 1)) + yi**2 * (
        sd1**4 + sd2**4 + 2.0 * r**2 * sd1**2 * sd2**2
    ) / (8.0 * sdp2**2 * (n - 1))
    return yi, float(vi)


def aggregate_dependent(yi: Iterable[float], vi: Iterable[float], rho: float) -> tuple[float, float]:
    """GLS combination of dependent effects under compound symmetry
    (metafor ``aggregate(..., struct = "CS", rho = rho)``)."""
    y = np.asarray(list(yi), dtype=float)
    v = np.asarray(list(vi), dtype=float)
    if len(y) == 1:
        return float(y[0]), float(v[0])
    sd = np.sqrt(v)
    cov = rho * np.outer(sd, sd)
    np.fill_diagonal(cov, v)
    w = np.linalg.solve(cov, np.ones_like(y))
    var = 1.0 / float(np.sum(w))
    return float(var * np.dot(w, y)), var


# ── Contrasts per study ──────────────────────────────────────────────────────

def _observed_design(participant: pd.Series, level: pd.Series) -> str:
    """'within' when most participants were observed at more than one level."""
    frame = pd.DataFrame({"p": participant, "l": level}).dropna()
    if frame.empty:
        return "unknown"
    per_participant = frame.groupby("p")["l"].nunique()
    return "within" if float((per_participant > 1).mean()) >= 0.5 else "between"


def _favours(yi: float, direction: str) -> bool | None:
    if direction == "higher_is_better":
        return bool(yi > 0)
    if direction == "lower_is_better":
        return bool(yi < 0)
    return None


def _contrast_rows(
    participant_means: pd.DataFrame,
    reference: tuple[str, ...],
    design: str,
    min_n: int,
) -> list[dict]:
    """Effect of every non-reference level against the reference."""
    present_ref = [lvl for lvl in reference if lvl in participant_means.columns]
    if not present_ref:
        return []
    ref = participant_means[present_ref].mean(axis=1, skipna=True)
    rows = []
    for level in participant_means.columns:
        if level in reference:
            continue
        if design == "within":
            pair = pd.DataFrame({"t": participant_means[level], "c": ref}).dropna()
            n = len(pair)
            if n < min_n:
                continue
            sd1, sd2 = float(pair["t"].std(ddof=1)), float(pair["c"].std(ddof=1))
            if sd1 <= 0 or sd2 <= 0:
                continue
            r = float(np.corrcoef(pair["t"], pair["c"])[0, 1])
            m1, m2 = float(pair["t"].mean()), float(pair["c"].mean())
            yi, vi = smcrph_within(m1, m2, sd1, sd2, r, n)
            rows.append({
                "level": level, "measure": "SMCRPH", "n": n, "n_level": n, "n_reference": n,
                "mean_level": m1, "mean_reference": m2, "sd_level": sd1, "sd_reference": sd2,
                "r": r, "yi": yi, "vi": vi,
            })
        else:
            t = participant_means[level].dropna()
            c = ref.dropna()
            n1, n2 = len(t), len(c)
            if n1 < 2 or n2 < 2 or n1 + n2 < min_n:
                continue
            sd1, sd2 = float(t.std(ddof=1)), float(c.std(ddof=1))
            if sd1 <= 0 and sd2 <= 0:
                continue
            m1, m2 = float(t.mean()), float(c.mean())
            yi, vi = smd_between(m1, m2, sd1, sd2, n1, n2)
            rows.append({
                "level": level, "measure": "SMD", "n": n1 + n2, "n_level": n1, "n_reference": n2,
                "mean_level": m1, "mean_reference": m2, "sd_level": sd1, "sd_reference": sd2,
                "r": np.nan, "yi": yi, "vi": vi,
            })
    return rows


def compute_contrast_effects(
    studies: Mapping[str, pd.DataFrame],
    registry: DesignRegistry,
    mapping_provenance: Mapping[tuple, str] | None = None,
    min_n: int = MIN_PARTICIPANTS,
) -> pd.DataFrame:
    """Condition contrasts for every declared factor × canonical DV of every study."""
    provenance = mapping_provenance or {}
    dv_meta = msa._load_dv_measurement_metadata()
    rows: list[dict] = []
    for study, design in sorted(registry.studies.items()):
        df = studies.get(study)
        if df is None or df.empty or not design.factors:
            continue
        participant = design.participant_series(df)
        if participant is None:
            logger.warning("%s: none of the participant columns %s found", study, design.participant)
            continue
        canonical = msa._canonical_numeric_frame(df)
        for factor in design.factors:
            level = factor.level_series(df)
            if level.notna().sum() == 0:
                logger.warning("%s/%s: no rows carry a level of this factor", study, factor.name)
                continue
            observed = _observed_design(participant, level)
            if observed != factor.design:
                logger.warning(
                    "%s/%s: declared %s but the data look %s; using the observed design",
                    study, factor.name, factor.design, observed,
                )
            if factor.levels:
                unknown = sorted(set(level.dropna()) - set(factor.levels))
                if unknown:
                    logger.warning("%s/%s: levels %s are not declared and are ignored", study, factor.name, unknown)
                    level = level.where(level.isin(list(factor.levels)))
            for dv in canonical.columns:
                frame = pd.DataFrame({"p": participant, "l": level, "y": canonical[dv]}).dropna()
                if frame.empty or frame["l"].nunique() < 2:
                    continue
                means = frame.groupby(["p", "l"])["y"].mean().unstack("l")
                direction = dv_meta.get(dv, {}).get("direction", "neutral")
                for row in _contrast_rows(means, factor.reference, observed, min_n):
                    se = float(np.sqrt(row["vi"]))
                    rows.append({
                        "study": study,
                        "factor": factor.name,
                        "canonical_factor": factor.canonical,
                        "canonical_level": factor.levels.get(row["level"]) if factor.levels else None,
                        "reference": "+".join(factor.reference),
                        "dv": dv,
                        "design_declared": factor.design,
                        "design_observed": observed,
                        **row,
                        "se": se,
                        "ci95_low": row["yi"] - 1.96 * se,
                        "ci95_high": row["yi"] + 1.96 * se,
                        "direction": direction,
                        "favours_level": _favours(row["yi"], direction),
                        "mapping_source": provenance.get((study, dv), "unknown"),
                    })
    return pd.DataFrame(rows, columns=CONTRAST_EFFECT_COLUMNS)


# ── Synthesis across studies ─────────────────────────────────────────────────

def study_level_effects(effects: pd.DataFrame, rho: float = DEFAULT_RHO) -> pd.DataFrame:
    """One effect per study × canonical factor × DV (dependent contrasts combined)."""
    canonical = effects[effects["canonical_factor"].notna()]
    rows = []
    for (factor, dv, study), group in canonical.groupby(["canonical_factor", "dv", "study"], sort=True):
        yi, vi = aggregate_dependent(group["yi"], group["vi"], rho)
        sources = sorted(set(group["mapping_source"].astype(str)))
        rows.append({
            "canonical_factor": factor,
            "dv": dv,
            "study": study,
            "n_contrasts": len(group),
            "levels": "+".join(sorted(group["level"].astype(str))),
            "n": int(group["n"].max()),
            "measure": "+".join(sorted(set(group["measure"]))),
            "yi": yi,
            "vi": vi,
            "se": float(np.sqrt(vi)),
            "rho": rho if len(group) > 1 else np.nan,
            "mapping_source": "llm_deduced" if "llm_deduced" in sources else "; ".join(sources),
            "direction": group["direction"].iloc[0],
        })
    return pd.DataFrame(rows)


def pool_contrasts(
    effects: pd.DataFrame,
    estimator: str = "REML",
    rho: float = DEFAULT_RHO,
    sensitivity_rhos: Iterable[float] = SENSITIVITY_RHOS,
) -> pd.DataFrame:
    """Pool study-level contrast effects across studies (k >= 2)."""
    per_study = study_level_effects(effects, rho)
    if per_study.empty:
        return pd.DataFrame(columns=CONTRAST_POOLED_COLUMNS)
    sensitivity = {r: study_level_effects(effects, r) for r in sensitivity_rhos}
    rows = []
    for (factor, dv), group in per_study.groupby(["canonical_factor", "dv"], sort=True):
        if len(group) < 2:
            continue
        pooled = msa.pool_effects(group["yi"].to_numpy(), group["vi"].to_numpy(), estimator)
        row = {
            "canonical_factor": factor,
            "dv": dv,
            "k_studies": len(group),
            "studies": "; ".join(group["study"]),
            "n_total": int(group["n"].sum()),
            **pooled,
            "estimator": estimator,
            "rho": rho,
            "direction": group["direction"].iloc[0],
            "k_llm_deduced": int((group["mapping_source"] == "llm_deduced").sum()),
        }
        row["includes_llm_deduced"] = row["k_llm_deduced"] > 0
        for r, alt in sensitivity.items():
            sub = alt[(alt["canonical_factor"] == factor) & (alt["dv"] == dv)]
            row[f"estimate_rho_{r:g}"] = msa.pool_effects(
                sub["yi"].to_numpy(), sub["vi"].to_numpy(), estimator
            )["random_effects_mean"]
        rows.append(row)
    return pd.DataFrame(rows, columns=CONTRAST_POOLED_COLUMNS)


def run_contrast_analysis(
    studies: Mapping[str, pd.DataFrame],
    output_dir: Path,
    designs_path: Path = DEFAULT_DESIGNS_PATH,
    mapping_provenance: Mapping[tuple, str] | None = None,
    estimator: str = "REML",
    rho: float = DEFAULT_RHO,
    exclude_llm_deduced: bool = True,
) -> dict:
    """Write contrast_effects.csv, contrast_study_effects.csv and
    contrast_meta_analysis.csv (plus ``*_llm_excluded`` siblings)."""
    registry = load_study_designs(designs_path)
    effects = compute_contrast_effects(studies, registry, mapping_provenance)
    output_dir.mkdir(parents=True, exist_ok=True)
    effects.to_csv(output_dir / "contrast_effects.csv", index=False)
    study_effects = study_level_effects(effects, rho)
    study_effects.to_csv(output_dir / "contrast_study_effects.csv", index=False)
    pooled = pool_contrasts(effects, estimator=estimator, rho=rho)
    pooled.to_csv(output_dir / "contrast_meta_analysis.csv", index=False)
    summary = {
        "designs_version": registry.version,
        "studies_with_design": len(registry.studies),
        "studies_with_contrasts": int(effects["study"].nunique()) if not effects.empty else 0,
        "n_contrast_effects": len(effects),
        "n_pooled": len(pooled),
        "rho": rho,
        "estimator": estimator,
    }
    if exclude_llm_deduced:
        clean = effects[effects["mapping_source"] != "llm_deduced"]
        pool_contrasts(clean, estimator=estimator, rho=rho).to_csv(
            output_dir / "contrast_meta_analysis_llm_excluded.csv", index=False
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input-dir", required=True, help="Standardized study directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--designs", default=str(DEFAULT_DESIGNS_PATH))
    parser.add_argument("--meta-view", default=None, help="meta_view.csv (default: <input-dir>/../meta_view.csv)")
    parser.add_argument("--estimator", choices=["DL", "REML"], default="REML")
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    input_dir = Path(args.input_dir)
    meta_view = Path(args.meta_view) if args.meta_view else input_dir.parent / "meta_view.csv"
    studies = msa.load_studies(input_dir, skip_files_without_dvs=True)
    summary = run_contrast_analysis(
        studies,
        Path(args.output_dir),
        designs_path=Path(args.designs),
        mapping_provenance=msa.load_mapping_provenance(meta_view),
        estimator=args.estimator,
        rho=args.rho,
    )
    print(summary)


if __name__ == "__main__":
    main()
