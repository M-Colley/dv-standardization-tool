"""Measurement invariance of shared scales across studies (multi-group CFA with lavaan).

Pooling a scale's scores across studies assumes it measures the same
construct in the same units everywhere. That is testable when the items are
available: a one-factor model is fitted per study at three nested levels
(Meredith, 1993; Vandenberg & Lance, 2000):

* **configural**: same factor structure in every study;
* **metric**: equal loadings, so relations with other variables are
  comparable;
* **scalar**: equal loadings and intercepts, so *means* are comparable.
  Pooling means across studies, as ``multi_study_analysis.py`` does,
  requires this level.

Items come from the instrument registry (``schemas/instruments.yaml``), with
reverse-keyed items reflected so every loading is positive. Studies
answer several times per participant, so participants (declared in
``schemas/study_designs.yaml``) are clusters: lavaan's MLR estimator with
``cluster=`` gives cluster-robust standard errors and scaled test
statistics. Likert items are treated as continuous.

Decision rules:

* configural fit is acceptable at robust CFI >= .90, RMSEA <= .08 and
  SRMR <= .08;
* a stricter level is rejected when CFI drops by .010 or more *and* RMSEA
  rises by .015 or more or SRMR by .030 (metric) / .010 (scalar)
  (Chen, 2007; Cheung & Rensvold, 2002).

The Satorra-Bentler scaled chi-square difference test is reported as well;
with a few hundred observations it rejects small, unimportant differences,
which is why the fit-index changes decide.

Fitting is done by lavaan through ``Rscript`` (``analyses/measurement_invariance.R``),
the reference implementation of these models; Python's SEM packages do
not fit multi-group models with cross-group equality constraints.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyses.contrast_meta_analysis import load_study_designs  # noqa: E402
from scripts import instruments as instruments_registry  # noqa: E402

logger = logging.getLogger(__name__)

R_SCRIPT = Path(__file__).with_suffix(".R")
LEVELS = ("configural", "metric", "scalar")
MIN_OBSERVATIONS = 30
MIN_CLUSTERS = 10
CONFIGURAL_MIN_CFI = 0.90
CONFIGURAL_MAX_RMSEA = 0.08
CONFIGURAL_MAX_SRMR = 0.08
DELTA_CFI = -0.010
DELTA_RMSEA = 0.015
DELTA_SRMR = {"metric": 0.030, "scalar": 0.010}

INVARIANCE_COLUMNS = [
    "instrument", "scale", "dv", "items", "studies", "n_obs", "n_clusters", "level",
    "cfi", "tli", "rmsea", "srmr", "chisq_scaled", "df", "delta_cfi", "delta_rmsea", "delta_srmr",
    "lrt_chisq_diff", "lrt_df_diff", "lrt_pvalue", "holds", "level_reached", "warning",
]


# ── R / lavaan ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def find_rscript() -> str | None:
    """Rscript on PATH, else the newest standard Windows install."""
    found = shutil.which("Rscript")
    if found:
        return found
    candidates = sorted(glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe"))
    return candidates[-1] if candidates else None


@lru_cache(maxsize=1)
def lavaan_available() -> bool:
    rscript = find_rscript()
    if rscript is None:
        return False
    probe = "quit(status = as.integer(!(requireNamespace('lavaan', quietly = TRUE) && "
    probe += "requireNamespace('jsonlite', quietly = TRUE))))"
    try:
        return subprocess.run([rscript, "-e", probe], capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def fit_levels(frame: pd.DataFrame, scale: str, items: list[str]) -> dict[str, Any]:
    """Fit configural/metric/scalar models with lavaan; return its JSON result."""
    rscript = find_rscript()
    if rscript is None:
        raise RuntimeError("Rscript not found")
    # lavaan model syntax needs plain identifiers; loadings are mapped back below.
    safe = {item: f"x{i + 1}" for i, item in enumerate(items)}
    with tempfile.TemporaryDirectory() as tmp:
        data_path, spec_path, out_path = (Path(tmp) / n for n in ("items.csv", "spec.json", "out.json"))
        frame.rename(columns=safe).to_csv(data_path, index=False)
        spec_path.write_text(json.dumps({"scale": scale, "items": list(safe.values())}), encoding="utf-8")
        result = subprocess.run(
            [rscript, str(R_SCRIPT), str(data_path), str(spec_path), str(out_path)],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0 or not out_path.is_file():
            raise RuntimeError(f"lavaan failed for {scale}: {result.stderr[-1500:]}")
        result = json.loads(out_path.read_text(encoding="utf-8"))
    original = {v: k for k, v in safe.items()}
    for entry in (result.get("levels") or {}).values():
        for loading in entry.get("std_loadings") or []:
            loading["item"] = original.get(loading["item"], loading["item"])
    result["items"] = items
    return result


# ── Data ─────────────────────────────────────────────────────────────────────

def scale_frames(
    studies: Mapping[str, pd.DataFrame],
    min_observations: int = MIN_OBSERVATIONS,
    min_clusters: int = MIN_CLUSTERS,
) -> list[tuple[instruments_registry.Instrument, instruments_registry.Score, pd.DataFrame]]:
    """Item data (group, cluster, items) for every scale observed in at least two studies."""
    designs = load_study_designs().studies
    out = []
    for inst in instruments_registry.load_instruments():
        for score in inst.scores:
            items = list(dict.fromkeys(score.items))
            blocks = []
            for study, df in sorted(studies.items()):
                resolved = instruments_registry._resolve_items(df, inst)
                if not all(i in resolved for i in items):
                    continue
                coding = instruments_registry.choose_coding(
                    [resolved[i] for i in items], inst.codings, label=f"{inst.id}/{study}"
                )
                if coding is None:
                    continue
                lo, hi = coding
                # Reflect reverse-keyed items so all loadings are positive.
                block = pd.DataFrame({
                    i: resolved[i] if inst.items[i].key == 1 else (lo + hi) - resolved[i] for i in items
                }).dropna()
                design = designs.get(study)
                participant = design.participant_series(df) if design else None
                cluster = (participant.loc[block.index] if participant is not None
                           else pd.Series(block.index, index=block.index)).astype(str)
                block = block[cluster.notna() & (cluster != "nan")]
                n_clusters = int(cluster.loc[block.index].nunique())
                if len(block) < min_observations or n_clusters < min_clusters:
                    continue
                block.insert(0, "cluster", study + ":" + cluster.loc[block.index])
                block.insert(0, "group", study)
                blocks.append(block)
            if len(blocks) >= 2:
                out.append((inst, score, pd.concat(blocks, ignore_index=True)))
    return out


# ── Decisions ────────────────────────────────────────────────────────────────

def _num(entry: Mapping[str, Any], key: str) -> float:
    value = entry.get(key)
    return float(value) if value is not None else float("nan")


def decide(result: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """One row per level and the highest level that holds."""
    rows, reached, previous = [], "none", None
    for level in LEVELS:
        entry = (result.get("levels") or {}).get(level) or {}
        row: dict[str, Any] = {"level": level, "warning": entry.get("warning") or entry.get("error", "")}
        if "error" in entry or not entry:
            row["holds"] = False
            rows.append(row)
            break
        row.update({
            "cfi": _num(entry, "cfi.robust"), "tli": _num(entry, "tli.robust"),
            "rmsea": _num(entry, "rmsea.robust"), "srmr": _num(entry, "srmr"),
            "chisq_scaled": _num(entry, "chisq.scaled"), "df": _num(entry, "df.scaled"),
            "lrt_chisq_diff": _num(entry, "lrt_chisq_diff"), "lrt_df_diff": _num(entry, "lrt_df_diff"),
            "lrt_pvalue": _num(entry, "lrt_pvalue"),
        })
        if level == "configural":
            holds = (row["cfi"] >= CONFIGURAL_MIN_CFI and row["rmsea"] <= CONFIGURAL_MAX_RMSEA
                     and row["srmr"] <= CONFIGURAL_MAX_SRMR)
        else:
            row["delta_cfi"] = row["cfi"] - previous["cfi"]
            row["delta_rmsea"] = row["rmsea"] - previous["rmsea"]
            row["delta_srmr"] = row["srmr"] - previous["srmr"]
            worse = row["delta_cfi"] <= DELTA_CFI and (
                row["delta_rmsea"] >= DELTA_RMSEA or row["delta_srmr"] >= DELTA_SRMR[level]
            )
            holds = bool(previous["holds"]) and not worse and bool(np.isfinite(row["cfi"]))
        row["holds"] = bool(holds)
        rows.append(row)
        if not holds:
            break
        reached, previous = level, row
    return rows, reached


def run_invariance(studies: Mapping[str, pd.DataFrame], output_dir: Path | None = None) -> pd.DataFrame:
    """Test every shared scale; write measurement_invariance.csv/.json when ``output_dir`` is given."""
    if not lavaan_available():
        logger.warning("Measurement invariance skipped: Rscript with lavaan and jsonlite not found")
        return pd.DataFrame(columns=INVARIANCE_COLUMNS)
    rows, raw = [], []
    for inst, score, frame in scale_frames(studies):
        items = [c for c in frame.columns if c not in ("group", "cluster")]
        try:
            result = fit_levels(frame, score.column, items)
        except RuntimeError as exc:
            logger.warning("%s", exc)
            continue
        raw.append(result)
        level_rows, reached = decide(result)
        for row in level_rows:
            rows.append({
                "instrument": inst.id, "scale": score.column, "dv": score.dv, "items": "; ".join(items),
                "studies": "; ".join(sorted(frame["group"].unique())), "n_obs": len(frame),
                "n_clusters": int(frame["cluster"].nunique()), **row, "level_reached": reached,
            })
    table = pd.DataFrame(rows, columns=INVARIANCE_COLUMNS)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_dir / "measurement_invariance.csv", index=False)
        (output_dir / "measurement_invariance.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return table


def invariance_by_dv(table: pd.DataFrame) -> dict[str, str]:
    """Level reached per DV: the scale's DV and each of its item DVs (lowest level if
    several), with the studies tested, e.g. ``"none (roads_chi25; running_into_traffic_chi26)"``.

    The test covers only studies that report every item of the scale, which
    can be fewer than the studies pooled for the DV.
    """
    order = {"none": 0, "configural": 1, "metric": 2, "scalar": 3}
    out: dict[str, tuple[str, str]] = {}
    for _, row in table.drop_duplicates("scale").iterrows():
        for dv in [row["dv"], *str(row["items"]).split("; ")]:
            level = row["level_reached"]
            if dv not in out or order[level] < order[out[dv][0]]:
                out[dv] = (level, row["studies"])
    return {dv: f"{level} ({studies})" for dv, (level, studies) in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input-dir", required=True, help="Standardized study directory")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from analyses.multi_study_analysis import load_studies

    table = run_invariance(load_studies(Path(args.input_dir), skip_files_without_dvs=True), Path(args.output_dir))
    with pd.option_context("display.width", 200):
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
