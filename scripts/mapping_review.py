"""Human review of column mappings: queue, decisions, write-back with provenance.

Only about six in ten LLM-deduced mappings are correct
(``analyses/output_llm_evaluation/``), so they need a person to check them.
This module turns a batch run into a review queue, and writes the reviewer's
decisions back into the source's mapping YAML, where the next run applies
them like any curated mapping.

Queue (``build_review_queue``): one row per (source, raw column) that needs a
human, most urgent first.

========  ===============  ==================================================
priority  status           why it is in the queue
========  ===============  ==================================================
1         llm_accepted     mapped by the LLM and used in the analysis
2         llm_rejected     the LLM suggested a DV that the plausibility check
                           rejected; the column may still be a DV
3         unmapped         survey/results column nobody mapped
4         screened_out     kept from the LLM as demographic, ID, timing, ...
========  ===============  ==================================================

Within a priority, rows whose LLM suggestion disagrees with the top
definition-retrieval candidate, then rows with the smallest retrieval margin
(top-1 minus top-2 similarity), come first. ``outside_range`` flags columns
whose observed values cannot lie on the suggested DV's scale. Each row shows the observed range
and mean next to the suggested DV's canonical range, and the retrieval
candidates with their labels.

Decisions (column ``decision``):

* ``accept``: keep the LLM suggestion (``raw: suggestion``);
* ``map``: map to the DV in ``target`` (must be a schema DV id);
* ``non_dv``: not a dependent variable (``raw: meta_<name>``). The batch
  runner never sends it to the LLM, and ``evaluate_llm_mapping.py`` counts it
  as a curated non-DV label;
* ``study_specific``: a real outcome with no schema DV yet (``raw: target``
  or ``raw: <name>``); candidates for extending the schema;
* empty or ``skip``: undecided.

``apply_decisions`` writes each decision into the source's mapping YAML (the
manifest's ``mapping_path``, or a new ``schemas/<source_id>_mapping.yaml``
that it registers in the manifest), round-trip so comments survive, with an
end-of-line note of reviewer and date, and appends every decision to a review
log. Reviewed DV mappings become gold labels of the LLM benchmark once the
batch is rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMAS_DIR = REPO_ROOT / "schemas"
DEFAULT_DV_SCHEMA = SCHEMAS_DIR / "standard_dv_mapping.yaml"
DEFAULT_LOG = SCHEMAS_DIR / "mapping_review_log.csv"
DECISIONS = ("accept", "map", "non_dv", "study_specific", "skip")
REVIEWABLE_DATASET_TYPES = {"questionnaire", "results_table"}
STATUS_PRIORITY = {"llm_accepted": 1, "llm_rejected": 2, "unmapped": 3, "screened_out": 4}
N_CANDIDATES = 5

QUEUE_COLUMNS = [
    "priority", "status", "source_id", "raw_column", "datasets", "dataset_type",
    "n", "mean", "sd", "min", "max",
    "suggestion", "suggestion_label", "suggestion_range", "rejection_reason",
    "outside_range", "retrieval_top", "retrieval_margin", "agrees_with_retrieval", "candidates",
    "decision", "target", "note",
]
LOG_COLUMNS = [
    "timestamp", "reviewer", "source_id", "raw_column", "decision", "written_target",
    "previous_status", "previous_suggestion", "mapping_file", "note",
    "n", "mean", "sd", "min", "max", "candidates",
]


# ── Queue ────────────────────────────────────────────────────────────────────

def _dv_info(schema_path: Path) -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(schema_path.read_text(encoding="utf-8")) or {}
    return {
        str(dv["id"]): {
            "label": str(dv.get("label", dv["id"])),
            "range": (dv.get("measurement") or {}).get("canonical_range"),
        }
        for dv in data.get("dvs") or [] if isinstance(dv, dict) and dv.get("id")
    }


def _pooled_stats(group: pd.DataFrame) -> dict[str, Any]:
    """n, mean and SD over all datasets of a column (between-dataset spread included)."""
    g = group.dropna(subset=["n"])
    g = g[g["n"] > 0]
    if g.empty:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    n = g["n"].astype(float)
    total = float(n.sum())
    mean = float((n * g["mean"].fillna(0.0)).sum() / total)
    within = float(((n - 1).clip(lower=0) * g["sd"].fillna(0.0) ** 2).sum())
    between = float((n * (g["mean"].fillna(mean) - mean) ** 2).sum())
    sd = (within + between) / (total - 1) if total > 1 else None
    out = {"n": int(total), "mean": mean, "sd": sd ** 0.5 if sd is not None else None}
    for key, fn in (("min", "min"), ("max", "max")):
        out[key] = float(getattr(g[key], fn)()) if key in g and g[key].notna().any() else None
    return out


def _outside(stats: Mapping[str, Any], canonical_range: Any) -> bool:
    """True when the observed values cannot lie on the suggested DV's scale."""
    if not canonical_range:
        return False
    lo, hi = (float(v) for v in canonical_range)
    observed = [stats.get(k) for k in ("min", "max", "mean")]
    return any(v is not None and not pd.isna(v) and not lo <= float(v) <= hi for v in observed)


def _format_range(value: Any) -> str:
    if not value:
        return ""
    lo, hi = value
    return f"{lo:g}–{hi:g}"


def build_review_queue(batch_dir: Path, schema_path: Path = DEFAULT_DV_SCHEMA) -> pd.DataFrame:
    """Review queue for a batch output directory (see the module docstring)."""
    from scripts.dv_retrieval import get_retriever

    meta = pd.read_csv(batch_dir / "meta_view.csv", low_memory=False)
    meta["original_alias"] = meta["original_alias"].astype(str)
    for col in ("min", "max"):
        if col not in meta:
            meta[col] = float("nan")
    dvs = _dv_info(schema_path)
    retriever = get_retriever(str(schema_path))

    rejected_path = batch_dir / "llm_rejected_suggestions.json"
    rejections = json.loads(rejected_path.read_text(encoding="utf-8")) if rejected_path.is_file() else []
    by_rejection = {(r["source_id"], str(r["alias"])): r for r in rejections}

    entries: dict[tuple[str, str], dict[str, Any]] = {}
    llm_rows = meta[meta["mapping_method"] == "llm"]
    for (source, alias), group in llm_rows.groupby(["source_id", "original_alias"], sort=False):
        entries[(source, alias)] = {"status": "llm_accepted", "suggestion": group["canonical_dv"].iloc[0],
                                    "rows": group}
    for (source, alias), rej in by_rejection.items():
        if (source, alias) in entries:
            continue
        status = "llm_rejected" if rej.get("stage") == "post_llm" else "screened_out"
        rows = meta[(meta["source_id"] == source) & (meta["original_alias"] == alias)]
        entries[(source, alias)] = {"status": status, "suggestion": rej.get("proposed_canonical"),
                                    "reason": rej.get("reason"), "rows": rows, "rejection": rej}
    unmapped = meta[(meta["mapping_method"] == "unmapped") & meta["dataset_type"].isin(REVIEWABLE_DATASET_TYPES)]
    for (source, alias), group in unmapped.groupby(["source_id", "original_alias"], sort=False):
        entries.setdefault((source, alias), {"status": "unmapped", "suggestion": None, "rows": group})

    records = []
    for (source, alias), entry in entries.items():
        rows = entry["rows"]
        stats = _pooled_stats(rows) if not rows.empty else {"n": 0, "mean": None, "sd": None, "min": None,
                                                              "max": None}
        rejection = entry.get("rejection") or {}
        for key in ("min", "max"):
            if stats.get(key) is None and rejection.get(key) is not None:
                stats[key] = float(rejection[key])
        ranked = retriever.rank(alias, k=N_CANDIDATES)
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else None
        suggestion = entry.get("suggestion")
        suggestion = None if suggestion is None or (isinstance(suggestion, float) and pd.isna(suggestion)) else str(
            suggestion)
        info = dvs.get(suggestion or "", {})
        datasets = sorted(set(rows["dataset_id"].astype(str))) if not rows.empty else sorted(
            rejection.get("datasets") or [])
        records.append({
            "priority": STATUS_PRIORITY[entry["status"]],
            "status": entry["status"],
            "source_id": source,
            "raw_column": alias,
            "datasets": "; ".join(datasets),
            "dataset_type": "; ".join(sorted(set(rows["dataset_type"].astype(str)))) if not rows.empty else "",
            **stats,
            "suggestion": suggestion or "",
            "suggestion_label": info.get("label", ""),
            "suggestion_range": _format_range(info.get("range")),
            "outside_range": _outside(stats, info.get("range")),
            "rejection_reason": entry.get("reason") or "",
            "retrieval_top": ranked[0][0] if ranked else "",
            "retrieval_margin": round(margin, 4) if margin is not None else None,
            "agrees_with_retrieval": bool(suggestion) and bool(ranked) and suggestion == ranked[0][0],
            "candidates": "; ".join(
                f"{dv} ({dvs.get(dv, {}).get('label', dv)}"
                + (f", {_format_range(dvs[dv]['range'])}" if dvs.get(dv, {}).get("range") else "") + ")"
                for dv, _ in ranked
            ),
            "decision": "",
            "target": "",
            "note": "",
        })
    queue = pd.DataFrame(records, columns=QUEUE_COLUMNS)
    if queue.empty:
        return queue
    queue = queue.sort_values(
        ["priority", "agrees_with_retrieval", "retrieval_margin", "source_id", "raw_column"],
        na_position="first",
    ).reset_index(drop=True)
    return queue


# ── Decisions ────────────────────────────────────────────────────────────────

def _slug(name: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    return slug[:max_length].rstrip("_") or "column"


@dataclass(frozen=True)
class Decision:
    source_id: str
    raw_column: str
    decision: str
    target: str = ""
    note: str = ""
    previous_status: str = ""
    previous_suggestion: str = ""
    evidence: Mapping[str, Any] | None = None

    def mapped_name(self, dv_ids: Iterable[str]) -> str | None:
        """Name the raw column is mapped to, or ``None`` for no write-back."""
        known = set(dv_ids)
        if self.decision in ("", "skip"):
            return None
        if self.decision == "accept":
            if not self.previous_suggestion:
                raise ValueError(f"{self.source_id}/{self.raw_column}: 'accept' needs an LLM suggestion")
            return self.previous_suggestion
        if self.decision == "map":
            if self.target not in known:
                raise ValueError(f"{self.source_id}/{self.raw_column}: target {self.target!r} is not a schema DV")
            return self.target
        if self.decision == "non_dv":
            return self.target if self.target.startswith("meta_") else f"meta_{_slug(self.raw_column)}"
        if self.decision == "study_specific":
            name = _slug(self.target or self.raw_column)
            if name in known:
                raise ValueError(f"{self.source_id}/{self.raw_column}: {name!r} is a schema DV; use 'map'")
            return name
        raise ValueError(f"{self.source_id}/{self.raw_column}: unknown decision {self.decision!r} "
                         f"(expected one of {DECISIONS})")


def decisions_from_queue(queue: pd.DataFrame) -> list[Decision]:
    """Decisions from a reviewed queue (rows without a decision are skipped)."""
    out = []
    for row in queue.fillna("").to_dict("records"):
        decision = str(row.get("decision", "")).strip().lower()
        if decision in ("", "skip"):
            continue
        out.append(Decision(
            source_id=str(row["source_id"]),
            raw_column=str(row["raw_column"]),
            decision=decision,
            target=str(row.get("target", "")).strip(),
            note=str(row.get("note", "")).strip(),
            previous_status=str(row.get("status", "")),
            previous_suggestion=str(row.get("suggestion", "")),
            evidence={k: row.get(k) for k in ("n", "mean", "sd", "min", "max", "candidates")},
        ))
    return out


# ── Write-back ───────────────────────────────────────────────────────────────

def _round_trip_yaml():
    from ruamel.yaml import YAML

    y = YAML()  # round-trip: keeps comments, order and quoting
    y.preserve_quotes = True
    y.width = 4096
    return y


def _mapping_file_for(
    source: Mapping[str, Any], schemas_dir: Path, repository_mapping: str | None
) -> tuple[Path, bool]:
    """(mapping YAML of a source, whether the manifest must be updated to point at it)."""
    requested = source.get("mapping_path")
    if requested:
        path = Path(str(requested))
        return (path if path.is_absolute() else REPO_ROOT / path), False
    path = schemas_dir / f"{source['source_id']}_mapping.yaml"
    if not path.exists():
        header = (
            f"# Reviewed column mapping for {source['source_id']} (scripts/mapping_review.py).\n"
            f"# Format: raw column: canonical DV id; meta_* marks a column as not a DV.\n"
        )
        if repository_mapping and Path(repository_mapping).is_file():
            # Keep the repository's own mapping, which mapping_path will now replace.
            header += f"# Started from the repository mapping {Path(repository_mapping).name}.\n"
            path.write_text(header + Path(repository_mapping).read_text(encoding="utf-8"), encoding="utf-8")
        else:
            path.write_text(header + "{}\n", encoding="utf-8")
    return path, True


def apply_decisions(
    decisions: list[Decision],
    manifest_path: Path,
    reviewer: str,
    schemas_dir: Path = SCHEMAS_DIR,
    log_path: Path = DEFAULT_LOG,
    schema_path: Path = DEFAULT_DV_SCHEMA,
    batch_dir: Path | None = None,
) -> dict[str, Any]:
    """Write decisions into mapping YAMLs and the review log; return a summary."""
    if not reviewer.strip():
        raise ValueError("a reviewer name is required (it is recorded with every decision)")
    dv_ids = set(_dv_info(schema_path))
    resolved = [(d, d.mapped_name(dv_ids)) for d in decisions]  # validate everything before writing
    y = _round_trip_yaml()
    manifest = y.load(manifest_path.read_text(encoding="utf-8"))
    sources = {str(s["source_id"]): s for s in manifest.get("sources") or []}
    repository_mappings: dict[str, str] = {}
    if batch_dir is not None and (batch_dir / "meta_view.csv").is_file():
        mv = pd.read_csv(batch_dir / "meta_view.csv", usecols=["source_id", "source_mapping"], low_memory=False)
        for source_id, values in mv.dropna().groupby("source_id")["source_mapping"]:
            repository_mappings[str(source_id)] = str(values.iloc[0])

    today = datetime.now(UTC).date().isoformat()
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    written: dict[str, int] = {}
    manifest_changed = False
    log_rows = []
    by_file: dict[Path, list[tuple[Decision, str]]] = {}
    for decision, name in resolved:
        if decision.source_id not in sources:
            raise ValueError(f"source {decision.source_id!r} is not in {manifest_path.name}")
        if name is None:
            continue
        path, register = _mapping_file_for(
            sources[decision.source_id], schemas_dir, repository_mappings.get(decision.source_id)
        )
        if register:
            sources[decision.source_id]["mapping_path"] = path.relative_to(REPO_ROOT).as_posix() if \
                path.is_relative_to(REPO_ROOT) else str(path)
            manifest_changed = True
        by_file.setdefault(path, []).append((decision, name))

    for path, items in by_file.items():
        data = y.load(path.read_text(encoding="utf-8"))
        if data is None:
            from ruamel.yaml.comments import CommentedMap

            data = CommentedMap()
        for decision, name in items:
            data[decision.raw_column] = name
            comment = f"reviewed {today} by {reviewer}: {decision.decision}"
            if decision.previous_suggestion and decision.decision != "accept":
                comment += f" (LLM had {decision.previous_suggestion})"
            data.yaml_add_eol_comment(comment, decision.raw_column)
            evidence = decision.evidence or {}
            log_rows.append({
                "timestamp": stamp, "reviewer": reviewer, "source_id": decision.source_id,
                "raw_column": decision.raw_column, "decision": decision.decision, "written_target": name,
                "previous_status": decision.previous_status, "previous_suggestion": decision.previous_suggestion,
                "mapping_file": path.name, "note": decision.note,
                **{k: evidence.get(k, "") for k in ("n", "mean", "sd", "min", "max", "candidates")},
            })
        with open(path, "w", encoding="utf-8") as fh:
            y.dump(data, fh)
        written[path.name] = len(items)

    if manifest_changed:
        backup = manifest_path.with_suffix(manifest_path.suffix + ".bak")
        shutil.copyfile(manifest_path, backup)
        with open(manifest_path, "w", encoding="utf-8") as fh:
            y.dump(manifest, fh)
    if log_rows:
        new_file = not log_path.exists()
        with open(log_path, "a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerows(log_rows)
    return {"written": written, "decisions": len(log_rows), "manifest_updated": manifest_changed,
            "log": str(log_path)}


def review_summary(log_path: Path = DEFAULT_LOG) -> dict[str, Any]:
    """Counts by decision, and the share of LLM mappings reviewers kept.

    The share of ``llm_accepted`` rows that were accepted is a direct human
    estimate of the LLM's precision, to compare with the benchmark's.
    """
    if not log_path.is_file():
        return {"decisions": 0}
    log = pd.read_csv(log_path)
    latest = log.drop_duplicates(["source_id", "raw_column"], keep="last")
    llm = latest[latest["previous_status"] == "llm_accepted"]
    return {
        "decisions": int(len(latest)),
        "reviewers": sorted(latest["reviewer"].dropna().unique().tolist()),
        "by_decision": latest["decision"].value_counts().to_dict(),
        "llm_mappings_reviewed": int(len(llm)),
        "llm_mappings_kept": int((llm["decision"] == "accept").sum()),
        "llm_precision_by_review": (float((llm["decision"] == "accept").mean()) if len(llm) else None),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Review LLM and unmapped column mappings of a batch run.")
    sub = parser.add_subparsers(dest="command", required=True)
    q = sub.add_parser("queue", help="write the review queue as CSV")
    q.add_argument("--batch-dir", type=Path, required=True)
    q.add_argument("--output", type=Path, default=Path("review_queue.csv"))
    q.add_argument("--status", nargs="*", choices=list(STATUS_PRIORITY), help="only these statuses")
    a = sub.add_parser("apply", help="write decisions from a reviewed queue CSV into the mapping YAMLs")
    a.add_argument("--decisions", type=Path, required=True)
    a.add_argument("--manifest", type=Path, required=True)
    a.add_argument("--reviewer", required=True)
    a.add_argument("--batch-dir", type=Path, default=None,
                   help="batch output, to start new mapping files from a source's repository mapping")
    a.add_argument("--log", type=Path, default=DEFAULT_LOG)
    s = sub.add_parser("summary", help="summarize the review log")
    s.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args(argv)

    if args.command == "queue":
        queue = build_review_queue(args.batch_dir)
        if args.status:
            queue = queue[queue["status"].isin(args.status)]
        queue.to_csv(args.output, index=False)
        print(f"{len(queue)} column(s) to review -> {args.output}")
        print(queue["status"].value_counts().to_string())
    elif args.command == "apply":
        queue = pd.read_csv(args.decisions, keep_default_na=False)
        result = apply_decisions(decisions_from_queue(queue), args.manifest, args.reviewer,
                                 log_path=args.log, batch_dir=args.batch_dir)
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(review_summary(args.log), indent=2))


if __name__ == "__main__":
    main()
