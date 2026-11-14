"""Pin the files of the second held-out test before anything is downloaded.

Reads the datasets that qualified in ``check_decisions.csv`` (decision ``include``) and pins
their files with the rules of the first test (``build_sources.classify``):

- tables up to 50 MB, documentation up to 20 MB, zip archives up to 100 MB whole and only the
  chosen members of larger ones (HTTP range requests; the first test's limit was 1 GB);
- videos, code, raw signals, pickles, and folders of 15 or more tables (per-trial logs) are left out.

Two per-dataset columns of ``check_decisions.csv`` adjust the rules, both decided from the
documentation during the screening, before any download:

- ``pin``: glob patterns (``;``-separated) of files or archive members to fetch whatever the
  rules say: MATLAB ``.mat`` files, delimited ``.txt`` results files, tables in a folder the
  log-folder rule would skip, or a questionnaire PDF (pickles stay refused);
- ``skip``: glob patterns of files or members the rules would take but that hold no DV of the
  study (raw simulator exports over many MB, analysis outputs).

Writes ``analyses/heldout_2/sources.yaml`` (fetched and verified by
``python -m opendv.analysis.confirmatory fetch``) and ``analyses/heldout_2/file_selection.csv``.

    python -m analyses.candidate_search.build_sources_2 [--offline]
"""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import yaml

from analyses.candidate_search.build_sources import (
    DOCUMENTS,
    LOG_FOLDER_TABLES,
    classify,
    list_zip,
)
from analyses.candidate_search.search_open_data import UNSAFE, file_records

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DECISIONS = HERE / "check_decisions.csv"
OUTPUT = ROOT / "analyses" / "heldout_2" / "sources.yaml"
SELECTION_LOG = ROOT / "analyses" / "heldout_2" / "file_selection.csv"
DATA_DIR = "data/raw/heldout_2"
MAX_PINNED = 1_000_000_000  # a pinned .mat/.txt larger than this is refused: fetch it by hand if ever needed
# Archives up to this size are downloaded whole and checked against their checksum; from larger
# ones only the pinned members are fetched by HTTP range requests (1 GB in the first test, which
# downloaded several archives of 300-800 MB for one small table each).
MAX_WHOLE_ARCHIVE = 100_000_000


def _patterns(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [p.strip() for p in value.split(";") if p.strip()]


def _matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) or fnmatch.fnmatch(PurePosixPath(path).name, p) for p in patterns)


def _role(path: str, size: int | None, pin: list[str], skip: list[str]) -> tuple[str, str]:
    if _matches(path, skip):
        return "skip", "no DV of the study (screening)"
    role, reason = classify(path, size)
    if _matches(path, pin) and not path.lower().endswith(".zip"):
        if PurePosixPath(path).suffix.lower() in UNSAFE:
            return "skip", "loading can run code"
        if int(size or 0) > MAX_PINNED:
            return "skip", "pinned by the screening but over 1 GB"
        if PurePosixPath(path).suffix.lower() in DOCUMENTS - {".txt"}:
            return "documentation", "pinned by the screening"
        return "table", "pinned by the screening"
    return role, reason


def pin_dataset(row: pd.Series, offline: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(sources.yaml entry, one log row per file and archive member with its role)."""
    pin, skip = _patterns(row.get("pin")), _patterns(row.get("skip"))
    records, note = file_records(row["doi"], {"url": row["url"]}, offline)
    log: list[dict[str, Any]] = []
    for record in records:
        role, reason = _role(record["path"], record["size"], pin, skip)
        if role != "skip" and not (record.get("md5") or record.get("sha256")):
            role, reason = "skip", "no published checksum"
        if role != "skip" and not record.get("url"):
            role, reason = "skip", "no download URL"
        if role != "skip" and record.get("restricted"):
            role, reason = "skip", "restricted access"
        log.append({"doi": row["doi"], "file": record["path"], "member": "", "bytes": record["size"],
                    "role": role, "reason": reason, "record": record})
        if role != "archive":
            continue
        listing = list_zip(record["url"], offline)
        if listing["error"]:
            log[-1]["role"], log[-1]["reason"] = "skip", f"members not listed ({listing['error'][:80]})"
        for member in listing["members"]:
            m_role, m_reason = _role(member["path"], member["size"], pin, skip)
            if m_role == "archive":
                m_role, m_reason = "skip", "nested archive"
            log.append({"doi": row["doi"], "file": record["path"], "member": member["path"],
                        "bytes": member["size"], "role": m_role, "reason": m_reason, "record": member})
    # Per-trial and per-session logs: a folder holding many tables, unless the screening pinned them.
    folder_counts: dict[tuple[str, str], int] = {}
    for item in log:
        if item["role"] == "table":
            key = (item["file"] if item["member"] else "", str(PurePosixPath(item["member"] or item["file"]).parent))
            folder_counts[key] = folder_counts.get(key, 0) + 1
    for item in log:
        key = (item["file"] if item["member"] else "", str(PurePosixPath(item["member"] or item["file"]).parent))
        if (item["role"] == "table" and folder_counts[key] >= LOG_FOLDER_TABLES
                and item["reason"] != "pinned by the screening"):
            item["role"], item["reason"] = "skip", f"log folder ({folder_counts[key]} tables)"

    files: list[dict[str, Any]] = []
    for item in log:
        if item["member"] or item["role"] == "skip":
            continue
        record = item["record"]
        spec: dict[str, Any] = {"path": record["path"], "size": record["size"],
                                **{k: record[k] for k in ("md5", "sha256") if record.get(k)}, "url": record["url"]}
        if item["role"] == "documentation":
            spec["role"] = "codebook"
        if item["role"] == "archive":
            members = [m for m in log if m["member"] and m["file"] == record["path"] and m["role"] != "skip"]
            if not members:
                item["role"], item["reason"] = "skip", "no needed member"
                continue
            spec["fetch"] = "whole" if int(record["size"] or 0) <= MAX_WHOLE_ARCHIVE else "range"
            spec["members"] = [{"path": m["member"], "size": m["bytes"], "crc32": m["record"]["crc32"]}
                               for m in members]
        files.append(spec)
    entry: dict[str, Any] = {
        "source_id": row["source_id"],
        "title": row["title"],
        "venue": row["venue"],
        "doi": row["paper_doi"],
        "data_doi": row["doi"],
        "license": row["licence"],
        "repository": row["url"],
        "domain": row["domain"],
        "origin": row["origin"],
        "files": files,
    }
    if note:
        entry["listing_note"] = note
    for item in log:
        item.pop("record")
    return entry, log


def build(offline: bool = False) -> pd.DataFrame:
    decisions = pd.read_csv(DECISIONS)
    chosen = decisions[decisions["decision"] == "include"].sort_values("source_id")
    entries, logs = [], []
    for _, row in chosen.iterrows():
        entry, log = pin_dataset(row, offline)
        entries.append(entry)
        logs.extend({"source_id": entry["source_id"], **r} for r in log)
    header = (
        f"# Datasets of the second held-out test ({DECISIONS.relative_to(ROOT).as_posix()}, decision include).\n"
        "# Generated by `python -m analyses.candidate_search.build_sources_2`; each file is pinned by its size\n"
        "# and the checksum its repository publishes, each archive member by its size and CRC-32.\n"
        "# Download and verify with\n"
        f"#   python -m opendv.analysis.confirmatory fetch --sources {OUTPUT.relative_to(ROOT).as_posix()} \\\n"
        f"#     --data-dir {DATA_DIR}\n"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump({"sources": entries}, sort_keys=False, allow_unicode=True, width=120)
    OUTPUT.write_text(header + body, encoding="utf-8", newline="\n")
    frame = pd.DataFrame(logs)
    # Skipped archive members (Unity projects, ROS bags, raw logs: tens of thousands) are logged as
    # one row per archive, reason and extension, with their count; everything else row by row.
    members = frame["member"].fillna("").astype(str)
    skipped = (frame["role"] == "skip") & (members != "")
    collapsed = frame[skipped].assign(
        member=members[skipped].map(lambda m: f"*{PurePosixPath(m).suffix.lower() or ' (no suffix)'}"))
    collapsed = (collapsed.groupby(["source_id", "doi", "file", "member", "role", "reason"], dropna=False)
                 .agg(bytes=("bytes", "sum"), count=("bytes", "size")).reset_index())
    log = pd.concat([frame[~skipped].assign(count=1), collapsed], ignore_index=True)
    log = log[["source_id", "doi", "file", "member", "bytes", "count", "role", "reason"]]
    log.to_csv(SELECTION_LOG, index=False, lineterminator="\n")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true", help="use cached listings only")
    args = parser.parse_args()
    build(offline=args.offline)
    sources = yaml.safe_load(OUTPUT.read_text(encoding="utf-8"))["sources"]
    total = 0
    for source in sources:
        files, size = 0, 0
        for spec in source["files"]:
            members = spec.get("members") or []
            files += len(members) or 1
            if members and spec.get("fetch") == "range":
                size += sum(int(m["size"]) for m in members)
            else:
                size += int(spec["size"] or 0)
        total += size
        print(f"{source['source_id']:34s} {files:4d} files {size / 1e6:8.2f} MB")
    print(f"{len(sources)} datasets, {total / 1e6:.1f} MB to download")


if __name__ == "__main__":
    main()
